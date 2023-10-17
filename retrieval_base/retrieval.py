import os

from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np
import copy

import pymultinest

from .spectrum import DataSpectrum, Photometry
from .parameters import Parameters
from .pRT_model import pRT_model
from .log_likelihood import LogLikelihood
from .PT_profile import get_PT_profile_class
from .chemistry import get_Chemistry_class
from .covariance import get_Covariance_class
from .callback import CallBack

import retrieval_base.figures as figs
import retrieval_base.auxiliary_functions as af

def pre_processing(conf, conf_data):

    # Set up the output directories
    af.create_output_dir(conf.prefix, conf.file_params)

    # --- Pre-process data ----------------------------------------------

    # Get instances of the DataSpectrum class 
    # for the target and telluric standard
    d_spec = DataSpectrum(
        wave=None, 
        flux=None, 
        err=None, 
        ra=conf_data['ra'], 
        dec=conf_data['dec'], 
        mjd=conf_data['mjd'], 
        pwv=conf_data['pwv'], 
        file_target=conf_data['file_target'], 
        file_wave=conf_data['file_wave'], 
        slit=conf_data['slit'], 
        wave_range=conf_data['wave_range'], 
        w_set=conf_data['w_set'], 
        )
    d_spec.clip_det_edges()
    
    d_std_spec = DataSpectrum(
        wave=None, 
        flux=None, 
        err=None, 
        ra=conf_data['ra_std'], 
        dec=conf_data['dec_std'], 
        mjd=conf_data['mjd_std'], 
        pwv=conf_data['pwv'], 
        file_target=conf_data['file_std'], 
        file_wave=conf_data['file_std'], 
        slit=conf_data['slit'], 
        wave_range=conf_data['wave_range'], 
        w_set=conf_data['w_set'], 
        )
    d_std_spec.clip_det_edges()

    # Instance of the Photometry class for the given magnitudes
    photom_2MASS = Photometry(magnitudes=conf.magnitudes)

    # Get transmission from telluric std and add to target's class
    #d_std_spec.get_transmission(T=conf.T_std, ref_rv=0, mode='bb')
    d_std_spec.get_transmission(
        T=conf_data['T_std'], log_g=conf_data['log_g_std'], 
        ref_rv=conf_data['rv_std'], ref_vsini=conf_data['vsini_std'], 
        #mode='bb'
        mode='PHOENIX'
        )
    
    # Interpolate onto the same wavelength grid as the target
    d_std_spec.transm = np.interp(
        d_spec.wave, d_std_spec.wave, d_std_spec.transm
        )
    d_std_spec.transm_err = np.interp(
        d_spec.wave, d_std_spec.wave, d_std_spec.transm_err
        )
    d_spec.add_transmission(d_std_spec.transm, d_std_spec.transm_err)
    del d_std_spec

    # Apply flux calibration using the 2MASS broadband magnitude
    d_spec.flux_calib_2MASS(
        photom_2MASS, 
        conf_data['filter_2MASS'], 
        tell_threshold=conf_data['tell_threshold'], 
        prefix=conf.prefix, 
        file_skycalc_transm=conf_data['file_skycalc_transm'], 
        )
    del photom_2MASS

    # Apply sigma-clipping
    d_spec.sigma_clip_median_filter(
        sigma=3, 
        filter_width=conf_data['sigma_clip_width'], 
        prefix=conf.prefix
        )
    #d_spec.sigma_clip_poly(sigma=3, prefix=conf.prefix)

    # Crop the spectrum
    d_spec.crop_spectrum()

    # Remove the ghost signatures
    d_spec.mask_ghosts()

    # Re-shape the spectrum to a 3-dimensional array
    d_spec.reshape_orders_dets()

    # Apply barycentric correction
    d_spec.bary_corr()

    if conf.apply_high_pass_filter:
        # Apply high-pass filter
        d_spec.high_pass_filter(
            removal_mode='divide', 
            filter_mode='gaussian', 
            sigma=300, 
            replace_flux_err=True
            )

    # Prepare the wavelength separation and average squared error arrays
    d_spec.prepare_for_covariance(
        prepare_err_eff=conf.prepare_for_covariance
        )

    # Plot the pre-processed spectrum
    figs.fig_spec_to_fit(
        d_spec, prefix=conf.prefix, w_set=d_spec.w_set
        )

    # Save the data
    np.save(conf.prefix+f'data/d_spec_wave_{d_spec.w_set}.npy', d_spec.wave)
    np.save(conf.prefix+f'data/d_spec_flux_{d_spec.w_set}.npy', d_spec.flux)
    np.save(conf.prefix+f'data/d_spec_err_{d_spec.w_set}.npy', d_spec.err)
    np.save(conf.prefix+f'data/d_spec_transm_{d_spec.w_set}.npy', d_spec.transm)

    np.save(conf.prefix+f'data/d_spec_flux_uncorr_{d_spec.w_set}.npy', d_spec.flux_uncorr)
    del d_spec.flux_uncorr
    del d_spec.transm, d_spec.transm_err

    # Save as pickle
    af.pickle_save(conf.prefix+f'data/d_spec_{d_spec.w_set}.pkl', d_spec)

    # --- Set up a pRT model --------------------------------------------

    # Create the Radtrans objects
    pRT_atm = pRT_model(
        line_species=conf.line_species, 
        d_spec=d_spec, 
        mode='lbl', 
        lbl_opacity_sampling=conf_data['lbl_opacity_sampling'], 
        cloud_species=conf.cloud_species, 
        rayleigh_species=['H2', 'He'], 
        continuum_opacities=['H2-H2', 'H2-He'], 
        log_P_range=(-6,2), 
        n_atm_layers=50, 
        rv_range=conf.free_params['rv'][0], 
        )

    # Save as pickle
    af.pickle_save(conf.prefix+f'data/pRT_atm_{d_spec.w_set}.pkl', pRT_atm)

class Retrieval:

    def __init__(self, conf, evaluation):

        self.conf = conf
        self.evaluation = evaluation

        self.d_spec  = {}
        self.pRT_atm = {}
        param_wlen_settings = {}
        for w_set in conf.config_data.keys():

            # Load the DataSpectrum and pRT_model classes
            self.d_spec[w_set]  = af.pickle_load(self.conf.prefix+f'data/d_spec_{w_set}.pkl')
            self.pRT_atm[w_set] = af.pickle_load(self.conf.prefix+f'data/pRT_atm_{w_set}.pkl')

            param_wlen_settings[w_set] = [self.d_spec[w_set].n_orders, self.d_spec[w_set].n_dets]

        # Create a Parameters instance
        self.Param = Parameters(
            free_params=self.conf.free_params, 
            constant_params=self.conf.constant_params, 
            PT_mode=self.conf.PT_mode, 
            n_T_knots=self.conf.n_T_knots, 
            enforce_PT_corr=self.conf.enforce_PT_corr, 
            chem_mode=self.conf.chem_mode, 
            cloud_mode=self.conf.cloud_mode, 
            cov_mode=self.conf.cov_mode, 
            wlen_settings=param_wlen_settings, 
            #n_orders=self.d_spec.n_orders, 
            #n_dets=self.d_spec.n_dets, 
            )
        
        self.Cov     = {}
        self.LogLike = {}
        for w_set in conf.config_data.keys():
            
            # Update the cloud/chemistry-mode
            self.pRT_atm[w_set].cloud_mode = self.Param.cloud_mode
            self.pRT_atm[w_set].chem_mode  = self.Param.chem_mode

            self.Cov[w_set] = np.empty(
                (self.d_spec[w_set].n_orders, self.d_spec[w_set].n_dets), dtype=object
                )
            for i in range(self.d_spec[w_set].n_orders):
                for j in range(self.d_spec[w_set].n_dets):
                    
                    # Select only the finite pixels
                    mask_ij = self.d_spec[w_set].mask_isfinite[i,j]

                    if not mask_ij.any():
                        continue

                    self.Cov[w_set][i,j] = get_Covariance_class(
                        self.d_spec[w_set].err[i,j,mask_ij], 
                        self.Param.cov_mode, 
                        separation=self.d_spec[w_set].separation[i,j], 
                        err_eff=self.d_spec[w_set].err_eff[i,j], 
                        flux_eff=self.d_spec[w_set].flux_eff[i,j], 
                        max_separation=self.conf.GP_max_separation, 
                        )

            del self.d_spec[w_set].separation, 
            del self.d_spec[w_set].err_eff, 
            del self.d_spec[w_set].flux_eff
            del self.d_spec[w_set].err

            self.LogLike[w_set] = LogLikelihood(
                self.d_spec[w_set], 
                n_params=self.Param.n_params, 
                scale_flux=self.conf.scale_flux, 
                scale_err=self.conf.scale_err, 
                )

        self.PT = get_PT_profile_class(
            self.pRT_atm[w_set].pressure, 
            self.Param.PT_mode, 
            conv_adiabat=True, 
            ln_L_penalty_order=self.conf.ln_L_penalty_order, 
            PT_interp_mode=self.conf.PT_interp_mode, 
            )
        self.Chem = get_Chemistry_class(
            self.pRT_atm[w_set].line_species, 
            self.pRT_atm[w_set].pressure, 
            self.Param.chem_mode, 
            spline_order=self.conf.chem_spline_order, 
            )
        
        self.CB = CallBack(
            d_spec=self.d_spec, 
            evaluation=self.evaluation, 
            n_samples_to_use=2000, 
            prefix=self.conf.prefix, 
            posterior_color='C0', 
            bestfit_color='C1', 
            species_to_plot_VMR=self.conf.species_to_plot_VMR, 
            species_to_plot_CCF=self.conf.species_to_plot_CCF, 
            )

        if (rank == 0) and self.evaluation:
            self.pRT_atm_broad = {}
            for w_set in conf.config_data.keys():
                
                if os.path.exists(self.conf.prefix+f'data/pRT_atm_broad_{w_set}.pkl'):

                    # Load the pRT model
                    self.pRT_atm_broad[w_set] = af.pickle_load(
                        self.conf.prefix+f'data/pRT_atm_broad_{w_set}.pkl'
                        )
                    continue

                # Create a wider pRT model during evaluation
                self.pRT_atm_broad[w_set] = copy.deepcopy(self.pRT_atm[w_set])
                self.pRT_atm_broad[w_set].get_atmospheres(CB_active=True)

                # Save for convenience
                af.pickle_save(
                    self.conf.prefix+f'data/pRT_atm_broad_{w_set}.pkl', 
                    self.pRT_atm_broad[w_set]
                    )

        # Set to None initially, changed during evaluation
        self.m_spec_species  = None
        self.pRT_atm_species = None
        self.LogLike_species = None

    def PMN_lnL_func(self, cube=None, ndim=None, nparams=None):

        time_A = time.time()

        # Param.params dictionary is already updated

        # Retrieve the temperatures
        try:
            temperature = self.PT(self.Param.params)
        except:
            # Something went wrong with interpolating
            temperature = self.PT(self.Param.params)
            return -np.inf

        if temperature.min() < 0:
            # Negative temperatures are rejected
            return -np.inf

        # Retrieve the ln L penalty (=0 by default)
        ln_L_penalty = 0
        if hasattr(self.PT, 'ln_L_penalty'):
            ln_L_penalty = self.PT.ln_L_penalty

        # Retrieve the chemical abundances
        if self.Param.chem_mode == 'free':
            mass_fractions = self.Chem(self.Param.VMR_species, self.Param.params)
        elif self.Param.chem_mode == 'eqchem':
            mass_fractions = self.Chem(self.Param.params, temperature)

        if not isinstance(mass_fractions, dict):
            # Non-H2 abundances added up to > 1
            return -np.inf

        if self.CB.return_PT_mf:
            # Return temperatures and mass fractions during evaluation
            return (temperature, mass_fractions)

        self.m_spec = {}
        ln_L = ln_L_penalty
        for h, w_set in enumerate(list(self.conf.config_data.keys())):
            
            pRT_atm_to_use = self.pRT_atm[w_set]
            if self.evaluation:
                # Retrieve the model spectrum, with the wider pRT model
                pRT_atm_to_use = self.pRT_atm_broad[w_set]
        
            # Retrieve the model spectrum
            self.m_spec[w_set] = pRT_atm_to_use(
                mass_fractions, 
                temperature, 
                self.Param.params, 
                get_contr=self.CB.active, 
                get_full_spectrum=self.evaluation, 
                )
        
            if (self.m_spec[w_set].flux <= 0).any() or \
                (~np.isfinite(self.m_spec[w_set].flux)).any():
                # Something is wrong in the spectrum
                return -np.inf

            for i in range(self.d_spec[w_set].n_orders):
                for j in range(self.d_spec[w_set].n_dets):

                    if not self.d_spec[w_set].mask_isfinite[i,j].any():
                        continue

                    # Update the covariance matrix
                    self.Cov[w_set][i,j](
                        self.Param.params, 
                        w_set, 
                        order=i, 
                        det=j, 
                        trunc_dist=self.conf.GP_trunc_dist, 
                        scale_GP_amp=self.conf.scale_GP_amp
                        )

            # Retrieve the log-likelihood
            ln_L += self.LogLike[w_set](
                self.m_spec[w_set], 
                self.Cov[w_set], 
                is_first_w_set=(h==0), 
                #ln_L_penalty=ln_L_penalty, 
                evaluation=self.evaluation, 
                )
        
        time_B = time.time()
        self.CB.elapsed_times.append(time_B-time_A)

        return ln_L

    def get_PT_mf_envelopes(self, posterior):

        # Return the PT profile and mass fractions
        self.CB.return_PT_mf = True

        # Objects to store the envelopes in
        self.Chem.mass_fractions_posterior = {}
        self.Chem.unquenched_mass_fractions_posterior = {}

        self.Chem.CO_posterior  = []
        self.Chem.FeH_posterior = []

        self.PT.temperature_envelopes = []

        # Sample envelopes from the posterior
        for i, params_i in enumerate(posterior):

            for j, key_j in enumerate(self.Param.param_keys):
                # Update the Parameters instance
                self.Param.params[key_j] = params_i[j]

                if key_j.startswith('log_'):
                    self.Param.params = self.Param.log_to_linear(self.Param.params, key_j)

                if key_j.startswith('invgamma_'):
                    self.Param.params[key_j.replace('invgamma_', '')] = self.Param.params[key_j]

                if key_j.startswith('gaussian_'):
                    self.Param.params[key_j.replace('gaussian_', '')] = self.Param.params[key_j]
                    
            # Update the parameters
            self.Param.read_PT_params(cube=None)
            self.Param.read_uncertainty_params()
            self.Param.read_chemistry_params()
            self.Param.read_cloud_params()

            # Class instances with best-fitting parameters
            returned = self.PMN_lnL_func()

            if i == 0:
                for line_species_i in self.Chem.mass_fractions.keys():
                    self.Chem.mass_fractions_posterior[line_species_i] = []

                if hasattr(self.Chem, 'unquenched_mass_fractions'):
                    for line_species_i in self.Chem.unquenched_mass_fractions.keys():
                        self.Chem.unquenched_mass_fractions_posterior[line_species_i] = []

            if isinstance(returned, float):
                # PT profile or mass fractions failed
                continue

            # Store the temperatures and mass fractions
            temperature_i, mass_fractions_i = returned
            self.PT.temperature_envelopes.append(temperature_i)
            # Loop over the line species
            for line_species_i in self.Chem.mass_fractions.keys():
                self.Chem.mass_fractions_posterior[line_species_i].append(
                    mass_fractions_i[line_species_i]
                    )

            # Store the C/O ratio and Fe/H
            self.Chem.CO_posterior.append(self.Chem.CO)
            self.Chem.FeH_posterior.append(self.Chem.FeH)

            if hasattr(self.Chem, 'unquenched_mass_fractions'):
                # Store the unquenched mass fractions
                for line_species_i in self.Chem.unquenched_mass_fractions.keys():
                    self.Chem.unquenched_mass_fractions_posterior[line_species_i].append(
                        self.Chem.unquenched_mass_fractions[line_species_i]
                        )

        # Convert profiles to 1, 2, 3-sigma equivalent and median
        q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
             0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
             ]            

        # Retain the pressure-axis
        self.PT.temperature_envelopes = af.quantiles(
            np.array(self.PT.temperature_envelopes), q=q, axis=0
            )

        self.Chem.mass_fractions_envelopes = {}
        #for line_species_i in self.Chem.line_species:
        for line_species_i in self.Chem.mass_fractions.keys():

            self.Chem.mass_fractions_posterior[line_species_i] = \
                np.array(self.Chem.mass_fractions_posterior[line_species_i])

            self.Chem.mass_fractions_envelopes[line_species_i] = af.quantiles(
                self.Chem.mass_fractions_posterior[line_species_i], q=q, axis=0
                )
        
        self.Chem.CO_posterior  = np.array(self.Chem.CO_posterior)
        self.Chem.FeH_posterior = np.array(self.Chem.FeH_posterior)

        self.Chem.unquenched_mass_fractions_envelopes = {}
        if hasattr(self.Chem, 'unquenched_mass_fractions'):
            # Store the unquenched mass fractions
            for line_species_i in self.Chem.unquenched_mass_fractions.keys():
                self.Chem.unquenched_mass_fractions_posterior[line_species_i] = \
                    np.array(self.Chem.unquenched_mass_fractions_posterior[line_species_i])

                self.Chem.unquenched_mass_fractions_envelopes[line_species_i] = af.quantiles(
                    self.Chem.unquenched_mass_fractions_posterior[line_species_i], q=q, axis=0
                    )

        self.CB.return_PT_mf = False

    def get_species_contribution(self):

        #self.m_spec_species, self.pRT_atm_species = {}, {}

        self.m_spec_species = dict.fromkeys(self.d_spec.keys(), {})
        self.pRT_atm_species = dict.fromkeys(self.d_spec.keys(), {})
        '''
        # Ignore all species
        for species_j, (line_species_j, _, _) in self.Chem.species_info.items():
            if line_species_j in self.Chem.line_species:
                self.Chem.neglect_species[species_j] = True
        # Create the spectrum and evaluate lnL
        self.PMN_lnL_func()
        m_spec_continuum = np.copy(self.m_spec.flux)
        pRT_atm_continuum = self.pRT_atm_broad.flux_pRT_grid.copy()
        '''

        # Assess the species' contribution
        for species_i in self.Chem.species_info:

            line_species_i = self.Chem.read_species_info(species_i, 'pRT_name')
            if line_species_i not in self.Chem.line_species:
                continue

            '''
            # Ignore all other species
            self.Chem.neglect_species = dict.fromkeys(self.Chem.neglect_species, True)
            self.Chem.neglect_species[species_i] = False
            
            # Create the spectrum and evaluate lnL
            self.PMN_lnL_func()

            flux_only = np.copy(self.m_spec.flux)
            self.pRT_atm_broad.flux_pRT_grid_only = [
                self.pRT_atm_broad.flux_pRT_grid[i].copy() \
                for i in range(self.d_spec.n_orders)
                ]
            
            for species_j in self.Chem.species_info:
                self.Chem.neglect_species[species_j] = False
            '''
            # Ignore one species at a time
            self.Chem.neglect_species = dict.fromkeys(self.Chem.neglect_species, False)
            self.Chem.neglect_species[species_i] = True

            # Create the spectrum and evaluate lnL
            self.PMN_lnL_func()

            #self.m_spec.flux_only = flux_only
            for w_set in self.d_spec.keys():
                self.m_spec_species[w_set][species_i]  = copy.deepcopy(self.m_spec[w_set])
                self.pRT_atm_species[w_set][species_i] = copy.deepcopy(self.pRT_atm_broad[w_set])

        # Include all species again
        self.Chem.neglect_species = dict.fromkeys(self.Chem.neglect_species, False)

    def get_all_spectra(self, posterior, save_spectra=False):

        if os.path.exists(self.conf.prefix+'data/m_flux_envelope.npy'):
            
            # Load the model spectrum envelope if it was computed before
            flux_envelope = np.load(self.conf.prefix+'data/m_flux_envelope.npy')

            # Convert envelopes to 1, 2, 3-sigma equivalent and median
            q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
                    0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
                    ]            
        
            # Retain the order-, detector-, and wavelength-axes
            flux_envelope = af.quantiles(
                np.array(flux_envelope), q=q, axis=0
                )
                
            return flux_envelope
        
        from tqdm import tqdm
        self.evaluation = False

        flux_envelope = np.nan * np.ones(
            (len(posterior), self.d_spec.n_orders, 
            self.d_spec.n_dets, self.d_spec.n_pixels)
            )
        ln_L_per_pixel_posterior = np.nan * np.ones(
            (len(posterior), self.d_spec.n_orders, 
            self.d_spec.n_dets, self.d_spec.n_pixels)
            )
        chi_squared_per_pixel_posterior = np.nan * np.ones(
            (len(posterior), self.d_spec.n_orders, 
            self.d_spec.n_dets, self.d_spec.n_pixels)
            )

        # Sample envelopes from the posterior
        for i, params_i in enumerate(tqdm(posterior)):

            for j, key_j in enumerate(self.Param.param_keys):
                # Update the Parameters instance
                self.Param.params[key_j] = params_i[j]

            # Update the parameters
            self.Param.read_PT_params(cube=None)
            self.Param.read_uncertainty_params()
            self.Param.read_chemistry_params()
            self.Param.read_cloud_params()

            # Create the spectrum
            self.PMN_lnL_func()

            ln_L_per_pixel_posterior[i,:,:,:]        = np.copy(self.LogLike.ln_L_per_pixel)
            chi_squared_per_pixel_posterior[i,:,:,:] = np.copy(self.LogLike.chi_squared_per_pixel)
            
            if not save_spectra:
                continue

            # Scale the model flux with the linear parameter
            flux_envelope[i,:,:,:] = self.m_spec.flux * self.LogLike.f[:,:,None]

            # Add a random sample from the covariance matrix
            for k in range(self.d_spec.n_orders):
                for l in range(self.d_spec.n_dets):

                    # Get the covariance matrix
                    cov_kl = self.LogLike.cov[k,l].get_dense_cov()

                    # Scale with the optimal uncertainty scaling
                    cov_kl *= self.LogLike.beta[k,l].beta**2

                    # Draw a random sample and add to the flux
                    random_sample = np.random.multivariate_normal(
                        np.zeros(len(cov_kl)), cov_kl, size=1
                        )
                    flux_envelope[i,k,l,:] += random_sample[0]
        
        self.evaluation = True

        np.save(self.conf.prefix+'data/ln_L_per_pixel_posterior.npy', ln_L_per_pixel_posterior)
        np.save(self.conf.prefix+'data/chi_squared_per_pixel_posterior.npy', chi_squared_per_pixel_posterior)
    
        if save_spectra:
            # Save the model spectrum envelope
            np.save(self.conf.prefix+'data/m_flux_envelope.npy', flux_envelope)

            # Convert envelopes to 1, 2, 3-sigma equivalent and median
            q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
                 0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
                 ]            

            # Retain the order-, detector-, and wavelength-axes
            flux_envelope = af.quantiles(
                np.array(flux_envelope), q=q, axis=0
                )

            return flux_envelope

    def PMN_callback_func(self, 
                          n_samples, 
                          n_live, 
                          n_params, 
                          live_points, 
                          posterior, 
                          stats,
                          max_ln_L, 
                          ln_Z, 
                          ln_Z_err, 
                          nullcontext
                          ):

        self.CB.active = True

        if self.evaluation:

            # Set-up analyzer object
            analyzer = pymultinest.Analyzer(
                n_params=self.Param.n_params, 
                outputfiles_basename=self.conf.prefix
                )
            stats = analyzer.get_stats()

            # Load the equally-weighted posterior distribution
            posterior = analyzer.get_equal_weighted_posterior()
            posterior = posterior[:,:-1]

            # Read the parameters of the best-fitting model
            bestfit_params = np.array(stats['modes'][0]['maximum a posterior'])

            # Get the PT and mass-fraction envelopes
            self.get_PT_mf_envelopes(posterior)
            
            # Get the model flux envelope
            #flux_envelope = self.get_all_spectra(posterior, save_spectra=False)

        else:

            # Read the parameters of the best-fitting model
            bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2]

            # Remove the last 2 columns
            posterior = posterior[:,:-2]

        # Evaluate the model with best-fitting parameters
        for i, key_i in enumerate(self.Param.param_keys):
            # Update the Parameters instance
            self.Param.params[key_i] = bestfit_params[i]
        
            if key_i.startswith('log_'):
                self.Param.params = self.Param.log_to_linear(self.Param.params, key_i)

            if key_i.startswith('invgamma_'):
                self.Param.params[key_i.replace('invgamma_', '')] = self.Param.params[key_i]

        # Update the parameters
        self.Param.read_PT_params(cube=None)
        self.Param.read_uncertainty_params()
        self.Param.read_chemistry_params()
        self.Param.read_cloud_params()

        if self.evaluation:
            # Get each species' contribution to the spectrum
            self.get_species_contribution()

        # Update class instances with best-fitting parameters
        self.PMN_lnL_func()
        self.CB.active = False

        for w_set in self.conf.config_data.keys():
            self.m_spec[w_set].flux_envelope = None

        pRT_atm_to_use = self.pRT_atm
        if self.evaluation:
            # Retrieve the model spectrum, with the wider pRT model
            pRT_atm_to_use = self.pRT_atm_broad
            #self.m_spec.flux_envelope = flux_envelope

        # Call the CallBack class and make summarizing figures
        self.CB(
            self.Param, self.LogLike, self.Cov, self.PT, self.Chem, 
            self.m_spec, pRT_atm_to_use, posterior, 
            m_spec_species=self.m_spec_species, 
            pRT_atm_species=self.pRT_atm_species
            )

    def PMN_run(self):
        
        # Pause the process to not overload memory
        time.sleep(1.5*rank*len(self.d_spec))

        # Run the MultiNest retrieval
        pymultinest.run(
            LogLikelihood=self.PMN_lnL_func, 
            Prior=self.Param, 
            n_dims=self.Param.n_params, 
            outputfiles_basename=self.conf.prefix, 
            resume=True, 
            verbose=True, 
            const_efficiency_mode=self.conf.const_efficiency_mode, 
            sampling_efficiency=self.conf.sampling_efficiency, 
            n_live_points=self.conf.n_live_points, 
            evidence_tolerance=self.conf.evidence_tolerance, 
            dump_callback=self.PMN_callback_func, 
            n_iter_before_update=self.conf.n_iter_before_update, 
            )

    def synthetic_spectrum(self):
        
        # Update the parameters
        synthetic_params = np.array([
                0.8, # R_p
                #5.5, # log_g
                #5.0, # log_g
                5.25, # log_g
                0.65, # epsilon_limb

                #-3.3, # log_12CO
                #-3.6, # log_H2O
                #-6.2, # log_CH4
                #-6.3, # log_NH3
                #-2.0, # log_C_ratio
                -3.3, # log_12CO
                -3.6, # log_H2O
                -4.9, # log_CH4
                -6.0, # log_NH3
                -5.5, # log_13CO

                41.0, # vsini
                22.5, # rv

                #1300, # T_eff
                1400, # T_eff
        ])

        # Evaluate the model with best-fitting parameters
        for i, key_i in enumerate(self.Param.param_keys):
            # Update the Parameters instance
            self.Param.params[key_i] = synthetic_params[i]

        # Update the parameters
        self.Param.read_PT_params(cube=None)
        self.Param.read_uncertainty_params()
        self.Param.read_chemistry_params()
        self.Param.read_cloud_params()

        # Create the synthetic spectrum
        self.PMN_lnL_func(cube=None, ndim=None, nparams=None)

        # Save the PT profile
        np.savetxt(self.conf.prefix+'data/SONORA_temperature.dat', self.PT.temperature)
        np.savetxt(self.conf.prefix+'data/SONORA_RCB.dat', np.array([self.PT.RCB]))
        
        # Insert the NaNs from the observed spectrum
        self.m_spec.flux[~self.d_spec.mask_isfinite] = np.nan

        # Add noise to the synthetic spectrum
        self.m_spec.flux = np.random.normal(self.m_spec.flux, self.d_spec.err)

        # Replace the observed spectrum with the synthetic spectrum
        self.d_spec.flux = self.m_spec.flux.copy()

        # Plot the pre-processed spectrum
        figs.fig_spec_to_fit(self.d_spec, prefix=self.conf.prefix)

        # Save the synthetic spectrum
        np.save(self.conf.prefix+'data/d_spec_wave.npy', self.d_spec.wave)
        np.save(self.conf.prefix+'data/d_spec_flux.npy', self.d_spec.flux)
        np.save(self.conf.prefix+'data/d_spec_err.npy', self.d_spec.err)
        np.save(self.conf.prefix+'data/d_spec_transm.npy', self.d_spec.transm)

        # Save as pickle
        af.pickle_save(self.conf.prefix+'data/d_spec.pkl', self.d_spec)