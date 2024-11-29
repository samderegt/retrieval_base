import os
os.environ['OMP_NUM_THREADS'] = '1'

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
from .chemistry import get_Chemistry_class, Chemistry
from .covariance import get_Covariance_class
from .callback import CallBack

import retrieval_base.figures as figs
import retrieval_base.auxiliary_functions as af

def pre_processing(conf, conf_data, m_set):
    """
    Pre-process the data and set up the pRT model.

    Parameters:
    conf (object): 
        Configuration object.
    conf_data (dict): 
        Configuration data dictionary.
    m_set (str): 
        Model setting identifier.
    """
    # Set up the output directories
    af.create_output_dir(conf.prefix, conf.file_params)

    # --- Pre-process data ----------------------------------------------

    #'''
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
        file_wave=conf_data['file_wave'], 
        slit=conf_data['slit'], 
        wave_range=conf_data['wave_range'], 
        w_set=conf_data['w_set'], 
        )
    d_std_spec.clip_det_edges()

    # Instance of the Photometry class for the given magnitudes
    photom_2MASS = Photometry(magnitudes=conf.magnitudes)

    has_molecfit = (conf_data.get('file_molecfit_transm') is not None)
    if not has_molecfit:
        # Get transmission from telluric std and add to target's class
        d_std_spec.get_transm(
            T=conf_data['T_std'], log_g=conf_data['log_g_std'], 
            ref_rv=conf_data['rv_std'], ref_vsini=conf_data['vsini_std'], 
            mode='PHOENIX'
            )
        
        # Interpolate onto the same wavelength grid as the target
        d_std_spec.transm = np.interp(
            d_spec.wave, d_std_spec.wave, d_std_spec.transm
            )
        d_std_spec.transm_err = np.interp(
            d_spec.wave, d_std_spec.wave, d_std_spec.transm_err
            )
        d_spec.add_transm(d_std_spec.transm, d_std_spec.transm_err)

    else:
        # Load the molecfit transmission spectrum
        d_spec.load_molecfit_transm(
            file_transm=conf_data['file_molecfit_transm'], 
            T=1500, 
            )
        d_std_spec.load_molecfit_transm(
            file_transm=conf_data['file_std_molecfit_transm'], 
            file_continuum=conf_data['file_std_molecfit_continuum'], 
            T=conf_data['T_std'], 
            )
        
        # Use the standard star throughput for the science target
        d_spec.throughput = np.copy(d_std_spec.throughput)

    del d_std_spec

    # Apply flux calibration using the 2MASS broadband magnitude
    d_spec.flux_calib_2MASS(
        photom_2MASS, 
        conf_data['filter_2MASS'], 
        tell_threshold=conf_data['tell_threshold'], 
        prefix=conf.prefix, 
        file_skycalc_transm=conf_data.get('file_skycalc_transm'), 
        molecfit=has_molecfit
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
    d_spec.mask_ghosts(wave_to_mask=conf_data.get('wave_to_mask'))

    # Re-shape the spectrum to a 3-dimensional array
    d_spec.reshape_orders_dets()

    # Apply barycentric correction
    d_spec.bary_corr()

    if conf.apply_high_pass_filter:
        # Apply high-pass filter
        d_spec.high_pass_filter(replace_flux_err=True)
        d_spec.high_pass_filtered = True

    # Prepare the wavelength separation and average squared error arrays
    d_spec.prepare_for_covariance(
        prepare_err_eff=conf.cov_kwargs.get('prepare_for_covariance', False)
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
    #'''
    '''
    w_set = conf_data['w_set']
    d_spec = af.pickle_load(conf.prefix+f'data/d_spec_{w_set}.pkl')
    '''

    # --- Set up a pRT model --------------------------------------------

    rv_range = conf.free_params.get('rv')
    if (rv_range is None) and (conf.free_params.get(m_set) is not None):
        rv_range = conf.free_params[m_set].get('rv')
    if rv_range is None:
        rv_range = [(-50,50)]
    rv_range = rv_range[0]

    vsini_range = conf.free_params.get('vsini')
    if (vsini_range is None) and (conf.free_params.get(m_set) is not None):
        vsini_range = conf.free_params[m_set].get('vsini')
    if vsini_range is None:
        vsini_range = [(0,50)]
    vsini_range = vsini_range[0]

    line_species = conf.chem_kwargs.get('line_species')
    if line_species is None:
        line_species = conf.chem_kwargs[m_set].get('line_species')

    cloud_species = conf.cloud_kwargs.get('cloud_species')
    if (cloud_species is None) and (conf.cloud_kwargs.get(m_set) is not None):
        cloud_species = conf.cloud_kwargs[m_set].get('cloud_species')

    cloud_mode = conf.cloud_kwargs.get('cloud_mode')
    if (cloud_mode is None) and (conf.cloud_kwargs.get(m_set) is not None):
        cloud_mode = conf.cloud_kwargs[m_set].get('cloud_mode')

    lbl_opacity_sampling = conf.constant_params.get('lbl_opacity_sampling')
    if lbl_opacity_sampling is None:
        lbl_opacity_sampling = conf.constant_params[m_set].get('lbl_opacity_sampling')
    if lbl_opacity_sampling is None:
        lbl_opacity_sampling = conf_data.get('lbl_opacity_sampling', 1)

    n_atm_layers = conf.constant_params.get('n_atm_layers')
    if n_atm_layers is None:
        n_atm_layers = conf.constant_params[m_set].get('n_atm_layers')
    if n_atm_layers is None:
        n_atm_layers = conf_data.get('n_atm_layers', 50)

    log_P_range = conf.constant_params.get('log_P_range')
    if log_P_range is None:
        log_P_range = conf.constant_params[m_set].get('log_P_range')
    if log_P_range is None:
        log_P_range = conf_data.get('log_P_range', (-5,3))

    # Create the Radtrans objects
    pRT_atm = pRT_model(
        line_species=line_species, 
        d_spec=d_spec, 
        mode='lbl', 
        lbl_opacity_sampling=lbl_opacity_sampling, 
        cloud_species=cloud_species, 
        cloud_mode=cloud_mode, 
        rayleigh_species=['H2', 'He'], 
        continuum_opacities=['H2-H2', 'H2-He'], 
        log_P_range=log_P_range, 
        n_atm_layers=n_atm_layers, 
        rv_range=rv_range, 
        vsini_range=vsini_range, 
        rotation_mode=conf.rotation_mode, 
        inclination=conf.constant_params.get('inclination', 0), 
        sum_m_spec=conf.sum_m_spec, 
        do_scat_emis=conf.constant_params.get('do_scat_emis', False), 
        line_opacity_kwargs=getattr(conf, 'line_opacity_kwargs', None), 
        )

    # Save as pickle
    af.pickle_save(conf.prefix+f'data/pRT_atm_{m_set}.pkl', pRT_atm)

class Retrieval:
    """
    Class for handling the retrieval process.
    """

    def __init__(self, conf, evaluation):
        """
        Initialize the Retrieval class.

        Parameters:
        conf (object): 
            Configuration object.
        evaluation (bool): 
            Flag indicating if evaluation mode is active.
        """
        self.conf = conf
        self.evaluation = evaluation

        self.sum_m_spec = self.conf.sum_m_spec

        self.d_spec  = {}
        self.pRT_atm = {}
        self.Cov     = {}
        self.LogLike = {}
        self.PT      = {}
        self.Chem    = {}
        
        self.model_settings = {}

        for m_set in conf.config_data.keys():

            # Load the DataSpectrum and pRT_model classes
            w_set = conf.config_data[m_set]['w_set']
            self.d_spec[m_set]  = af.pickle_load(self.conf.prefix+f'data/d_spec_{w_set}.pkl')
            self.pRT_atm[m_set] = af.pickle_load(self.conf.prefix+f'data/pRT_atm_{m_set}.pkl')

            self.model_settings[m_set] = [self.d_spec[m_set].n_orders, self.d_spec[m_set].n_dets]

        # Create a Parameters instance
        self.Param = Parameters(
            free_params=self.conf.free_params, 
            constant_params=self.conf.constant_params, 
            model_settings=self.model_settings, 

            PT_kwargs=self.conf.PT_kwargs, 
            chem_kwargs=self.conf.chem_kwargs, 
            cov_kwargs=self.conf.cov_kwargs, 
            cloud_kwargs=self.conf.cloud_kwargs, 
            )
        
        PT_kwargs, chem_kwargs, cov_kwargs, cloud_kwargs = \
            self.Param.get_sorted_kwargs()
        
        for m_set, d_spec_i in self.d_spec.items():

            self.Cov[m_set] = np.empty(
                (d_spec_i.n_orders, d_spec_i.n_dets), dtype=object
                )
            for i in range(d_spec_i.n_orders):
                for j in range(d_spec_i.n_dets):
                    
                    # Select only the finite pixels
                    mask_ij = d_spec_i.mask_isfinite[i,j]
                    if not mask_ij.any():
                        continue

                    self.Cov[m_set][i,j] = get_Covariance_class(
                        d_spec_i.err[i,j,mask_ij], 
                        separation=d_spec_i.separation[i,j], 
                        err_eff=d_spec_i.err_eff[i,j], 
                        flux_eff=d_spec_i.flux_eff[i,j], 
                        **cov_kwargs[m_set]
                        )

            del d_spec_i.separation
            del d_spec_i.err_eff
            del d_spec_i.flux_eff
            del d_spec_i.err

            self.LogLike[m_set] = LogLikelihood(
                d_spec_i, 
                n_params=self.Param.n_params, 
                scale_flux=self.conf.scale_flux, 
                scale_err=self.conf.scale_err, 
                scale_rel_to_ij=getattr(self.conf, 'scale_rel_to_ij', (-1,-1))
                )

            self.PT[m_set] = get_PT_profile_class(
                self.pRT_atm[m_set].pressure, 
                **PT_kwargs[m_set], 
                )
            self.Chem[m_set] = get_Chemistry_class(
                self.pRT_atm[m_set].pressure, 
                CustomOpacity=getattr(self.pRT_atm[m_set], 'LineOpacity', None), 
                **chem_kwargs[m_set], 
                )
        
        self.CB = CallBack(
            d_spec=self.d_spec, 
            evaluation=self.evaluation, 
            prefix=self.conf.prefix, 
            species_to_plot_VMR=self.conf.species_to_plot_VMR, 
            species_to_plot_CCF=self.conf.species_to_plot_CCF, 
            model_settings=self.model_settings, 
            )
        
        if self.sum_m_spec:
            # Only a single LogLike + Cov object necessary
            self.LogLike = self.LogLike[m_set]
            self.Cov     = self.Cov[m_set]

        if (rank == 0) and self.evaluation:
            self.pRT_atm_broad = {}
            for m_set in conf.config_data.keys():
                
                if os.path.exists(self.conf.prefix+f'data/pRT_atm_broad_{m_set}.pkl'):

                    # Load the pRT model
                    self.pRT_atm_broad[m_set] = af.pickle_load(
                        self.conf.prefix+f'data/pRT_atm_broad_{m_set}.pkl'
                        )
                    continue

                # Create a wider pRT model during evaluation
                self.pRT_atm_broad[m_set] = copy.deepcopy(self.pRT_atm[m_set])
                self.pRT_atm_broad[m_set].get_atmospheres(CB_active=True)

                # Save for convenience
                af.pickle_save(
                    self.conf.prefix+f'data/pRT_atm_broad_{m_set}.pkl', 
                    self.pRT_atm_broad[m_set]
                    )

        # Set to None initially, changed during evaluation
        self.m_spec_species  = None
        self.pRT_atm_species = None
        self.LogLike_species = None

    def get_PT_mf(self, m_set, Param_i):
        """
        Retrieve the PT profile and mass fractions.

        Parameters:
        m_set (str): 
            Model setting identifier.
        Param_i (object): 
            Parameter instance.

        Returns:
        tuple or float: 
            Temperature and mass fractions or -np.inf if failed.
        """
        # Retrieve the temperatures
        try:
            temperature = self.PT[m_set](Param_i.params)
        except:
            temperature = -np.inf
            return temperature

        if temperature.min() < 0:
            # Negative temperatures are rejected
            return -np.inf
        
        if (temperature.min() < 150) and (Param_i.chem_mode=='fastchem'):
            # Temperatures too low for reasonable FastChem convergence
            return -np.inf

        # Retrieve the chemical abundances
        mass_fractions = self.Chem[m_set](
            Param_i.params, temperature, param_VMRs=Param_i.VMR_species
            )

        if not isinstance(mass_fractions, dict):
            # Non-H2 abundances added up to > 1
            return -np.inf
        
        return temperature, mass_fractions
        
    def PMN_lnL_func(self, cube=None, ndim=None, nparams=None):
        """
        Compute the log-likelihood function for MultiNest.

        Parameters:
        cube (array, optional): 
            Parameter cube.
        ndim (int, optional): 
            Number of dimensions.
        nparams (int, optional): 
            Number of parameters.

        Returns:
        float: 
            Log-likelihood value.
        """
        time_A = time.time()

        # Param.params dictionary is already updated
        
        ln_L = 0
        self.m_spec = {}

        pRT_atm_to_use = self.pRT_atm
        if self.evaluation:
            # Retrieve the model spectrum, with the wider pRT model
            pRT_atm_to_use = self.pRT_atm_broad
            
        for i, (m_set, Param_i) in enumerate(self.Param.Params_m_set.items()):
            
            # Retrieve the PT profile and abundances
            returned = self.get_PT_mf(m_set, Param_i)
            if isinstance(returned, float):
                # PT profile or mass fractions failed
                return -np.inf
    
            if i == 0:
                # Retrieve the ln L penalty (=0 by default)
                ln_L_penalty = getattr(self.PT[m_set], 'ln_L_penalty', 0)
                ln_L += ln_L_penalty

            temperature, mass_fractions = returned

            # Retrieve the model spectrum
            self.m_spec[m_set] = pRT_atm_to_use[m_set](
                mass_fractions, 
                temperature, 
                Param_i.params, 
                get_contr=self.CB.active, 
                get_full_spectrum=self.evaluation, 
                CO=self.Chem[m_set].CO, 
                FeH=self.Chem[m_set].FeH, 
                )

            # Update the covariance matrix
            for j in range(self.d_spec[m_set].n_orders):
                for k in range(self.d_spec[m_set].n_dets):

                    if not self.d_spec[m_set].mask_isfinite[j,k].any():
                        continue

                    if self.sum_m_spec:
                        self.Cov[j,k](
                            Param_i.params, m_set, order=j, det=k, 
                            **self.conf.cov_kwargs, 
                            )
                    else:
                        self.Cov[m_set][j,k](
                            Param_i.params, m_set, order=j, det=k, 
                            **self.conf.cov_kwargs, 
                            )
            
            if self.sum_m_spec:
                # Spectra need to be combined at this point
                continue
            
            # Compute log-likelihood separate for each model-setting
            M = self.m_spec[m_set].flux[:,:,None,:]

            # Retrieve the log-likelihood
            ln_L += self.LogLike[m_set](M, self.Cov[m_set])
        
        if self.sum_m_spec:
            
            # Separate the model settings
            m_set_1, *other_m_set = pRT_atm_to_use.keys()

            # Collect the pRT model wavelengths and fluxes
            other_pRT_wave, other_pRT_flux = [], []
            for m_set in other_m_set:
                other_pRT_wave.append(
                    pRT_atm_to_use[m_set].pRT_wave
                )
                other_pRT_flux.append(
                    pRT_atm_to_use[m_set].pRT_flux
                )
                # Clear up some memory
                del pRT_atm_to_use[m_set].pRT_wave, pRT_atm_to_use[m_set].pRT_flux

            # Sum the different model settings
            self.m_spec[m_set_1] = pRT_atm_to_use[m_set_1].combine_models(
                other_pRT_wave=other_pRT_wave, 
                other_pRT_flux=other_pRT_flux, 
                get_contr=self.CB.active, 
                get_full_spectrum=self.evaluation, 
                )

            # Reshape the model
            M = self.m_spec[m_set_1].flux[:,:,None,:]

            # Retrieve the log-likelihood
            ln_L += self.LogLike(M, self.Cov)

        time_B = time.time()
        self.CB.elapsed_times.append(time_B-time_A)

        return ln_L
    
    def parallel_for_loop(self, func, iterable, **kwargs):
        """
        Parallel for loop using MPI.

        Parameters:
        func (function): 
            Function to apply.
        iterable (iterable): 
            Iterable to loop over.
        **kwargs: 
            Additional keyword arguments.

        Returns:
        list: 
            Combined results from all processes.
        """
        n_iter = len(iterable)
        n_procs = comm.Get_size()
        
        # Number of iterables to compute per process
        perrank = int(n_iter / n_procs) + 1

        # Lower, upper indices to compute for this rank
        low, high = rank*perrank, (rank+1)*perrank
        if rank == comm.Get_size()-1:
            # Final rank has fewer iterations
            high = n_iter

        # Run the function
        returned = []
        for i in range(low, high):
            if i >= len(iterable):
                break
            returned_i = func(iterable[i], **kwargs)
            returned.append(returned_i)

        # Pause until all processes finished
        comm.Barrier()

        # Combine the outputs
        all_returned = comm.gather(returned, root=0)
        if rank != 0:
            return
        
        if not hasattr(returned_i, '__len__'):
            # Only 1 value returned per process
            
            # Concatenate the lists
            flat_all_returned = []
            for sublist in all_returned:
                for item in sublist:
                    flat_all_returned.append(item)

            return flat_all_returned
        
        # Multiple values returned per process
        flat_all_returned = [[] for _ in range(len(returned_i))]
        for sublist_1 in all_returned:
            for sublist_2 in sublist_1:
                for i, item in enumerate(sublist_2):
                    
                    flat_all_returned[i].append(item)

        return flat_all_returned

    def get_PT_mf_envelopes(self, posterior, m_set):
        """
        Get the PT profile and mass fractions envelopes.

        Parameters:
        posterior (array): 
            Posterior distribution.
        m_set (str): 
            Model setting identifier.
        """
        # Return the PT profile and mass fractions
        #self.CB.return_PT_mf = {m_set: True for m_set in self.model_settings.keys()}

        # Objects to store the envelopes in
        self.Chem[m_set].mass_fractions_posterior = {}
        self.Chem[m_set].unquenched_mass_fractions_posterior = {}

        self.Chem[m_set].CO_posterior  = []
        self.Chem[m_set].FeH_posterior = []

        #self.PT.temperature_envelopes = []

        def func(params_i):

            self.Param.apply_prior = False
            self.Param(params_i)
            self.Param.apply_prior = True

            # Class instances with best-fitting parameters
            returned = self.get_PT_mf(m_set, self.Param.Params_m_set[m_set])
            
            if isinstance(returned, float):
                # PT profile or mass fractions failed
                return None, None, None, None, None

            # Store the temperatures and mass fractions
            temperature_i, mass_fractions_i = returned
            unquenched_mass_fractions_i = None
            if hasattr(self.Chem[m_set], 'unquenched_mass_fractions'):
                unquenched_mass_fractions_i = self.Chem[m_set].unquenched_mass_fractions

            # Return the temperature, mass fractions, unquenched, C/O ratio and Fe/H
            return (
                temperature_i, 
                mass_fractions_i, 
                unquenched_mass_fractions_i, 
                self.Chem[m_set].CO, 
                self.Chem[m_set].FeH
                )
        
        # Compute the mass fractions posterior in parallel
        returned = self.parallel_for_loop(func, posterior)

        if returned is None:
            return
        
        self.PT[m_set].temperature_posterior, \
        mass_fractions_posterior, \
        unquenched_mass_fractions_posterior, \
        self.Chem[m_set].CO_posterior, \
        self.Chem[m_set].FeH_posterior \
            = returned
        
        self.PT[m_set].temperature_posterior = np.array(self.PT[m_set].temperature_posterior)
        self.Chem[m_set].CO_posterior  = np.array(self.Chem[m_set].CO_posterior)
        self.Chem[m_set].FeH_posterior = np.array(self.Chem[m_set].FeH_posterior)

        # Create the lists to store mass fractions per line species
        for line_species_i in mass_fractions_posterior[0].keys():

            self.Chem[m_set].mass_fractions_posterior[line_species_i] = []

            if unquenched_mass_fractions_posterior[0] is None:
                continue
            self.Chem[m_set].unquenched_mass_fractions_posterior[line_species_i] = []

        # Store the mass fractions posterior in the correct order
        for mf_i, unquenched_mf_i in zip(
            mass_fractions_posterior, unquenched_mass_fractions_posterior
            ):
            
            # Loop over the line species
            for line_species_i in mf_i.keys():

                self.Chem[m_set].mass_fractions_posterior[line_species_i].append(
                    mf_i[line_species_i]
                    )

                if unquenched_mf_i is None:
                    continue
                # Store the unquenched mass fractions
                self.Chem[m_set].unquenched_mass_fractions_posterior[line_species_i].append(
                    unquenched_mf_i[line_species_i]
                    )

        # Convert profiles to 1, 2, 3-sigma equivalent and median
        q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
             0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
             ]            

        # Retain the pressure-axis
        self.PT[m_set].temperature_envelopes = af.quantiles(
            self.PT[m_set].temperature_posterior, q=q, axis=0
            )

        self.Chem[m_set].mass_fractions_envelopes = {}
        self.Chem[m_set].unquenched_mass_fractions_envelopes = {}

        for line_species_i in self.Chem[m_set].mass_fractions.keys():

            self.Chem[m_set].mass_fractions_posterior[line_species_i] = \
                np.array(self.Chem[m_set].mass_fractions_posterior[line_species_i])

            self.Chem[m_set].mass_fractions_envelopes[line_species_i] = af.quantiles(
                self.Chem[m_set].mass_fractions_posterior[line_species_i], q=q, axis=0
                )
            
            if unquenched_mass_fractions_posterior[0] is None:
                continue

        # Store the unquenched mass fractions
        for line_species_i in self.Chem[m_set].unquenched_mass_fractions_posterior.keys():
            self.Chem[m_set].unquenched_mass_fractions_posterior[line_species_i] = \
                np.array(self.Chem[m_set].unquenched_mass_fractions_posterior[line_species_i])

            self.Chem[m_set].unquenched_mass_fractions_envelopes[line_species_i] = af.quantiles(
                self.Chem[m_set].unquenched_mass_fractions_posterior[line_species_i], q=q, axis=0
                )

        #self.CB.return_PT_mf = False

    def get_species_contribution(self):
        """
        Assess the species' contribution to the spectrum.
        """
        self.m_spec_species  = dict.fromkeys(self.d_spec.keys(), {})
        self.pRT_atm_species = dict.fromkeys(self.d_spec.keys(), {})

        # Assess the species' contribution
        for species_i in Chemistry.species_info.index:

            line_species_i = Chemistry.read_species_info(species_i, 'pRT_name')

            for m_set in self.Chem.keys():
                if (line_species_i not in self.Chem[m_set].line_species) and \
                    (line_species_i not in getattr(self.Chem[m_set], 'custom_pRT_names', [])):
                    continue

                # Ignore one species at a time
                self.Chem[m_set].neglect_species = \
                    dict.fromkeys(self.Chem[m_set].neglect_species, False)
                self.Chem[m_set].neglect_species[species_i] = True

            # Create the spectrum and evaluate lnL
            self.PMN_lnL_func()

            for m_set in self.Chem.keys():
                if self.m_spec[m_set] is None:
                    continue
                self.m_spec_species[m_set][species_i]  = copy.deepcopy(self.m_spec[m_set])
                self.pRT_atm_species[m_set][species_i] = copy.deepcopy(self.pRT_atm_broad[m_set])

            # Turn all but species_i off
            for m_set in self.Chem.keys():
                if (line_species_i not in self.Chem[m_set].line_species) and \
                    (line_species_i not in getattr(self.Chem[m_set], 'custom_pRT_names', [])):
                    continue

                self.Chem[m_set].neglect_species = \
                    dict.fromkeys(self.Chem[m_set].neglect_species, True)
                self.Chem[m_set].neglect_species[species_i] = False

            # Create the spectrum and evaluate lnL
            self.PMN_lnL_func()

            for m_set in self.Chem.keys():
                if self.m_spec_species[m_set][species_i] is None:
                    continue
                if self.m_spec[m_set] is None:
                    continue
                self.m_spec_species[m_set][species_i].flux_only = \
                    self.m_spec[m_set].flux.copy()
                self.pRT_atm_species[m_set][species_i].flux_pRT_grid_only = \
                    self.pRT_atm_broad[m_set].flux_pRT_grid.copy()
                
        for m_set in self.Chem.keys():
            # Include all species again
            self.Chem[m_set].neglect_species = \
                dict.fromkeys(self.Chem[m_set].neglect_species, False)

    def get_all_spectra(self, posterior, save_spectra=False):
        """
        Get all spectra from the posterior distribution.

        Parameters:
        posterior (array): 
            Posterior distribution.
        save_spectra (bool, optional): 
            Flag to save spectra.

        Returns:
        array: 
            Flux envelope.
        """
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
        """
        Callback function for MultiNest.

        Parameters:
        n_samples (int): 
            Number of samples.
        n_live (int): 
            Number of live points.
        n_params (int): 
            Number of parameters.
        live_points (array): 
            Live points.
        posterior (array): 
            Posterior distribution.
        stats (dict): 
            Statistics.
        max_ln_L (float): 
            Maximum log-likelihood.
        ln_Z (float): 
            Log-evidence.
        ln_Z_err (float): 
            Log-evidence error.
        nullcontext (context): 
            Null context.
        """
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
            for m_set in self.model_settings.keys():
                self.get_PT_mf_envelopes(posterior, m_set=m_set)
            
            # Get the model flux envelope
            #flux_envelope = self.get_all_spectra(posterior, save_spectra=False)

        else:
            
            # Read the parameters of the best-fitting model
            bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2]

            # Remove the last 2 columns
            posterior = posterior[:,:-2]

        if rank != 0:
            return
        
        # Evaluate the model with best-fitting parameters
        self.Param.apply_prior = False
        if (bestfit_params < 1).all() and (bestfit_params > 0).all():
            self.Param.apply_prior = True
        self.Param(bestfit_params)
        self.Param.apply_prior = True

        if self.evaluation:
            # Get each species' contribution to the spectrum
            self.get_species_contribution()

        # Update class instances with best-fitting parameters
        self.PMN_lnL_func()
        self.CB.active = False

        for m_set in self.conf.config_data.keys():
            if self.m_spec[m_set] is None:
                continue
            self.m_spec[m_set].flux_envelope = None

        pRT_atm_to_use = self.pRT_atm
        if self.evaluation:
            # Retrieve the model spectrum, with the wider pRT model
            pRT_atm_to_use = self.pRT_atm_broad

            # Save the updated pRT emission contribution
            for m_set in self.pRT_atm_broad.keys():
                for i, atm_i in enumerate(self.pRT_atm_broad[m_set].atm):
                    np.save(self.conf.prefix+f'data/contr_em_order{i}_{m_set}', atm_i.contr_em)
        
        # Call the CallBack class and make summarizing figures
        self.CB(
            self.Param, 
            self.LogLike, 
            self.Cov, 
            self.PT, 
            self.Chem, 
            self.m_spec, 
            pRT_atm_to_use, 
            posterior, 
            m_spec_species=self.m_spec_species, 
            pRT_atm_species=self.pRT_atm_species, 
            )

    def PMN_run(self):
        """
        Run the MultiNest retrieval.
        """
        # Pause the process to not overload memory on start-up
        time.sleep(0.3*rank*len(self.d_spec))

        # Run the MultiNest retrieval
        #pymultinest.solve(
        pymultinest.run(
            LogLikelihood=self.PMN_lnL_func, 
            Prior=self.Param, 
            n_dims=self.Param.n_params, 
            outputfiles_basename=self.conf.prefix, 
            resume=True, 
            #resume=False, 
            verbose=True, 
            const_efficiency_mode=self.conf.const_efficiency_mode, 
            sampling_efficiency=self.conf.sampling_efficiency, 
            n_live_points=self.conf.n_live_points, 
            evidence_tolerance=self.conf.evidence_tolerance, 
            dump_callback=self.PMN_callback_func, 
            n_iter_before_update=self.conf.n_iter_before_update, 
            )