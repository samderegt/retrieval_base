import os
# To not have numpy start parallelizing on its own
os.environ['OMP_NUM_THREADS'] = '1'

from mpi4py import MPI
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# Pause the process to not overload memory
time.sleep(1.5*rank)

import pymultinest
import matplotlib.pyplot as plt
import numpy as np
import argparse

from retrieval_base.spectrum import DataSpectrum, ModelSpectrum, Photometry
from retrieval_base.parameters import Parameters
from retrieval_base.pRT_model import pRT_model
from retrieval_base.log_likelihood import LogLikelihood
from retrieval_base.PT_profile import PT_profile_free, PT_profile_Molliere
from retrieval_base.chemistry import FreeChemistry, EqChemistry
from retrieval_base.callback import CallBack

import retrieval_base.figures as figs
import retrieval_base.auxiliary_functions as af

import config_DENIS as conf

def pre_processing():

    # Set up the output directories
    af.create_output_dir(conf.prefix, conf.file_params)

    # --- Pre-process data ----------------------------------------------

    # Get instances of the DataSpectrum class 
    # for the target and telluric standard
    d_spec = DataSpectrum(
        wave=None, 
        flux=None, 
        err=None, 
        ra=conf.ra, 
        dec=conf.dec, 
        mjd=conf.mjd, 
        pwv=conf.pwv, 
        file_target=conf.file_target, 
        file_wave=conf.file_wave, 
        slit=conf.slit, 
        wave_range=conf.wave_range, 
        )
    d_spec.clip_det_edges()
    
    d_std_spec = DataSpectrum(
        wave=None, 
        flux=None, 
        err=None, 
        ra=conf.ra_std, 
        dec=conf.dec_std, 
        mjd=conf.mjd_std, 
        pwv=conf.pwv, 
        file_target=conf.file_std, 
        file_wave=conf.file_wave, 
        slit=conf.slit, 
        wave_range=conf.wave_range, 
        )
    d_std_spec.clip_det_edges()

    # Instance of the Photometry class for the given magnitudes
    photom_2MASS = Photometry(magnitudes=conf.magnitudes)

    # Get transmission from telluric std and add to target's class
    d_std_spec.get_transmission(T=conf.T_std, ref_rv=0, mode='bb')
    d_spec.add_transmission(d_std_spec.transm, d_std_spec.transm_err)
    del d_std_spec

    # Apply flux calibration using the 2MASS broadband magnitude
    d_spec.flux_calib_2MASS(
        photom_2MASS, 
        conf.filter_2MASS, 
        tell_threshold=0.3, 
        prefix=conf.prefix, 
        file_skycalc_transm=conf.file_skycalc_transm, 
        )
    del photom_2MASS

    # Apply sigma-clipping
    d_spec.sigma_clip_poly(sigma=3, prefix=conf.prefix)

    # Crop the spectrum
    d_spec.crop_spectrum()

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

    # Plot the pre-processed spectrum
    figs.fig_spec_to_fit(d_spec, prefix=conf.prefix)

    # Save as pickle
    af.pickle_save(conf.prefix+'data/d_spec.pkl', d_spec)

    # --- Set up a pRT model --------------------------------------------

    # Create the Radtrans objects
    pRT_atm = pRT_model(
        line_species=conf.line_species, 
        d_spec=d_spec, 
        mode='lbl', 
        lbl_opacity_sampling=conf.lbl_opacity_sampling, 
        cloud_species=conf.cloud_species, 
        rayleigh_species=['H2', 'He'], 
        continuum_opacities=['H2-H2', 'H2-He'], 
        log_P_range=(-6,2), 
        n_atm_layers=50, 
        )

    # Save as pickle
    af.pickle_save(conf.prefix+'data/pRT_atm.pkl', pRT_atm)

class PMN_Retrieval:

    def __init__(self):
        
        # Load the DataSpectrum and pRT_model classes
        self.d_spec  = af.pickle_load(conf.prefix+'data/d_spec.pkl')
        self.pRT_atm = af.pickle_load(conf.prefix+'data/pRT_atm.pkl')

        # Create a Parameters instance
        self.Param = Parameters(
            free_params=conf.free_params, 
            constant_params=conf.constant_params, 
            n_orders=self.d_spec.n_orders, 
            n_dets=self.d_spec.n_dets
            )    

        if 'beta_tell' not in self.Param.param_keys:
            # Transmissivity will not be requested, save on memory
            self.d_spec.transm     = None
            self.d_spec.transm_err = None

        # Update the cloud/chemistry-mode
        self.pRT_atm.cloud_mode = self.Param.cloud_mode
        self.pRT_atm.chem_mode  = self.Param.chem_mode

        self.LogLike = LogLikelihood(
            self.d_spec, 
            n_params=self.Param.n_params, 
            scale_flux=conf.scale_flux, 
            scale_err=conf.scale_err, 
            scale_GP_amp=conf.scale_GP_amp, 
            )

        if self.Param.PT_mode == 'Molliere':
            self.PT = PT_profile_Molliere(
                self.pRT_atm.pressure, 
                conv_adiabat=True
                )
        elif self.Param.PT_mode == 'free':
            self.PT = PT_profile_free(
                self.pRT_atm.pressure, 
                ln_L_penalty_order=conf.ln_L_penalty_order
                )

        if self.Param.chem_mode == 'free':
            self.Chem = FreeChemistry(
                self.pRT_atm.line_species, self.pRT_atm.pressure
                )
        elif self.Param.chem_mode == 'eqchem':
            self.Chem = EqChemistry(
                self.pRT_atm.line_species, self.pRT_atm.pressure
                )

        self.CB = CallBack(
            d_spec=self.d_spec, 
            evaluation=args.evaluation, 
            n_samples_to_use=2000, 
            prefix=conf.prefix, 
            posterior_color='C0', 
            bestfit_color='C3', 
            )
        
        self.return_PT_mf = False

    def PMN_lnL_func(self, cube, ndim, nparams):

        time_A = time.time()

        # Param.params is already updated

        # Retrieve the temperatures
        try:
            temperature = self.PT(self.Param.params)
        except:
            # Something went wrong with interpolating
            return -np.inf

        if temperature.min() < 0:
            # Negative temperatures are rejected
            return -np.inf

        # Retrieve the ln L penalty (=0 by default)
        ln_L_penalty = self.PT.ln_L_penalty

        # Retrieve the chemical abundances
        if self.Param.chem_mode == 'free':
            mass_fractions = self.Chem(self.Param.VMR_species)
        elif self.Param.chem_mode == 'eqchem':
            mass_fractions = self.Chem(self.Param.params)

        if not isinstance(mass_fractions, dict):
            # Non-H2 abundances added up to > 1
            return -np.inf

        if self.return_PT_mf:
            # Return temperatures and mass fractions during evaluation
            return (temperature, mass_fractions)

        # Retrieve the model spectrum
        m_spec = self.pRT_atm(
            mass_fractions, 
            temperature, 
            self.Param.params, 
            get_contr=False, 
            )

        # Retrieve the log-likelihood
        ln_L = self.LogLike(
            m_spec, 
            self.Param.params, 
            ln_L_penalty=ln_L_penalty
            )

        time_B = time.time()
        self.CB.elapsed_times.append(time_B-time_A)

        if self.CB.active:

            # Return the class instances during callback
            return (self.LogLike, self.PT, self.Chem, m_spec, self.pRT_atm)

        return ln_L

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

        if args.evaluation:
            
            # Return the PT profile and mass fractions
            self.return_PT_mf = True

            # Set-up analyzer object
            analyzer = pymultinest.Analyzer(
                n_params=n_params, 
                outputfiles_basename=conf.prefix
                )
            stats = analyzer.get_stats()

            # Load the equally-weighted posterior distribution
            posterior = analyzer.get_equal_weighted_posterior()
            posterior = posterior[:,:-1]

            # Read the parameters of the best-fitting model
            bestfit_params = np.array(stats['modes'][0]['maximum a posterior'])
           
            # Objects to store the envelopes in 
            mass_fractions_envelopes = {}
            for line_species_i in self.pRT_atm.line_species:
                mass_fractions_envelopes[line_species_i] = []
            temperature_envelopes = []

            # Sample envelopes from the posterior 
            for params_i in posterior:

                for j, key_j in enumerate(self.Param.param_keys):
                    # Update the Parameters instance
                    self.Param.params[key_j] = params_i[j]

                # Update the parameters
                self.Param.read_PT_params(cube=None)
                self.Param.read_uncertainty_params()
                self.Param.read_chemistry_params()
                self.Param.read_cloud_params()

                # Class instances with best-fitting parameters
                returned = self.PMN_lnL_func(
                    cube=None, ndim=None, nparams=None
                    )
                if isinstance(returned, float):
                    # PT profile or mass fractions failed
                    continue
                
                # Store the temperatures and mass fractions
                temperature_i, mass_fractions_i = returned
                temperature_envelopes.append(temperature_i)
                # Loop over the line species
                for line_species_i in self.pRT_atm.line_species:
                    mass_fractions_envelopes[line_species_i].append(
                        mass_fractions_i[line_species_i]
                        )

            # Convert profiles to 1, 2, 3-sigma equivalent and median
            q = [0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
                 0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
                 ]            

            # Retain the pressure-axis
            temperature_envelopes = af.quantiles(
                np.array(temperature_envelopes), q=q, axis=0
                )
            for line_species_i in self.pRT_atm.line_species:
                mass_fractions_envelopes[line_species_i] = af.quantiles(
                    np.array(mass_fractions_envelopes[line_species_i]), q=q, axis=0
                    )

            self.return_PT_mf = False

        else:

            # Read the parameters of the best-fitting model
            bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2]

            # Remove the last 2 columns
            posterior = posterior[:,:-2]

            mass_fractions_envelopes, temperature_envelopes = None, None

        # Evaluate the model with best-fitting parameters
        for i, key_i in enumerate(self.Param.param_keys):
            # Update the Parameters instance
            self.Param.params[key_i] = bestfit_params[i]

        # Update the parameters
        self.Param.read_PT_params(cube=None)
        self.Param.read_uncertainty_params()
        self.Param.read_chemistry_params()
        self.Param.read_cloud_params()

        # Class instances with best-fitting parameters
        LogLike, PT, Chem, m_spec, pRT_atm = self.PMN_lnL_func(
            cube=None, ndim=None, nparams=None
            )
        self.CB.active = False

        # Call the CallBack class and make summarizing figures
        self.CB(
            self.Param, LogLike, PT, Chem, m_spec, pRT_atm, posterior, 
            temperature_envelopes, mass_fractions_envelopes
            )

    def run(self):

        # Set-up the MultiNest retrieval
        pymultinest.run(
            LogLikelihood=self.PMN_lnL_func, 
            Prior=self.Param, 
            n_dims=self.Param.n_params, 
            outputfiles_basename=conf.prefix, 
            resume=True, 
            verbose=True, 
            const_efficiency_mode=conf.const_efficiency_mode, 
            sampling_efficiency=conf.sampling_efficiency, 
            n_live_points=conf.n_live_points, 
            evidence_tolerance=conf.evidence_tolerance, 
            dump_callback=self.PMN_callback_func, 
            n_iter_before_update=conf.n_iter_before_update, 
            )
        
if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_processing', action='store_true')
    parser.add_argument('--retrieval', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    #parser.add_argument('--spec_posterior', action='store_true')
    args = parser.parse_args()

    if args.pre_processing:
        pre_processing()

    if args.retrieval:
        #retrieval()
        ret = PMN_Retrieval()
        ret.run()

    if args.evaluation:
        #retrieval()
        ret = PMN_Retrieval()
        ret.run()