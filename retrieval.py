# create parameters instance

# retrieval function
# ------------------
# get temperature
# save ln_L penalty (if exists)
#
# get mass_fractions
# 
# get model spectrum
# shift_broaden_rebin
# apply radius-scaling
# 
# (high_pass_filter)
# 
# get covariance
#
# marginalize over linear detector-scaling
# compute log-likelihood per order/detector
#
# callback to make figures/save some info

# multinest set-up

import os
# To not have numpy start parallelizing on its own
os.environ['OMP_NUM_THREADS'] = '1'

from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# Pause the process to not overload memory
time.sleep(1*rank)

import matplotlib.pyplot as plt
import numpy as np
import argparse

from spectrum import DataSpectrum, ModelSpectrum, Photometry
from pRT_model import pRT_model
import auxiliary_functions as af

from params_DENIS import *


def pre_processing():

    # Set up the output directories
    af.create_output_dir(prefix, file_params)

    # --- Pre-process data ----------------------------------------------

    # Get instances of the DataSpectrum class 
    # for the target and telluric standard
    d_spec = DataSpectrum(
        wave=None, 
        flux=None, 
        err=None, 
        ra=ra, 
        dec=dec, 
        mjd=mjd, 
        pwv=pwv, 
        file_target=file_target, 
        file_wave=file_wave, 
        slit=slit, 
        wave_range=wave_range, 
        )
    d_spec.clip_det_edges()
    
    d_std_spec = DataSpectrum(
        wave=None, 
        flux=None, 
        err=None, 
        ra=ra_std, 
        dec=dec_std, 
        mjd=mjd_std, 
        pwv=pwv, 
        file_target=file_std, 
        file_wave=file_wave, 
        slit=slit, 
        wave_range=wave_range, 
        )
    d_std_spec.clip_det_edges()

    # Instance of the Photometry class for the given magnitudes
    photom_2MASS = Photometry(magnitudes=magnitudes)

    # Get transmission from telluric std and add to target's class
    d_std_spec.get_transmission(T=T_std, ref_rv=0, mode='bb')
    d_spec.add_transmission(d_std_spec.transm, d_std_spec.transm_err)
    del d_std_spec

    # Apply flux calibration using the 2MASS broadband magnitude
    d_spec.flux_calib_2MASS(photom_2MASS, filter_2MASS, tell_threshold=0.2)
    del photom_2MASS

    # Apply sigma-clipping
    d_spec.sigma_clip_poly(sigma=3)

    # Crop the spectrum
    d_spec.crop_spectrum()

    # Re-shape the spectrum to a 3-dimensional array
    d_spec.reshape_orders_dets()

    # Apply barycentric correction
    d_spec.bary_corr()

    # Save as pickle
    af.pickle_save(prefix+'data/d_spec.pkl', d_spec)

    '''
    #plt.plot(d_spec.wave, d_spec.transm, c='b', lw=1)
    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            plt.plot(d_spec.wave[i,j], d_spec.flux[i,j], c='k', lw=1)
    plt.show()
    '''

    # --- Set up a pRT model --------------------------------------------

    # pRT model is somewhat wider than observed spectrum
    wave_range_micron = np.concatenate(
        (d_spec.wave.min(axis=(1,2))[None,:]-1, 
         d_spec.wave.max(axis=(1,2))[None,:]+1
         ))
    wave_range_micron *= 1e-3

    pRT_atm = pRT_model(
        line_species=line_species, 
        wave_range_micron=wave_range_micron, 
        mode='lbl', 
        lbl_opacity_sampling=lbl_opacity_sampling, 
        cloud_species=cloud_species, 
        rayleigh_species=['H2', 'He'], 
        continuum_opacities=['H2-H2', 'H2-He'], 
        log_P_range=(-6,2), 
        n_atm_layers=50
        )

    af.pickle_save(prefix+'data/pRT_atm.pkl', pRT_atm)

def retrieval():

    # Load the DataSpectrum and pRT_model classes
    d_spec  = af.pickle_load(prefix+'data/d_spec.pkl')
    pRT_atm = af.pickle_load(prefix+'data/pRT_atm.pkl')

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
        retrieval()
        pass

    if args.evaluation:
        pass