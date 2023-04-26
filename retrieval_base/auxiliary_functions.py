import pickle
import os
import shutil

import numpy as np
from scipy.interpolate import interp1d
import petitRADTRANS.nat_cst as nc

def pickle_save(file, object_to_pickle):

    with open(file, 'wb') as f:
        pickle.dump(object_to_pickle, f)
        
def pickle_load(file):
    
    with open(file, 'rb') as f:
        pickled_object = pickle.load(f)

    return pickled_object

def create_output_dir(prefix, file_params):
    
    # Make the output directory
    if not os.path.exists('/'.join(prefix.split('/')[:-1])):
        os.makedirs('/'.join(prefix.split('/')[:-1]))

    # Make the plots directory
    if not os.path.exists(prefix+'plots'):
        os.makedirs(prefix+'plots')

    # Make the data directory
    if not os.path.exists(prefix+'data'):
        os.makedirs(prefix+'data')

    # Make a copy of the parameters file
    shutil.copy(file_params, prefix+'data/'+file_params)

def quantiles(x, q, weights=None, axis=-1):
    '''
    Compute (weighted) quantiles from an input set of samples.
    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.
    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.
    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.
    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.
    '''

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError('Quantiles must be between 0. and 1.')

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q), axis=axis)
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError('Dimension mismatch: len(weights) != len(x).')
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles

def CCF_to_SNR(rv, CCF, rv_to_exclude=(-100,100)):

    # Select samples from outside the expected peak
    rv_mask = (rv < rv_to_exclude[0]) | (rv > rv_to_exclude[1])

    # Correct the offset
    CCF -= np.nanmean(CCF[rv_mask])

    # Standard-deviation of the cross-correlation function
    std_CCF = np.nanstd(CCF[rv_mask])

    # Convert to signal-to-noise
    return CCF/std_CCF
    
def CCF(d_spec, 
        m_spec, 
        m_wave_pRT_grid, 
        m_flux_pRT_grid, 
        m_spec_wo_species=None, 
        m_flux_wo_species_pRT_grid=None, 
        LogLike=None, 
        rv=np.arange(-500,500+1e-6,1), 
        ):

    CCF = np.zeros((d_spec.n_orders, d_spec.n_dets, len(rv)))
    d_ACF = np.zeros_like(CCF)
    m_ACF = np.zeros_like(CCF)

    # Loop over all orders and detectors
    for i in range(d_spec.n_orders):

        m_wave_i = m_wave_pRT_grid[i]
        m_flux_i = m_flux_pRT_grid[i]
        
        if m_flux_wo_species_pRT_grid is not None:
            # Perform the cross-correlation on the residuals
            m_flux_i -= m_flux_wo_species_pRT_grid[i]

        # Function to interpolate the model spectrum
        m_interp_func = interp1d(
            m_wave_i, m_flux_i, bounds_error=False, fill_value=np.nan
            )
        
        for j in range(d_spec.n_dets):

            # Select only the pixels within this order
            mask_ij = d_spec.mask_isfinite[i,j,:]

            d_wave_ij = d_spec.wave[i,j,mask_ij]
            d_flux_ij = d_spec.flux[i,j,mask_ij]

            if LogLike is not None:
                # Use the covariance matrix to weigh 
                # the cross-correlation coefficients
                cov_ij = LogLike.cov[i,j]

                # Scale the data instead of the models
                d_flux_ij /= LogLike.f[i,j]

            if m_spec_wo_species is not None:
                # Perform the cross-correlation on the residuals
                d_flux_ij -= m_spec_wo_species.flux[i,j,mask_ij]

            # Function to interpolate the observed spectrum
            d_interp_func = interp1d(
                d_wave_ij, d_flux_ij, bounds_error=False, fill_value=np.nan
                )
            
            # Create a static model template
            m_flux_ij_static = m_interp_func(d_wave_ij)
            
            for k, rv_k in enumerate(rv):

                # Apply Doppler shift
                d_wave_ij_shifted = d_wave_ij * (1 + rv_k/(nc.c*1e-5))

                # Interpolate the spectra onto the new wavelength grid
                d_flux_ij_shifted = d_interp_func(d_wave_ij_shifted)
                m_flux_ij_shifted = m_interp_func(d_wave_ij_shifted)

                # Compute the cross-correlation coefficient, weighted 
                # by the covariance matrix
                #CCF_k = np.dot(m_flux_ij_shifted, d_flux_ij/d_spec.err[i,j,mask_ij]**2)
                CCF[i,j,k] = np.dot(m_flux_ij_shifted, cov_ij.solve(d_flux_ij))
                
                # Compute the auto-correlation coefficients, weighted 
                # by the covariance matrix
                #m_ACF_k = np.dot(m_flux_ij_shifted, m_flux_ij_static/d_spec.err[i,j,mask_ij]**2)
                m_ACF[i,j,k] = np.dot(m_flux_ij_shifted, cov_ij.solve(m_flux_ij_static))

                #d_ACF_k = np.dot(d_flux_ij_shifted, d_flux_ij/d_spec.err[i,j,mask_ij]**2)
                d_ACF[i,j,k] = np.dot(d_flux_ij_shifted, cov_ij.solve(d_flux_ij))

                # Scale the correlation coefficients
                if LogLike is not None:
                    CCF[i,j,k]   *= LogLike.f[i,j]/LogLike.beta[i,j]**2
                    m_ACF[i,j,k] *= LogLike.f[i,j]/LogLike.beta[i,j]**2
                    d_ACF[i,j,k] *= LogLike.f[i,j]/LogLike.beta[i,j]**2

    return rv, CCF, d_ACF, m_ACF