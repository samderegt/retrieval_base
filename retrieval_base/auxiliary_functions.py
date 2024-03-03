import pickle
import os
import shutil
import wget

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, generic_filter

from astropy.io import fits

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

def CCF_to_SNR(rv, CCF, ACF=None, rv_to_exclude=(-100,100)):

    # Select samples from outside the expected peak
    rv_mask = (rv < rv_to_exclude[0]) | (rv > rv_to_exclude[1])

    # Correct the offset
    CCF -= np.nanmean(CCF[rv_mask])

    if ACF is not None:
        # Correct the offset
        ACF -= np.nanmean(ACF[rv_mask])

        # Subtract the (scaled) ACF before computing the standard-deviation
        excluded_CCF = (CCF - ACF*(CCF/ACF)[rv==0])[rv_mask]
    else:
        excluded_CCF = CCF[rv_mask]

    # Standard-deviation of the cross-correlation function
    std_CCF = np.nanstd(excluded_CCF)
    
    # Convert to signal-to-noise
    CCF_SNR = CCF / std_CCF
    if ACF is not None:
        ACF_SNR = ACF*(CCF/ACF)[rv==0]/std_CCF
    else:
        ACF_SNR = None
    
    return CCF_SNR, ACF_SNR, (CCF - ACF*(CCF/ACF)[rv==0])/std_CCF
    
def CCF(d_spec, 
        m_spec, 
        m_wave_pRT_grid, 
        m_flux_pRT_grid, 
        m_spec_wo_species=None, 
        m_flux_wo_species_pRT_grid=None, 
        LogLike=None, 
        Cov=None, 
        rv=np.arange(-500,500+1e-6,1), 
        apply_high_pass_filter=True, 
        ):

    CCF = np.zeros((d_spec.n_orders, d_spec.n_dets, len(rv)))
    d_ACF = np.zeros_like(CCF)
    m_ACF = np.zeros_like(CCF)

    # Loop over all orders and detectors
    for i in range(d_spec.n_orders):

        if m_wave_pRT_grid is not None:
            m_wave_i = m_wave_pRT_grid[i]
            m_flux_i = m_flux_pRT_grid[i].copy()

            if m_flux_wo_species_pRT_grid is not None:
                # Perform the cross-correlation on the residuals
                m_flux_i -= m_flux_wo_species_pRT_grid[i].copy()
                #m_flux_i = m_flux_wo_species_pRT_grid[i].copy()

            # Function to interpolate the model spectrum
            m_interp_func = interp1d(
                m_wave_i, m_flux_i, bounds_error=False, fill_value=np.nan
                )
        
        for j in range(d_spec.n_dets):

            # Select only the pixels within this order
            mask_ij = d_spec.mask_isfinite[i,j,:]

            if not mask_ij.any():
                continue

            if m_wave_pRT_grid is None:
                m_wave_i = m_spec.wave[i,j]
                m_flux_i = m_spec.flux[i,j]

                # Function to interpolate the model spectrum
                m_interp_func = interp1d(
                    m_wave_i[np.isfinite(m_flux_i)], 
                    m_flux_i[np.isfinite(m_flux_i)], 
                    bounds_error=False, fill_value=np.nan
                    )

            d_wave_ij = d_spec.wave[i,j,mask_ij]
            d_flux_ij = d_spec.flux[i,j,mask_ij]

            if LogLike is not None:
                # Scale the data instead of the models
                d_flux_ij /= LogLike.f[i,j]
            
            if Cov is not None:
                # Use the covariance matrix to weigh 
                # the cross-correlation coefficients
                cov_ij = Cov[i,j]

            if m_spec_wo_species is not None:
                # Perform the cross-correlation on the residuals
                d_flux_ij -= m_spec_wo_species.flux[i,j,mask_ij]

            # Function to interpolate the observed spectrum
            d_interp_func = interp1d(
                d_wave_ij, d_flux_ij, bounds_error=False, 
                fill_value=np.nan
                )
            
            # Create a static model template
            m_flux_ij_static = m_interp_func(d_wave_ij)

            if apply_high_pass_filter:
                # Apply high-pass filter
                d_flux_ij -= gaussian_filter1d(d_flux_ij, sigma=300, mode='reflect')
                m_flux_ij_static -= gaussian_filter1d(
                    m_flux_ij_static, sigma=300, mode='reflect'
                    )
            
            
            for k, rv_k in enumerate(rv):

                # Apply Doppler shift
                d_wave_ij_shifted = d_wave_ij * (1 + rv_k/(nc.c*1e-5))

                # Interpolate the spectra onto the new wavelength grid
                d_flux_ij_shifted = d_interp_func(d_wave_ij_shifted)
                m_flux_ij_shifted = m_interp_func(d_wave_ij_shifted)

                if apply_high_pass_filter:
                    # Apply high-pass filter
                    d_flux_ij_shifted -= gaussian_filter1d(
                        d_flux_ij_shifted, sigma=300, mode='reflect'
                        )
                    m_flux_ij_shifted -= gaussian_filter1d(
                        m_flux_ij_shifted, sigma=300, mode='reflect'
                        )

                # Compute the cross-correlation coefficients, weighted 
                # by the covariance matrix
                if Cov is None:
                    CCF[i,j,k]   = np.nansum(m_flux_ij_shifted*d_flux_ij)# / np.isfinite((m_flux_ij_shifted*d_flux_ij)).sum()
                    m_ACF[i,j,k] = np.nansum(m_flux_ij_shifted*m_flux_ij_static)# / np.isfinite((m_flux_ij_shifted*m_flux_ij_static)).sum()
                    d_ACF[i,j,k] = np.nansum(d_flux_ij_shifted*d_flux_ij)# / np.isfinite((d_flux_ij_shifted*d_flux_ij)).sum()
                else:
                    CCF[i,j,k] = np.dot(
                        m_flux_ij_shifted, cov_ij.solve(d_flux_ij)
                        )
                    # Auto-correlation coefficients
                    m_ACF[i,j,k] = np.dot(
                        m_flux_ij_shifted, cov_ij.solve(m_flux_ij_static)
                        )
                    d_ACF[i,j,k] = np.dot(
                        d_flux_ij_shifted, cov_ij.solve(d_flux_ij)
                        )

            # Scale the correlation coefficients
            if LogLike is not None:
                CCF[i,j,:]   *= LogLike.f[i,j]/LogLike.beta[i,j]**2
                m_ACF[i,j,:] *= LogLike.f[i,j]/LogLike.beta[i,j]**2
                d_ACF[i,j,:] *= LogLike.f[i,j]/LogLike.beta[i,j]**2

    return rv, CCF, d_ACF, m_ACF

def get_PHOENIX_model(T, log_g, FeH=0, wave_range=(500,3000), PHOENIX_path='./data/PHOENIX/'):

    # Only certain combinations of parameters are allowed
    T_round     = int(200*np.round(T/200))
    log_g_round = 0.5*np.round(log_g/0.5)
    FeH_round   = 0.5*np.round(FeH/0.5)
    FeH_sign    = '+' if FeH>0 else '-'

    file_name = 'lte{0:05d}-{1:.2f}{2}{3:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
    file_name = file_name.format(
        T_round, log_g_round, FeH_sign, np.abs(FeH_round)
        )
    
    # Join the file name to the url
    url = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/' + file_name
    wave_url = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS//WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

    # Make the directory
    if not os.path.exists(PHOENIX_path):
        os.mkdir(PHOENIX_path)

    file_path = os.path.join(PHOENIX_path, file_name)
    wave_path = os.path.join(PHOENIX_path, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

    if not os.path.exists(wave_path):
        print(f'Downloading wavelength grid from {wave_url}')
        wget.download(wave_url, wave_path)

    if not os.path.exists(file_path):
        print(f'Downloading PHOENIX spectrum from {url}')
        wget.download(url, file_path)

    # Load the spectrum
    hdu = fits.open(file_path)
    PHOENIX_flux = hdu[0].data.astype(float) # erg s^-1 cm^-2 cm^-1

    hdu = fits.open(wave_path)
    PHOENIX_wave = hdu[0].data.astype(float) / 10 # Convert from A to nm

    mask_wave = (PHOENIX_wave > wave_range[0]) & \
        (PHOENIX_wave < wave_range[1])
    
    return PHOENIX_wave[mask_wave], PHOENIX_flux[mask_wave]

def read_results(prefix, n_params):

    import json
    import pymultinest

    # Set-up analyzer object
    analyzer = pymultinest.Analyzer(
        n_params=n_params, 
        outputfiles_basename=prefix
        )
    stats = analyzer.get_stats()

    # Load the equally-weighted posterior distribution
    posterior = analyzer.get_equal_weighted_posterior()
    posterior = posterior[:,:-1]

    # Read the parameters of the best-fitting model
    bestfit = np.array(stats['modes'][0]['maximum a posterior'])

    PT = pickle_load(f'{prefix}data/bestfit_PT.pkl')
    Chem = pickle_load(f'{prefix}data/bestfit_Chem.pkl')

    m_spec = pickle_load(f'{prefix}data/bestfit_m_spec.pkl')
    d_spec = pickle_load(f'{prefix}data/d_spec.pkl')

    LogLike = pickle_load(f'{prefix}data/bestfit_LogLike.pkl')

    try:
        Cov = pickle_load(f'{prefix}data/bestfit_Cov.pkl')
    except:
        Cov = None

    int_contr_em           = np.load(f'{prefix}data/bestfit_int_contr_em.npy')
    int_contr_em_per_order = np.load(f'{prefix}data/bestfit_int_contr_em_per_order.npy')
    int_opa_cloud          = np.load(f'{prefix}data/bestfit_int_opa_cloud.npy')

    f = open(prefix+'data/bestfit.json')
    bestfit_params = json.load(f)
    f.close()

    res = (
        posterior, 
        bestfit, 
        PT, 
        Chem, 
        int_contr_em, 
        int_contr_em_per_order, 
        int_opa_cloud, 
        m_spec, 
        d_spec, 
        LogLike, 
        Cov, 
        bestfit_params
        )

    return res