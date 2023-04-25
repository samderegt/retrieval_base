import pickle
import os
import shutil

import numpy as np

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
    