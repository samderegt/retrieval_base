import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

import scipy.constants as sc
sc.r_jup_mean = 69911.0e3 # [m]
sc.m_jup      = 1.899e27  # [kg]
sc.amu = sc.m_u # [kg]
    
def save_pickle(obj, filename):
    """Save an object to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    """Load an object from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def get_subfigures_per_chip(N):

    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.3

    fig = plt.figure(figsize=(10,3*N))
    gs = fig.add_gridspec(nrows=N)
    subfig = np.array([fig.add_subfigure(gs[i]) for i in range(N)])

    return fig, subfig

def print_bestfit_params(ParamTable, LogLike):
    """Print the best-fit parameters."""

    print('\nBest-fitting free parameters:')
    for idx, (idx_free) in enumerate(ParamTable.table['idx_free']):
        if pd.isna(idx_free):
            # Not a free parameter
            continue

        name, m_set, val = ParamTable.table.loc[idx][['name','m_set','val']]
        print(f'{name} ({m_set})'.ljust(30,' ') + f' = {val:.2f}')

    print('\nChi-squared (w/o covariance-scaling): {:.2f}'.format(LogLike.chi_squared_0_red))
    if not LogLike.scale_flux:
        return
    print('\nOptimal flux-scaling parameters:')
    print(LogLike.phi.round(2))
    
    print('R_p (R_Jup):')
    print(np.sqrt(LogLike.phi).round(2))