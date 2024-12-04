import numpy as np
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

    fig = plt.figure(figsize=(10,3*N))
    gs = fig.add_gridspec(nrows=N)
    subfig = np.array([fig.add_subfigure(gs[i]) for i in range(N)])

    return fig, subfig