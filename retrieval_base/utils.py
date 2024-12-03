import pickle

import scipy.constants as sc
sc.r_jup_mean = 69911.0e3 # [m]
sc.m_jup      = 1.899e27  # [kg]

def save_pickle(obj, filename):
    """Save an object to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    """Load an object from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)