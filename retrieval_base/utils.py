import pickle

def save_pickle(obj, filename):
    """Save an object to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    """Load an object from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)