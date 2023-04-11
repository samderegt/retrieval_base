import numpy as np

class Parameters:

    def __init__(self, param_priors, param_constant, param_mathtext):

        self.param_priors   = param_priors
        self.param_constant = param_constant
        self.param_mathtext = param_mathtext

        
        pass

    def __call__(self, cube, ndim, nparams):
        # Apply the prior boundaries

        return