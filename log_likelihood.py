import numpy as np

from .covariance import Covariance, GaussianProcesses

class LogLikelihood:

    def __init__(self, data_spec):

        self.data_spec = data_spec

    def __call__(self, model_spec, params):
        
        self.params = params
        self.model_spec = model_spec
        
        # Loop over all orders and detectors
        for i in range(self.data_spec.n_orders):
            for j in range(self.data_spec.n_dets):
                # ...
                pass