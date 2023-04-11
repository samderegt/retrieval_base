import numpy as np
from scipy.sparse import csc_matrix

from sksparse.cholmod import cholesky

class Covariance:
     
    def __init__(self, err):

        self.cov = err**2
        self.is_matrix = (self.cov.ndim == 2)

    def add_data_err_scaling(self, beta):

        # Scale the uncertainty with a (beta) factor
        if not self.is_matrix:
            self.cov *= beta**2
        else:
            self.cov[np.diag_indices_from(self.cov)] *= beta**2

    def add_model_err(self, model_err):

        # Add a model uncertainty term
        if not self.is_matrix:
            self.cov += model_err**2
        else:
            self.cov += np.diag(model_err**2)

    def get_logdet(self):

        # Calculate the log of the determinant
        self.logdet = np.sum(np.log(self.cov))


def GaussianProcesses(Covariance):

    def __init__(self, err):

        # Give arguments to the parent class
        super().__init__(err)

        # Make covariance 2-dimensional
        self.cov = np.diag(self.cov)
        self.is_matrix = True

    def add_RBF_kernel(self, a, l, delta_wave, trunc_dist=3):

        # Hann window function to ensure sparsity
        w_ij = (delta_wave < trunc_dist*l)

        # Gaussian radial-basis function kernel
        Sigma_ij = np.zeros_like(delta_wave)
        Sigma_ij[w_ij] = a**2 * np.exp(-(delta_wave[w_ij])**2/(2*l**2))
        
        # Add the (scaled) Poisson noise
        if self.is_matrix:
            self.cov = Sigma_ij + self.cov
        else:
            self.cov = Sigma_ij + np.diag(self.cov)
            self.is_matrix = True

        # Create a sparse CSC matrix
        self.cov = csc_matrix(self.cov)
        #Sigma_ij_sparse = Sigma_ij

    def get_sparse_cholesky(self):

        # Calculate the sparse Cholesky decomposition
        self.cov_cholesky = cholesky(self.cov)

        # Calculate the log of the determinant
        self.logdet = self.cov_cholesky.logdet()

    def get_cholesky(self):

        # TODO: non-sparse cholesky decomposition
        pass