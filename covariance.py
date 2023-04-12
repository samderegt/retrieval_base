import numpy as np
from scipy.sparse import csc_matrix

from sksparse.cholmod import cholesky

class Covariance:
     
    def __init__(self, err):

        self.cov = err**2
        self.is_matrix = (self.cov.ndim == 2)

        # Set to None initially
        self.cov_cholesky = None

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

    def solve(self, b):
        '''
        Solve the system cov*x = b, for x (x = cov^{-1}*b).

        Input
        -----
        b : np.ndarray
            Righthand-side of cov*x = b.
        
        Returns
        -------
        '''
        
        if not self.is_matrix:
            # Only invert the diagonal
            return 1/self.cov * b


class GaussianProcesses(Covariance):

    def __init__(self, err, is_sparse=True):

        # Give arguments to the parent class
        super().__init__(err)

        # Make covariance 2-dimensional
        self.cov = np.diag(self.cov)
        self.is_matrix = True

        self.is_sparse = is_sparse

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

        if self.is_sparse:
            # Create a sparse CSC matrix
            self.cov = csc_matrix(self.cov)

        # Retrieve a (sparse) Cholesky decomposition
        self.get_cholesky()
        
    def get_cholesky(self):

        if self.is_sparse:
            # Calculate the sparse Cholesky decomposition
            self.cov_cholesky = cholesky(self.cov)

            # Calculate the log of the determinant
            self.logdet = self.cov_cholesky.logdet()

            # Set the solve function 
            self.solve = self.solve_sparse_cholesky
        else:
            # TODO: non-sparse decomposition
            pass

    def solve_sparse_cholesky(self, b):
        '''
        Solve the system cov*x = b, for x (x = cov^{-1}*b). 
        Employs a sparse cholesky decomposition.

        Input
        -----
        b : np.ndarray
            Righthand-side of cov*x = b.
        
        Returns
        -------
        '''

        return self.cov_cholesky.solve_A(b)