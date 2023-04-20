import numpy as np
from scipy.sparse import csc_matrix
from scipy.linalg import cholesky_banded, cho_solve_banded

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
        x : np.ndarray

        '''
        
        if not self.is_matrix:
            # Only invert the diagonal
            return 1/self.cov * b
        else:
            return np.linalg.solve(self.cov, b)


class GaussianProcesses(Covariance):

    def __init__(self, err, cholesky_mode='banded'):

        # Give arguments to the parent class
        super().__init__(err)
        self.err = err

        # Make covariance 2-dimensional
        self.cov = np.diag(self.cov)
        self.is_matrix = True

        self.cholesky_mode = cholesky_mode

    def add_RBF_kernel(self, a, l, wave, trunc_dist=5, scale_GP_amp=False):

        if not self.is_matrix:
            # Create diagonal matrix
            self.cov = np.diag(self.cov)
            self.is_matrix = True

        # Wavelength separation between pixels
        delta_wave = np.abs(wave[None,:] - wave[:,None])

        # Hann window function to ensure sparsity
        w_ij = (delta_wave < trunc_dist*l)

        if scale_GP_amp:
            # Use amplitude as fraction of flux uncertainty
            GP_amp = a**2 * 1/2*(self.err[None,:]**2 + self.err[:,None]**2)[w_ij]

            # Geometric mean
            #GP_amp = a**2 * np.sqrt(err[None,:]**2 * err[:,None]**2)[w_ij]
        else:
            GP_amp = a**2

        # Gaussian radial-basis function kernel
        self.cov[w_ij] = self.cov[w_ij] + GP_amp * np.exp(-(delta_wave[w_ij])**2/(2*l**2))
        
        '''
        # Gaussian radial-basis function kernel
        Sigma_ij = np.zeros_like(delta_wave)
        Sigma_ij[w_ij] = a**2 * np.exp(-(delta_wave[w_ij])**2/(2*l**2))

        if err is not None:
            # Use amplitude as fraction of flux uncertainty
            Sigma_ij[w_ij] *= 1/2*(err[None,:]**2 + err[:,None]**2)[w_ij]

            # Geometric mean
            #GP_amp = a**2 * np.sqrt(err[i]**2 * err[j]**2)
        
        # Add the (scaled) Poisson noise
        if self.is_matrix:
            self.cov = Sigma_ij + self.cov
        else:
            self.cov = Sigma_ij + np.diag(self.cov)
            self.is_matrix = True
        '''

        if self.cholesky_mode == 'sparse':
            # Create a sparse CSC matrix
            self.cov = csc_matrix(self.cov)

        elif self.cholesky_mode == 'banded':
            # Banded Cholesky decomposition
            self.cov_banded = []

            for i in range(len(self.cov)):
                # Retrieve the i-th diagonal
                diag_i = np.diag(self.cov, k=i)

                if (diag_i != 0).any():
                    # Only store the non-zero diagonals
                    # Pad the diagonals to the same sizes
                    self.cov_banded.append(
                        np.concatenate((diag_i, np.zeros(i)))
                        )
                else:
                    # There are no more non-zero diagonals coming
                    break
            
            # Convert to array for scipy
            self.cov_banded = np.array(self.cov_banded)

        # Retrieve a (sparse) Cholesky decomposition
        self.get_cholesky()
        
    def get_cholesky(self):

        if self.cholesky_mode == 'sparse':
            # Compute sparse Cholesky decomposition
            self.cov_cholesky = cholesky(self.cov)

        elif self.cholesky_mode == 'banded':
            # Compute banded Cholesky decomposition
            self.cov_cholesky = cholesky_banded(
                self.cov_banded, lower=True
                )
            
        # Set the solve function
        self.solve = self.solve_cholesky

    def get_logdet(self):
        
        # Calculate the log of the determinant
        if self.cholesky_mode == 'sparse':
            # Sparse Cholesky decomposition
            self.logdet = self.cov_cholesky.logdet()

        elif self.cholesky_mode == 'banded':
            # Use diagonal elements of banded Cholesky decomposition
            self.logdet = 2*np.sum(np.log(self.cov_cholesky[0]))

    def solve_cholesky(self, b):
        '''
        Solve the system cov*x = b, for x (x = cov^{-1}*b). 
        Employs a sparse or banded Cholesky decomposition.

        Input
        -----
        b : np.ndarray
            Righthand-side of cov*x = b.
        
        Returns
        -------
        x : np.ndarray

        '''

        if self.cholesky_mode == 'sparse':
            # Sparse Cholesky decomposition   
            return self.cov_cholesky.solve_A(b)

        elif self.cholesky_mode == 'banded':
            # Banded Cholesky decomposition
            return cho_solve_banded((self.cov_cholesky, True), b)