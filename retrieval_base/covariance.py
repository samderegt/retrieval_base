import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

def get_Covariance_class(err, **kwargs):

    if kwargs.get('cov_mode') == 'GP':
        # Use a GaussianProcesses instance
        return GaussianProcesses(err, **kwargs)
    
    # Use a Covariance instance instead
    return Covariance(err, **kwargs)

class Covariance:
     
    def __init__(self, err, **kwargs):

        # Set-up the covariance matrix
        self.err = err
        self.cov_reset()

        # Set to None initially
        self.cov_cholesky = None

    def __call__(self, params, w_set, order, det, **kwargs):

        # Reset the covariance matrix
        self.cov_reset()

        if params.get(f'beta_{w_set}') is None:
            return
        
        if params[f'beta_{w_set}'][order,det] != 1:
            self.add_data_err_scaling(
                params[f'beta_{w_set}'][order,det]
                )

    def cov_reset(self):

        # Create the covariance matrix from the uncertainties
        self.cov = self.err**2
        self.is_matrix = (self.cov.ndim == 2)

        self.cov_shape = self.cov.shape

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
        
        if self.is_matrix:
            return np.linalg.solve(self.cov, b)
            
        # Only invert the diagonal
        return 1/self.cov * b
    
    def get_dense_cov(self):

        if self.is_matrix:
            return self.cov
        
        return np.diag(self.cov)

class GaussianProcesses(Covariance):

    def get_banded(cls, array, max_value=None):

        # Make banded covariance matrix
        banded_array = []

        for i in range(len(array)):
            # Retrieve the i-th diagonal
            diag_i = np.diag(array, k=i)

            if (diag_i == 0).all() and (i != 0):
                # There are no more non-zero diagonals coming
                break
            
            if max_value is not None:
                if (diag_i > max_value).all():
                    break

            # Only store the non-zero diagonals
            # Pad the diagonals to the same sizes
            banded_array.append(
                np.concatenate((diag_i, np.zeros(i)))
                )
        
        # Convert to array for scipy
        banded_array = np.asarray(banded_array)

        return banded_array

    def __init__(self, err, separation, err_eff=None, flux_eff=None, max_separation=None, **kwargs):
        '''
        Create a covariance matrix suited for Gaussian processes. 

        Input
        -----
        err : np.ndarray
            Uncertainty in the flux.
        separation : np.ndarray
            Separation between pixels, can be in units of wavelength, 
            pixels, or velocity.
        err_eff : np.ndarray
            Average squared error between pixels.
        '''
        
        # Pre-computed average error and wavelength separation
        self.separation = np.abs(separation)
        self.err_eff  = err_eff
        self.flux_eff = flux_eff

        # Convert to banded matrices
        self.separation = self.get_banded(
            self.separation, max_value=max_separation
            )
        if isinstance(self.err_eff, np.ndarray):
            self.err_eff = self.get_banded(self.err_eff)
            self.err_eff = self.err_eff[:self.separation.shape[0]]

        if isinstance(self.flux_eff, np.ndarray):
            self.flux_eff = self.get_banded(self.flux_eff)
            self.flux_eff = self.flux_eff[:self.separation.shape[0]]

        # Give arguments to the parent class
        super().__init__(err)

    def __call__(self, params, w_set, order, det, **kwargs):

        # Reset the covariance matrix
        self.cov_reset()
        
        beta = params.get('beta', params.get(f'beta_{w_set}'))
        a = params.get('a', params.get(f'a_{w_set}'))
        l = params.get('l', params.get(f'l_{w_set}'))

        if beta is not None:
            self.add_data_err_scaling(
                params[f'beta_{w_set}'][order,det]
                )
        
        if (a is not None) and (l is not None):
            # Add a radial-basis function kernel
            self.add_RBF_kernel(
                a=a[order,det], 
                l=l[order,det], 
                array=self.err_eff, 
                **kwargs
                )

    def cov_reset(self):

        # Create the covariance matrix from the uncertainties
        self.cov = np.zeros_like(self.separation)
        self.cov[0] = self.err**2

        self.is_matrix = True

    def add_RBF_kernel(self, a, l, array, trunc_dist=5, scale_GP_amp=False, **kwargs):
        '''
        Add a radial-basis function kernel to the covariance matrix. 
        The amplitude can be scaled by the flux-uncertainties of 
        pixels i and j if scale_GP_amp=True. 

        Input
        -----
        a : float
            Square-root of amplitude of the RBF kernel.
        l : float
            Length-scale of the RBF kernel.
        trunc_dist : float
            Distance at which to truncate the kernel 
            (|wave_i-wave_j| < trunc_dist*l). This ensures
            a relatively sparse covariance matrix. 
        scale_GP_amp : bool
            If True, scale the amplitude at each covariance element, 
            using the flux-uncertainties of the corresponding pixels
            (A = a**2 * (err_i**2 + err_j**2)/2).
        '''

        # Hann window function to ensure sparsity
        w_ij = (self.separation < trunc_dist*l)

        # GP amplitude
        GP_amp = a**2
        if scale_GP_amp:
            # Use amplitude as fraction of flux uncertainty
            '''
            if isinstance(self.err_eff, float):
                GP_amp *= self.err_eff**2
            else:
                GP_amp *= self.err_eff[w_ij]**2
            '''
            if isinstance(array, float):
                GP_amp *= array**2
            else:
                GP_amp *= array[w_ij]**2

        # Gaussian radial-basis function kernel
        self.cov[w_ij] += GP_amp * np.exp(-(self.separation[w_ij])**2/(2*l**2))

    def get_cholesky(self):
        '''
        Get the Cholesky decomposition. Employs a banded 
        decomposition with scipy. 
        '''
        '''
        cov_full = np.zeros(self.cov_shape, dtype=np.float64)
        cov_full[self.max_separation_mask] = self.cov

        # Make banded covariance matrix
        self.cov_banded = []

        for i in range(len(cov_full)):
            # Retrieve the i-th diagonal
            diag_i = np.diag(cov_full, k=i)

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
        self.cov_banded = np.asarray(self.cov_banded)
        '''
        self.cov = self.cov[(self.cov!=0).any(axis=1),:]

        # Compute banded Cholesky decomposition
        self.cov_cholesky = cholesky_banded(
            self.cov, lower=True
            )

    def get_logdet(self):
        '''
        Calculate the log of the determinant. Uses diagonal 
        elements of banded Cholesky decomposition.
        '''

        self.logdet = 2*np.sum(np.log(self.cov_cholesky[0]))

    def solve(self, b):
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

        return cho_solve_banded((self.cov_cholesky, True), b)
    
    def get_dense_cov(self):
        
        # Full covariance matrix
        cov_full = np.zeros((self.cov.shape[1], self.cov.shape[1]))
        
        for i, diag_i in enumerate(self.cov):

            if i != 0:
                diag_i = diag_i[:-i]

            # Fill upper diagonals
            cov_full += np.diag(diag_i, k=i)
            if i != 0:
                # Fill lower diagonals
                cov_full += np.diag(diag_i, k=-i)

        return cov_full