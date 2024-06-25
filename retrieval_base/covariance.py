import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

def get_Covariance_class(err, **kwargs):
    """
    Factory function to instantiate either a GaussianProcesses or Covariance class based on 'cov_mode'.

    Args:
    - err (np.ndarray): Array of uncertainties.
    - **kwargs: Additional keyword arguments passed to the chosen class.

    Returns:
    - Instance of either GaussianProcesses or Covariance class.
    """
    if kwargs.get('cov_mode') == 'GP':
        # Use a GaussianProcesses instance
        return GaussianProcesses(err, **kwargs)
    
    # Use a Covariance instance instead
    return Covariance(err, **kwargs)

class Covariance:
    """
    Class for handling covariance matrix operations.
    """
     
    def __init__(self, err, **kwargs):
        """
        Initialize Covariance class.

        Args:
        - err (np.ndarray): Array of uncertainties.
        - **kwargs: Additional keyword arguments.
        """
        # Set up the covariance matrix
        self.err = err
        self.cov_reset()

        # Initialize as None
        self.cov_cholesky = None

    def __call__(self, params, w_set, order, det, **kwargs):
        """
        Callable method to handle covariance matrix operations.

        Args:
        - params (dict): Parameters for covariance operation.
        - w_set (str): Key for accessing specific parameters.
        - order (int): Order of the parameter.
        - det (int): Detector number.

        Returns:
        - None
        """
        # Reset the covariance matrix
        self.cov_reset()

        if params.get(f'beta_{w_set}') is None:
            return
        
        if params[f'beta_{w_set}'][order,det] != 1:
            self.add_data_err_scaling(
                params[f'beta_{w_set}'][order,det]
                )

    def cov_reset(self):
        """
        Reset covariance matrix to initial state based on input uncertainties.

        Returns:
        - None
        """
        # Create the covariance matrix from the uncertainties
        self.cov = self.err**2
        self.is_matrix = (self.cov.ndim == 2)
        self.cov_shape = self.cov.shape

    def add_data_err_scaling(self, beta):
        """
        Scale the covariance matrix by a given factor beta.

        Args:
        - beta (float): Scaling factor.

        Returns:
        - None
        """
        # Scale the uncertainty with a (beta) factor
        if not self.is_matrix:
            self.cov *= beta**2
        else:
            self.cov[np.diag_indices_from(self.cov)] *= beta**2

    def add_model_err(self, model_err):
        """
        Add a model uncertainty term to the covariance matrix.

        Args:
        - model_err (float): Model uncertainty value.

        Returns:
        - None
        """
        # Add a model uncertainty term
        if not self.is_matrix:
            self.cov += model_err**2
        else:
            self.cov += np.diag(model_err**2)

    def get_logdet(self):
        """
        Calculate the log determinant of the covariance matrix.

        Returns:
        - logdet (float): Log determinant of the covariance matrix.
        """
        self.logdet = np.sum(np.log(self.cov))
        return self.logdet

    def solve(self, b):
        """
        Solve the system cov*x = b for x (x = cov^{-1}*b).

        Args:
        - b (np.ndarray): Right-hand side of cov*x = b.

        Returns:
        - x (np.ndarray): Solution x for the equation cov*x = b.
        """
        if self.is_matrix:
            return np.linalg.solve(self.cov, b)
        
        # Only invert the diagonal
        return 1/self.cov * b
    
    def get_dense_cov(self):
        """
        Return the dense representation of the covariance matrix.

        Returns:
        - cov (np.ndarray): Dense representation of the covariance matrix.
        """
        if self.is_matrix:
            return self.cov
        
        return np.diag(self.cov)

class GaussianProcesses(Covariance):
    """
    Class for handling Gaussian processes and associated covariance operations.
    """

    def get_banded(cls, array, max_value=None, pad_value=0, n_pixels=2048):
        """
        Create a banded covariance matrix representation.

        Args:
        - array (np.ndarray): Input array for covariance matrix.
        - max_value (float): Maximum value for the banded matrix.
        - pad_value (float): Padding value for the banded matrix.
        - n_pixels (int): Number of pixels.

        Returns:
        - banded_array (np.ndarray): Banded representation of the covariance matrix.
        """
        # Make banded covariance matrix
        banded_array = []

        for k in range(n_pixels):
            if array.dtype == object:
                # Array consists of order-detector pairs
                n_orders, n_dets = array.shape
                diag_k = []
                for i in range(n_orders):
                    for j in range(n_dets):
                        # Retrieve the k-th diagonal
                        diag_ijk = np.diag(array[i,j], k=k)
                        # Pad the diagonals to the same sizes
                        diag_ijk = np.concatenate((diag_ijk, pad_value*np.ones(k)))
                        # Append to diagonals of other order/detectors
                        diag_k.append(diag_ijk)
                diag_k = np.concatenate(diag_k)
            else:
                # Retrieve the k-th diagonal
                diag_k = np.diag(array, k=k)
                # Pad the diagonals to the same sizes
                diag_k = np.concatenate((diag_k, pad_value*np.ones(k)))
            
            if (diag_k == 0).all() and (k != 0):
                # There are no more non-zero diagonals coming
                break
            
            if max_value is not None:
                if (diag_k > max_value).all():
                    break
            
            # Only store the non-zero diagonals
            # Pad the diagonals to the same sizes
            banded_array.append(diag_k)
        
        # Convert to array for scipy
        banded_array = np.asarray(banded_array)
        return banded_array

    def __init__(self, err, separation, err_eff=None, flux_eff=None, max_separation=None, **kwargs):
        """
        Initialize GaussianProcesses class.

        Args:
        - err (np.ndarray): Array of uncertainties.
        - separation (np.ndarray): Separation between pixels.
        - err_eff (np.ndarray): Average squared error between pixels.
        - flux_eff (np.ndarray): Flux effectiveness.
        - max_separation (float): Maximum separation value.
        - **kwargs: Additional keyword arguments.
        """
        # Pre-computed average error and wavelength separation
        self.separation = np.abs(separation)
        self.err_eff  = err_eff
        self.flux_eff = flux_eff

        # Convert to banded matrices
        self.separation = self.get_banded(
            self.separation, max_value=max_separation, pad_value=1000, 
            )

        # Give arguments to the parent class
        super().__init__(err)

    def __call__(self, params, w_set, order=0, det=0, **kwargs):
        """
        Callable method to handle Gaussian processes and covariance operations.

        Args:
        - params (dict): Parameters for covariance operation.
        - w_set (str): Key for accessing specific parameters.
        - order (int): Order of the parameter.
        - det (int): Detector number.
        - **kwargs: Additional keyword arguments.

        Returns:
        - None
        """
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
        """
        Reset covariance matrix to initial state for Gaussian processes.

        Returns:
        - None
        """
        # Create the covariance matrix from the uncertainties
        self.cov = np.zeros_like(self.separation)
        self.cov[0] = self.err**2
        self.is_matrix = True

    def add_RBF_kernel(self, a, l, array, trunc_dist=5, scale_GP_amp=False, **kwargs):
        """
        Add a radial-basis function (RBF) kernel to the covariance matrix.

        Args:
        - a (float): Square-root of amplitude of the RBF kernel.
        - l (float): Length-scale of the RBF kernel.
        - array (np.ndarray): Input array for covariance matrix.
        - trunc_dist (float): Truncation distance for the kernel.
        - scale_GP_amp (bool): Flag to scale the amplitude of the kernel.

        Returns:
        - None
        """
        # Hann window function to ensure sparsity
        w_ij = (self.separation < trunc_dist*l)

        # GP amplitude
        GP_amp = a**2
        if scale_GP_amp:
            # Use amplitude as fraction of flux uncertainty
            if isinstance(array, float):
                GP_amp *= array**2
            else:
                GP_amp *= array[w_ij]**2

        # Gaussian radial-basis function kernel
        self.cov[w_ij] += GP_amp * np.exp(-(self.separation[w_ij])**2/(2*l**2))

    def get_cholesky(self):
        """
        Get the Cholesky decomposition of the covariance matrix.

        Returns:
        - None
        """
        self.cov = self.cov[(self.cov!=0).any(axis=1),:]
        # Compute banded Cholesky decomposition
        self.cov_cholesky = cholesky_banded(
            self.cov, lower=True, check_finite=False
            )

    def get_logdet(self):
        """
        Calculate the log determinant of the covariance matrix for Gaussian processes.

        Returns:
        - logdet (float): Log determinant of the covariance matrix.
        """
        self.logdet = 2*np.sum(np.log(self.cov_cholesky[0]))
        return self.logdet

    def solve(self, b):
        """
        Solve the system cov*x = b for x (x = cov^{-1}*b) for Gaussian processes.

        Args:
        - b (np.ndarray): Right-hand side of cov*x = b.

        Returns:
        - x (np.ndarray): Solution x for the equation cov*x = b.
        """
        return cho_solve_banded((self.cov_cholesky, True), b)
    
    def get_dense_cov(self):
        """
        Return the dense representation of the covariance matrix for Gaussian processes.

        Returns:
        - cov (np.ndarray): Dense representation of the covariance matrix.
        """
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