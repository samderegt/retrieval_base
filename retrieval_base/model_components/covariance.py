import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

from ..utils import sc

def get_class(d_spec, sum_model_settings=False, **kwargs):
    """
    Get a list of Covariance objects for each chip in the dataset.

    Args:
        d_spec (dict): Dictionary containing spectral data.
        sum_model_settings (bool, optional): If True, only compare to one model setting. Defaults to False.
        **kwargs: Additional keyword arguments for Covariance initialization.

    Returns:
        list: List of Covariance objects.
    """
    Cov = []
    for m_set in d_spec.keys():
        # Loop over each chip
        for d_wave, d_err in zip(d_spec[m_set].wave, d_spec[m_set].err):
            mask = np.isfinite(d_err)
            Cov.append(Covariance(d_wave[mask], d_err[mask], m_set, **kwargs))

        if sum_model_settings:
            # Only compare to one model setting
            break

    return Cov

class Covariance:
    """
    Class to handle covariance matrix operations.
    """
    
    @staticmethod
    def get_banded(array, max_value=np.inf, pad_value=0):
        """
        Convert a full matrix to a banded matrix.

        Args:
            array (np.ndarray): Input full matrix.
            max_value (float, optional): Maximum value to consider for banding. Defaults to np.inf.
            pad_value (float, optional): Value to pad the diagonals with. Defaults to 0.

        Returns:
            np.ndarray: Banded matrix.
        """
        banded_array = []
        for k in range(array.shape[0]):
            # Retrieve the k-th diagonal
            diag_k = np.diag(array, k=k)
            # Pad the diagonal to the same sizes
            diag_k = np.pad(diag_k, (0,k), constant_values=pad_value)

            if (diag_k == 0).all() and (k != 0):
                # There are no more non-zero diagonals coming
                break

            if (diag_k > max_value).all():
                # Only larger values are coming
                break

            banded_array.append(diag_k)

        # Convert to array for scipy
        return np.asarray(banded_array)

    def __init__(self, d_wave, d_err, m_set='all', trunc_dist=5, **kwargs):
        """
        Initialize the Covariance object.

        Args:
            d_wave (np.ndarray): Wavelength data.
            d_err (np.ndarray): Error data.
            m_set (str): Model setting identifier.
            trunc_dist (int, optional): Truncation distance for the covariance matrix. Defaults to 5.
            **kwargs: Additional keyword arguments.
        """
        # Model setting
        self.m_set = m_set

        # Which kernel to use (if any)
        self.kernel_mode     = kwargs.get('kernel_mode')
        self.separation_mode = kwargs.get('separation_mode', 'wave')

        # Set up the covariance matrix
        self.var = d_err**2
        
        # Scale the amplitude wrt the median variance
        self.scale_amp = kwargs.get('scale_amp', False) 
        # Truncation distance for the covariance matrix
        self.trunc_dist = trunc_dist

        # Convert wavelength separation to a banded matrix
        self.separation = np.abs(d_wave[None,:]-d_wave[:,None])
        
        if self.separation_mode == 'velocity':
            # Use velocity separation
            self.separation = (sc.c*1e-3) * self.separation / (np.abs(d_wave[None,:]+d_wave[:,None])/2)
        
        # Maximum separation/diagonal to consider for the covariance matrix
        max_separation = kwargs.get('max_wave_sep')
        if max_separation is None:
            max_separation = kwargs.get('max_separation')

        if max_separation is None:
            max_separation = np.max(self.separation)

        # Convert the separation to a banded matrix
        self.separation = self.get_banded(
            self.separation, max_value=max_separation, pad_value=1e6
        )

        # Initialize the covariance matrix
        self._reset_cov()

    def __call__(self, ParamTable, **kwargs):
        """
        Evaluate the covariance matrix with given parameters.

        Args:
            ParamTable (dict): Parameters for the model.
            **kwargs: Additional keyword arguments.
        """
        self._reset_cov()

        ParamTable.set_queried_m_set(
            'all', self.m_set, add_linked_m_set=True
            )

        b = ParamTable.get('b')
        if b is not None:
            # Multiply the error by a factor of 10^b
            self._multiply_err(b)
        
        amp    = ParamTable.get('a', 1.)
        length = ParamTable.get('l')
        if None in [length, self.kernel_mode]:
            # If either parameter is None, do not add a kernel
            self._get_cholesky()
            return
        
        if self.kernel_mode == 'rbf':
            self._add_radial_basis_function_kernel(amp=amp, length=length)
        elif self.kernel_mode == 'matern':
            self._add_matern_kernel(amp=amp, length=length)
        
        ParamTable.set_queried_m_set('all')

        self._get_cholesky()

    def solve(self, b):
        """
        Solve the system cov*x = b for x (x = cov^{-1}*b).

        Args:
            b (np.ndarray): Right-hand side of cov*x = b.

        Returns:
            np.ndarray: Solution x for the equation cov*x = b.
        """
        if np.squeeze(self.cov).ndim == 1:
            return b / np.squeeze(self.cov)
        
        return cho_solve_banded((self.cov_cholesky, True), b, check_finite=False)
    
    def _reset_cov(self):
        """
        Reset the covariance matrix to its initial state.
        """
        self.cov = np.zeros_like(self.separation, dtype=float)
        self.cov[0] = self.var.copy()

        if self.scale_amp:
            # Scale the amplitude wrt the median variance
            self.var_eff = np.median(self.var)
        
    def _get_cholesky(self):
        """
        Compute the Cholesky decomposition of the covariance matrix.
        """
        mask_nonzero_diag = (self.cov != 0).any(axis=1)

        if mask_nonzero_diag.sum()==1:
            # Only the diagonal is non-zero
            self.cov = self.cov[[0],:]
            self.cov_cholesky = np.sqrt(self.cov)

            # Log determinant of the covariance matrix
            self.logdet = 2*np.sum(np.log(self.cov_cholesky))
            return
        
        # Compute banded Cholesky decomposition on non-zero diagonals
        self.cov = self.cov[mask_nonzero_diag,:]
        self.cov_cholesky = cholesky_banded(self.cov, lower=True, check_finite=False)

        # Log determinant of the covariance matrix
        self.logdet = 2*np.sum(np.log(self.cov_cholesky[0]))

    def _multiply_err(self, b):
        """
        Multiply the error by a factor of 10^b. Or equivalently, 
        inflate the (co)-variance by a factor of 10^(2*b).
        
        Args:
            b (float): Factor to inflate the error.
        """
        self.cov *= 10**(2*b)
        if hasattr(self, 'var_eff'):
            self.var_eff *= 10**(2*b)

    def _add_radial_basis_function_kernel(self, amp, length):
        """
        Add a radial basis function kernel to the covariance matrix.

        Args:
            amp (float): Amplitude of the kernel.
            length (float): Length scale of the kernel.
        """
        # Hann window function to ensure sparsity
        w_ij = (self.separation < self.trunc_dist*length)

        # Normalised separation
        norm_sep = self.separation[w_ij] / length

        # GP amplitude
        var_eff = getattr(self, 'var_eff', 1.)
        amp = amp**2 * var_eff

        # Gaussian radial-basis function kernel
        self.cov[w_ij] += amp * np.exp(-1/2*norm_sep**2)

    def _add_matern_kernel(self, amp, length, nu=1.5):
        """
        Add a Matern kernel to the covariance matrix.

        Args:
            amp (float): Amplitude of the kernel.
            length (float): Length scale of the kernel.
            nu (float, optional): Smoothness parameter. Defaults to 1.5.
        """
        # Hann window function to ensure sparsity
        w_ij = (self.separation < self.trunc_dist*length)
        
        # Normalised separation
        norm_sep = self.separation[w_ij] / length

        # GP amplitude
        var_eff = getattr(self, 'var_eff', 1.)
        amp = amp**2 * var_eff

        # Matern kernel
        if nu == 0.5:
            self.cov[w_ij] += amp * np.exp(-norm_sep)
        elif nu == 1.5:
            self.cov[w_ij] += amp * (1 + np.sqrt(3)*norm_sep) * np.exp(-np.sqrt(3)*norm_sep)
        elif nu == 2.5:
            self.cov[w_ij] += (
                amp * (1 + np.sqrt(5)*norm_sep + 5/3*norm_sep**2) * np.exp(-np.sqrt(5)*norm_sep)
            )
        else:
            raise ValueError(f"Unsupported nu value: {nu}. Supported values are 0.5, 1.5, and 2.5.")
        