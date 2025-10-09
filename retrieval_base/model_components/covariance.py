import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

from ..utils import sc

def get_class(**kwargs):
    """
    Returns an instance of Covariance class.

    Args:
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Covariance: An instance of Covariance class.
    """
    return Covariance(**kwargs)

class CovarianceMatrix:

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
    
    def __init__(self, d_wave, d_err, trunc_dist=5, separation_mode='wave', scale_amp=False, **kwargs):
        """
        Initialize the CovarianceMatrix object.
        
        Args:
            d_wave (list of np.ndarray): List of wavelength arrays for each chip.
            d_err (list of np.ndarray): List of error arrays for each chip.
            trunc_dist (float, optional): Truncation distance for kernels. Defaults to 5.
            separation_mode (str, optional): Mode of separation ('wave' or 'velocity'). Defaults to 'wave'.
            scale_amp (bool, optional): Whether to scale the amplitude wrt the median variance. Defaults to False.
            **kwargs: Additional keyword arguments.
        """

        # Maximum separation/diagonal to consider for the covariance matrix
        max_separation = kwargs.get('max_wave_sep', kwargs.get('max_separation', np.inf))

        self.separation_mode = separation_mode
        self.scale_amp  = scale_amp   # Scale the amplitude wrt the median variance
        self.trunc_dist = trunc_dist

        # Set up covariance matrix
        mask = np.isfinite(d_err)
        self.var = d_err[mask]**2

        self.separation = np.abs(d_wave[mask][None,:]-d_wave[mask][:,None])
        if separation_mode == 'velocity':
            # Use velocity separation
            self.separation = (sc.c*1e-3) * self.separation / (np.abs(d_wave[mask][None,:]+d_wave[mask][:,None])/2)

        # Convert the separation to a banded matrix
        self.separation = self.get_banded(self.separation, max_value=max_separation, pad_value=1e6)

        # Initialize the covariance matrix
        self._reset_cov()

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
        Compute the Cholesky decomposition of the covariance matrices.
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

class Covariance:
    """
    Class to handle covariance matrix operations.
    """

    def __init__(self, d_spec, m_set, kernel_mode=None, **kwargs):
        """
        Initialize the Covariance object.

        Args:
            d_spec (object): Data spectrum object containing wavelength and error arrays.
            m_set (str): Model setting identifier.
            kernel_mode (str, optional): Type of kernel to use ('rbf' or 'matern'). Defaults to None.
            **kwargs: Additional keyword arguments.
            
        """
        # Model setting
        self.m_set = m_set

        # Set up the covariance matrices for each chip
        self.CovMats = [
            CovarianceMatrix(d_wave=d_wave_i, d_err=d_err_i, **kwargs)
            for d_wave_i, d_err_i in zip(d_spec.wave, d_spec.err)
        ]

        self.kernel_mode = kernel_mode

    def __call__(self, ParamTable):
        """
        Evaluate the covariance matrix with given parameters.

        Args:
            ParamTable (dict): Parameters for the model.
        """
        # Reset the covariance matrices
        [CM._reset_cov() for CM in self.CovMats]

        # Read the parameters
        b, amp, length = self._read_params(ParamTable)

        if b is not None:
            # Multiply the error by a factor of 10^b
            [CM._multiply_err(b) for CM in self.CovMats]

        if None in [length, self.kernel_mode]:
            # If either parameter is None, do not add a kernel
            [CM._get_cholesky() for CM in self.CovMats]
            return
        
        if self.kernel_mode == 'rbf':
            [CM._add_radial_basis_function_kernel(amp=amp, length=length) for CM in self.CovMats]
        elif self.kernel_mode == 'matern':
            [CM._add_matern_kernel(amp=amp, length=length) for CM in self.CovMats]

        # Compute the Cholesky decomposition
        [CM._get_cholesky() for CM in self.CovMats]

    def __iter__(self):
        return iter(self.CovMats)
    
    def __len__(self):
        return len(self.CovMats)
    
    def __getitem__(self, index):
        return self.CovMats[index]

    def _read_params(self, ParamTable):
        """
        Get the parameters from the ParamTable.

        Args:
            ParamTable (dict): Parameter table.
        """
        ParamTable.set_queried_m_set(
            'all', self.m_set, add_linked_m_set=False
            )
        b = ParamTable.get('b')
        
        amp    = ParamTable.get('a', 1.)
        length = ParamTable.get('l')

        # Also check the linked model settings
        ParamTable.set_queried_m_set(
            'all', self.m_set, add_linked_m_set=True
            )
        if b is None:
            b = ParamTable.get('b')
        if amp == 1.:
            amp = ParamTable.get('a', 1.)
        if length is None:
            length = ParamTable.get('l')
        
        ParamTable.set_queried_m_set('all')
        return b, amp, length
    