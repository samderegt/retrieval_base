import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

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
            Cov.append(Covariance(d_wave[mask], d_err[mask], **kwargs))

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

    def __init__(self, d_wave, d_err, trunc_dist=5, max_wave_sep=None, scale_amp=False, **kwargs):
        """
        Initialize the Covariance object.

        Args:
            d_wave (np.ndarray): Wavelength data.
            d_err (np.ndarray): Error data.
            trunc_dist (int, optional): Truncation distance for the covariance matrix. Defaults to 5.
            max_wave_sep (float, optional): Maximum wavelength separation. Defaults to None.
            scale_amp (bool, optional): If True, scale the amplitude. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        # Set up the covariance matrix
        self.var = d_err**2

        self.trunc_dist = trunc_dist
        if scale_amp:
            self.var_eff = np.median(self.var)

        # Convert wavelength separation to a banded matrix
        self.wave_sep = np.abs(d_wave[None,:] - d_wave[:,None])
        if max_wave_sep is None:
            max_wave_sep = np.max(np.diff(d_wave))

        self.wave_sep = self.get_banded(self.wave_sep, max_value=max_wave_sep, pad_value=1000)

        self._reset_cov()

    def __call__(self, ParamTable, **kwargs):
        """
        Evaluate the covariance matrix with given parameters.

        Args:
            ParamTable (dict): Parameters for the model.
            **kwargs: Additional keyword arguments.
        """
        self._reset_cov()
        
        amp    = ParamTable.get('a')
        length = ParamTable.get('l')
        if None not in [amp, length]:
            self._add_radial_basis_function_kernel(amp=amp, length=length)
            
        self._get_cholesky()

    def solve(self, b):
        """
        Solve the system cov*x = b for x (x = cov^{-1}*b).

        Args:
            b (np.ndarray): Right-hand side of cov*x = b.

        Returns:
            np.ndarray: Solution x for the equation cov*x = b.
        """
        if self.cov.ndim == 1:
            return b / self.cov
        
        return cho_solve_banded((self.cov_cholesky, True), b, check_finite=False)
    
    def _reset_cov(self):
        """
        Reset the covariance matrix to its initial state.
        """
        self.cov = np.zeros_like(self.wave_sep)
        self.cov[0] = self.var.copy()
        
    def _get_cholesky(self):
        """
        Compute the Cholesky decomposition of the covariance matrix.
        """
        mask_nonzero_diag = (self.cov != 0).any(axis=1)

        if mask_nonzero_diag.sum()==1:
            # Only the diagonal is non-zero
            self.cov = self.cov[0]
            self.cov_cholesky = np.sqrt(self.cov)

            # Log determinant of the covariance matrix
            self.logdet = 2*np.sum(np.log(self.cov_cholesky))
            return
        
        # Compute banded Cholesky decomposition on non-zero diagonals
        self.cov = self.cov[mask_nonzero_diag,:]
        self.cov_cholesky = cholesky_banded(self.cov, lower=True, check_finite=False)

        # Log determinant of the covariance matrix
        self.logdet = 2*np.sum(np.log(self.cov_cholesky[0]))

    def _add_radial_basis_function_kernel(self, amp, length):
        """
        Add a radial basis function kernel to the covariance matrix.

        Args:
            amp (float): Amplitude of the kernel.
            length (float): Length scale of the kernel.
        """
        # Hann window function to ensure sparsity
        w_ij = (self.wave_sep < self.trunc_dist*length)

        # GP amplitude
        var_eff = getattr(self, 'var_eff', 1.)
        amp = amp**2 * var_eff

        # Gaussian radial-basis function kernel
        self.cov[w_ij] += amp * np.exp(-(self.wave_sep[w_ij])**2/(2*length**2))