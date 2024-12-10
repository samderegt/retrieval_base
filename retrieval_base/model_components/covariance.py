import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

def get_class(d_spec, sum_model_settings=False, **kwargs):
    
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
    
    @staticmethod
    def get_banded(array, max_value=np.inf, pad_value=0):
        
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

        self.reset_cov()

    def reset_cov(self):
        self.cov = np.zeros_like(self.wave_sep)
        self.cov[0] = self.var.copy()
        
    def get_cholesky(self):

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

    def solve(self, b):
        """
        Solve the system cov*x = b for x (x = cov^{-1}*b).

        Parameters:
        b (np.ndarray): 
            Right-hand side of cov*x = b.

        Returns:
        np.ndarray: 
            Solution x for the equation cov*x = b.
        """
        if self.cov.ndim == 1:
            return b / self.cov
        
        return cho_solve_banded((self.cov_cholesky, True), b, check_finite=False)

    def add_radial_basis_function_kernel(self, amp, length):

        # Hann window function to ensure sparsity
        w_ij = (self.wave_sep < self.trunc_dist*length)

        # GP amplitude
        var_eff = getattr(self, 'var_eff', 1.)
        amp = amp**2 * var_eff

        # Gaussian radial-basis function kernel
        self.cov[w_ij] += amp * np.exp(-(self.wave_sep[w_ij])**2/(2*length**2))

    def __call__(self, ParamTable, **kwargs):

        self.reset_cov()
        
        amp    = ParamTable.get('a')
        length = ParamTable.get('l')
        if None not in [amp, length]:
            self.add_radial_basis_function_kernel(amp=amp, length=length)
            
        self.get_cholesky()