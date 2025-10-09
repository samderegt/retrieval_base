import numpy as np
from scipy.special import loggamma

def get_class(**kwargs):
    """
    Returns an instance of LogLikelihood class.

    Args:
        **kwargs: Arbitrary keyword arguments.

    Returns:
        LogLikelihood: An instance of LogLikelihood class.
    """
    return LogLikelihood(**kwargs)

class LogLikelihood:
    def __init__(self, sum_model_settings=True, scale_flux=False, scale_err=False, alpha=2, N_phi=1, scale_relative_to_chip=-1, **kwargs):
        """
        Initializes the LogLikelihood class.

        Args:
            sum_model_settings (bool, optional): Whether to sum model settings. Defaults to True.
            scale_flux (bool, optional): Whether to scale flux. Defaults to False.
            scale_err (bool, optional): Whether to scale error. Defaults to False.
            alpha (int, optional): Alpha parameter. Defaults to 2.
            N_phi (int, optional): Number of phi. Defaults to 1.
            scale_relative_to_chip (int, optional): Scale relative to chip. Defaults to -1.
            **kwargs: Arbitrary keyword arguments.
        """
        # Combine all model settings
        self.sum_model_settings = sum_model_settings

        self.scale_flux = scale_flux
        self.scale_err  = scale_err

        self.alpha = alpha
        self.N_phi = N_phi
        self.scale_relative_to_chip = scale_relative_to_chip

    def _combine_model_settings(self, d_spec, m_spec):

        d_flux = {m_set: d_spec_i.flux for m_set, d_spec_i in d_spec.items()}
        m_flux = {m_set: 0 for m_set in d_spec.keys()}
        for m_set_i, d_spec_i in d_spec.items():

            if m_set_i not in m_flux:
                # Already summed into another model setting
                continue

            if not self.sum_model_settings:
                # No need to sum
                m_flux[m_set_i] = m_spec[m_set_i].flux_binned
                continue

            # Check if other model settings share wavelengths and sum
            for m_set_j, d_spec_j in d_spec.items():
                if np.array_equal(d_spec_i.wave, d_spec_j.wave):
                    m_flux[m_set_i] += m_spec[m_set_j].flux_binned
                    if m_set_j != m_set_i:
                        # Remove summed model setting
                        m_flux.pop(m_set_j, None)

        return m_flux, d_flux

    def __call__(self, d_spec, m_spec, Cov):

        self.ln_L = 0.
        self.chi_squared_0 = 0.
        self.N_d = 0

        # Sum the model spectra if requested
        m_flux, d_flux = self._combine_model_settings(d_spec, m_spec)

        self.m_flux_phi = {}
        self.phi, self.s_squared = {}, {}

        for m_set in m_flux.keys():
            self.m_flux_phi[m_set] = [np.nan] * len(m_flux[m_set])
            self.phi[m_set]        = [1.] * len(m_flux[m_set])
            self.s_squared[m_set]  = [1.] * len(m_flux[m_set])

            for i, (m_flux_i, d_flux_i, Cov_i) in enumerate(zip(m_flux[m_set], d_flux[m_set], Cov[m_set])):

                apply_scaling = self.scale_flux & (i != self.scale_relative_to_chip)

                # Number of data points
                mask = np.isfinite(d_flux_i)
                N_d  = mask.sum()
                self.N_d += N_d
                if N_d == 0:
                    continue

                self.m_flux_phi[m_set][i] = m_flux_i
                if apply_scaling:
                    # Find optimal flux-scaling to match observations
                    _, self.phi[m_set][i] = self._get_flux_scaling(d_flux_i[mask], m_flux_i[mask], Cov_i)
                    self.m_flux_phi[m_set][i] = np.dot(m_flux_i, self.phi[m_set][i])

                # Residuals compared to scaled model
                residuals = d_flux_i - self.m_flux_phi[m_set][i]
                
                # Chi-squared for optimal linear scaling
                chi_squared_0 = np.dot(residuals[mask].T, Cov_i.solve(residuals[mask]))
                self.chi_squared_0 += chi_squared_0

                # Get the log of the determinant
                logdet_cov_0 = Cov_i.logdet

                if (not apply_scaling) and (not self.scale_err):
                    # No scaling, use simple log-likelihood
                    self.ln_L += -1/2*(
                        N_d*np.log(2*np.pi) + logdet_cov_0 + chi_squared_0
                    )
                    continue

                logdet_MT_inv_cov_0_M = 0
                if apply_scaling:
                    # Covariance matrix of phi
                    MT_inv_cov_0_M = np.dot(m_flux_i[mask].T, Cov_i.solve(m_flux_i[mask]))

                    # Take the (log)-determinant of the phi-covariance matrix
                    # This introduces uncertainty on phi into log-likelihood
                    logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M) # 1x1 matrix (i.e. scalar)

                if self.scale_err:
                    # Scale the variance to maximise log-likelihood
                    self.s_squared[m_set][i] = self._get_err_scaling(chi_squared_0, N_d)

                # Constant term for this order/detector
                self.ln_L += -1/2*(N_d-self.N_phi) * np.log(2*np.pi) + \
                    loggamma(1/2*(N_d-self.N_phi+self.alpha-1))

                # Add this order/detector to the total log-likelihood
                self.ln_L += -1/2*(
                    logdet_cov_0 + logdet_MT_inv_cov_0_M + \
                    (N_d-self.N_phi+self.alpha-1) * np.log(chi_squared_0)
                )

        # Reduced chi-squared without optimal variance-scaling
        self.chi_squared_0_red = self.chi_squared_0 / self.N_d
        
        return self.ln_L

    def _get_flux_scaling(self, d, M, Cov):
        """
        Calculates the optimal flux scaling.

        Args:
            d (array): Data array.
            M (array): Model array.
            Cov (object): Covariance object.

        Returns:
            tuple: Scaled flux and phi.
        """
        # Left- and right-hand sides
        lhs = np.dot(M.T, Cov.solve(M))
        rhs = np.dot(M.T, Cov.solve(d))

        if M.ndim == 1:
            # Only a planet component
            phi = rhs / lhs
            return np.dot(M, phi), phi
        
        raise NotImplementedError
    
        # Multiple model components
        phi = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return np.dot(M, phi), phi

    def _get_err_scaling(self, chi_squared, N_d):
        """
        Calculates the optimal error scaling.

        Args:
            chi_squared (float): Chi-squared value.
            N_d (int): Number of data points.

        Returns:
            float: Scaled error.
        """
        s_squared = np.sqrt(1/N_d * chi_squared)
        return s_squared