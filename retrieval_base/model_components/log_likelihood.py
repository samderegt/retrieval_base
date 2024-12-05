import numpy as np
from scipy.special import loggamma

def get_class(**kwargs):
    return LogLikelihood(**kwargs)

class LogLikelihood:
    def __init__(self, d_spec, sum_model_settings=True, scale_flux=False, scale_err=False, alpha=2, N_phi=1, scale_relative_to_chip=-1, **kwargs):

        # Combine all model settings
        self.combine_model_settings(d_spec, sum_model_settings)

        self.scale_flux = scale_flux
        self.scale_err  = scale_err

        self.alpha = alpha
        self.N_phi = N_phi
        self.scale_relative_to_chip = scale_relative_to_chip

    def combine_model_settings(self, d_spec, sum_model_settings=False):
        
        # All model settings
        self.model_settings = list(d_spec.keys())
        self.sum_model_settings = sum_model_settings

        self.d_flux = []
        for m_set in self.model_settings:
            for d_flux_i in d_spec[m_set].flux:
                # Add each chip
                self.d_flux.append(d_flux_i)

            if self.sum_model_settings:
                # Only compare to one model setting
                break
        
        self.n_chips = len(self.d_flux)

    def get_flux_scaling(self, d, M, Cov):
        
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

    def get_err_scaling(self, chi_squared, N_d):
        s_squared = np.sqrt(1/N_d * chi_squared)
        return s_squared
    
    def __call__(self, m_flux, Cov, **kwargs):

        self.ln_L = 0.
        self.chi_squared_0 = 0.

        self.m_flux_phi = [np.nan] * self.n_chips
        self.phi        = np.ones(self.n_chips)
        self.s_squared  = np.ones(self.n_chips)

        self.N_d = 0

        # Loop over all chips
        for i, (d_flux_i, m_flux_i, Cov_i) in enumerate(zip(self.d_flux, m_flux, Cov)):

            apply_scaling = self.scale_flux & (i != self.scale_relative_to_chip)
            
            # Number of data points
            mask = np.isfinite(d_flux_i)
            N_d  = mask.sum()
            self.N_d += N_d
            if N_d == 0:
                continue

            self.m_flux_phi[i] = m_flux_i
            if apply_scaling:
                # Find optimal flux-scaling to match observations
                self.m_flux_phi[i][mask], self.phi[i] = \
                    self.get_flux_scaling(d_flux_i[mask], m_flux_i[mask], Cov_i)
                
            # Residuals compared to scaled model
            residuals = d_flux_i - self.m_flux_phi[i]

            # Chi-squared for optimal linear scaling
            chi_squared_0 = np.dot(residuals[mask].T, Cov_i.solve(residuals[mask]))
            self.chi_squared_0 += chi_squared_0

            if apply_scaling:
                # Covariance matrix of phi
                MT_inv_cov_0_M = np.dot(m_flux_i[mask].T, Cov_i.solve(m_flux_i[mask]))

                # Take the (log)-determinant of the phi-covariance matrix
                # This introduces uncertainty on phi into log-likelihood
                logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M) # 1x1 matrix (i.e. scalar)

            if self.scale_err:
                # Scale the variance to maximise log-likelihood
                self.s_squared[i] = self.get_err_scaling(chi_squared_0, N_d)

            # Get the log of the determinant
            logdet_cov_0 = Cov_i.logdet

            # Constant term for this order/detector
            self.ln_L += -1/2*(N_d-self.N_phi) * np.log(2*np.pi) + \
                loggamma(1/2*(N_d-self.N_phi+self.alpha-1))

            # Add this order/detector to the total log-likelihood
            self.ln_L += -1/2*(
                logdet_cov_0 + logdet_MT_inv_cov_0_M + \
                (N_d-self.N_phi+self.alpha-1) * np.log(chi_squared_0)
            )

        # Reduced chi-squared without variance scaling
        self.chi_squared_0_red = self.chi_squared_0 / self.N_d
        
        return self.ln_L