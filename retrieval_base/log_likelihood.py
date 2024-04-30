import numpy as np
from scipy.special import loggamma

class LogLikelihood:

    def __init__(self, 
                 d_spec, 
                 n_params, 
                 scale_flux=False, 
                 scale_err=False, 
                 alpha=2, 
                 N_phi=1, 
                 ):

        # Observed spectrum is constant
        self.d_spec = d_spec
        self.d_flux = self.d_spec.flux
        self.d_mask = self.d_spec.mask_isfinite

        self.n_orders = self.d_spec.n_orders
        self.n_dets   = self.d_spec.n_dets

        self.scale_flux   = scale_flux
        self.scale_err    = scale_err
        
        # Number of degrees of freedom
        self.N_d      = self.d_mask.sum()
        self.N_params = n_params
        
        self.alpha = alpha

        # Number of linear scaling parameters
        self.N_phi = N_phi
        
    def __call__(self, M, Cov, **kwargs):

        self.ln_L   = 0
        self.chi2_0 = 0

        # Array to store the linear flux-scaling terms
        self.phi = np.ones((self.n_orders, self.n_dets, self.N_phi))
        # Uncertainty-scaling
        self.s2  = np.ones((self.n_orders, self.n_dets))

        self.m_flux_phi = np.nan * np.ones_like(self.d_flux)

        # Loop over all orders and detectors
        for i in range(self.d_spec.n_orders):
            for j in range(self.d_spec.n_dets):

                # Apply mask to model and data, calculate residuals
                mask_ij = self.d_mask[i,j,:]

                # Number of data points
                N_d = mask_ij.sum()
                if N_d == 0:
                    continue

                # Avoid repeated indexing/masking of data and model
                d_flux_ij = self.d_flux[i,j,mask_ij]
                M_ij = M[i,j,:,mask_ij]

                if Cov[i,j].is_matrix:
                    # Retrieve a Cholesky decomposition
                    Cov[i,j].get_cholesky()
                    
                if self.scale_flux:
                    # Find the optimal phi-vector to match the observed spectrum
                    self.m_flux_phi[i,j,mask_ij], self.phi[i,j,:] = \
                        self.get_flux_scaling(d_flux_ij, M_ij, Cov[i,j])
                else:
                    # Use first element (planet) in model-matrix
                    self.m_flux_phi[i,j] = M[i,j,0]

                # Residuals wrt scaled model
                residuals_phi = (self.d_flux[i,j] - self.m_flux_phi[i,j])

                # Chi-squared for the optimal linear scaling
                inv_cov_0_residuals_phi = Cov[i,j].solve(residuals_phi[mask_ij])
                chi2_0 = np.dot(residuals_phi[mask_ij].T, inv_cov_0_residuals_phi)

                logdet_MT_inv_cov_0_M = 0
                if self.scale_flux:
                    # Covariance matrix of phi
                    inv_cov_0_M    = Cov[i,j].solve(M_ij)
                    MT_inv_cov_0_M = np.dot(M_ij.T, inv_cov_0_M)

                    # Take the (log)-determinant of the phi-covariance matrix
                    # This introduces uncertainty on phi into log-likelihood
                    if not isinstance(MT_inv_cov_0_M, np.ndarray):
                        # 1x1 matrix (i.e. scalar)
                        logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M)
                    else:
                        # TODO: not sure how stable this slogdet is...
                        logdet_MT_inv_cov_0_M = np.linalg.slogdet(MT_inv_cov_0_M)[1]

                if self.scale_err:
                    # Scale the variance to maximise log-likelihood
                    self.s2[i,j] = self.get_err_scaling(chi2_0, N_d)
                    
                # Get the log of the determinant (log prevents over/under-flow)
                logdet_cov_0 = Cov[i,j].get_logdet()

                # Constant term for this order/detector
                self.ln_L += -1/2*(N_d-self.N_phi) * np.log(2*np.pi) + \
                    loggamma(1/2*(N_d-self.N_phi+self.alpha-1))

                # Add this order/detector to the total log-likelihood
                self.ln_L += -1/2*(
                    logdet_cov_0 + logdet_MT_inv_cov_0_M + \
                    (N_d-self.N_phi+self.alpha-1) * np.log(chi2_0)
                )

                self.chi2_0 += chi2_0

        # Reduced chi-squared
        self.chi2_0_red = self.chi2_0 / self.N_d

        return self.ln_L

    def get_flux_scaling(self, d_flux_ij, m_flux_ij, cov_ij):
        '''
        Following Ruffio et al. (2019). Find the optimal linear 
        scaling parameter to minimize the chi-squared error. 

        Solve for the linear scaling parameter f in:
        (M^T * cov^-1 * M) * f = M^T * cov^-1 * d

        Input
        -----
        d_flux_ij : np.ndarray
            Flux of the observed spectrum.
        m_flux_ij : np.ndarray
            Flux of the model spectrum.
        cov_ij : Covariance class
            Instance of the Covariance class. Should have a 
            solve() method to avoid matrix-inversion.

        Returns
        -------
        m_flux_ij*phi_ij : np.ndarray
            Scaled model flux.
        phi_ij : 
            Optimal linear scaling factor.
        '''

        if (m_flux_ij.ndim==2) and (m_flux_ij.shape[-1]==1):
            m_flux_ij = m_flux_ij[:,0]
                
        # Left-hand side
        lhs = np.dot(m_flux_ij.T, cov_ij.solve(m_flux_ij))
        # Right-hand side
        rhs = np.dot(m_flux_ij.T, cov_ij.solve(d_flux_ij))

        # Return the scaled model flux
        if (m_flux_ij.ndim==2):
            # TODO: Apply prior on linear scaling factors
            # from scipy.optimize import lsq_linear
            phi_ij = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        else:
            phi_ij = rhs / lhs
        
        return np.dot(m_flux_ij, phi_ij), phi_ij

    def get_err_scaling(self, chi_squared_ij_scaled, N_ij):
        '''
        Following Ruffio et al. (2019). Find the optimal uncertainty
        scaling parameter to maximize the log-likelihood. 

        Input
        -----
        chi_squared_ij_scaled : float
            Chi-squared error of the optimally-scaled model spectrum.
        N_ij : int
            Number of datapoints/pixels in spectrum.

        Returns
        -------
        s2_ij : float
            Optimal uncertainty scaling factor.
        '''

        # Find uncertainty scaling that maximizes log-likelihood
        s2_ij = np.sqrt(1/N_ij * chi_squared_ij_scaled)
        return s2_ij