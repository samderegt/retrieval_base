import numpy as np

class LogLikelihood:

    def __init__(self, 
                 d_spec, 
                 n_params, 
                 scale_flux=False, 
                 scale_err=False, 
                 ):

        # Observed spectrum is constant
        self.d_spec   = d_spec
        self.n_params = n_params

        # Number of degrees of freedom
        self.n_dof = self.d_spec.mask_isfinite.sum() - self.n_params

        self.scale_flux   = scale_flux
        self.scale_err    = scale_err
        
    def __call__(self, m_spec, Cov, ln_L_penalty=0):
        '''
        Evaluate the total log-likelihood given the model spectrum and parameters.

        Input
        -----
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class.
        Cov : Covariance class
            Instance of the GaussianProcesses or Covariance class. 
        ln_L_penalty : float
            Penalty term to be added to the total log-likelihood. Default is 0.       
        '''
        
        #self.m_spec = m_spec

        # Set up the total log-likelihood for this model 
        # (= 0 if there is no penalty)
        self.ln_L = ln_L_penalty
        self.chi_squared = 0
        
        # Arrays to store log-likelihood and chi-squared per pixel in
        #self.ln_L_per_pixel        = ln_L_penalty * np.ones_like(self.d_spec.flux)
        self.ln_L_per_pixel        = np.nan * np.ones_like(self.d_spec.flux)
        self.chi_squared_per_pixel = np.nan * np.ones_like(self.d_spec.flux)

        # Array to store the linear flux-scaling terms
        self.f    = np.ones((self.d_spec.n_orders, self.d_spec.n_dets))
        # Array to store the uncertainty-scaling terms
        self.beta = np.ones((self.d_spec.n_orders, self.d_spec.n_dets))
        
        N_tot = self.d_spec.mask_isfinite.sum()

        # Loop over all orders and detectors
        for i in range(self.d_spec.n_orders):
            for j in range(self.d_spec.n_dets):

                # Apply mask to model and data, calculate residuals
                mask_ij = self.d_spec.mask_isfinite[i,j,:]

                # Number of data points
                N_ij = mask_ij.sum()
                if N_ij == 0:
                    continue

                m_flux_ij = m_spec.flux[i,j,mask_ij]
                d_flux_ij = self.d_spec.flux[i,j,mask_ij]
                d_err_ij  = self.d_spec.err[i,j,mask_ij]

                res_ij = (d_flux_ij - m_flux_ij)
                
                if Cov[i,j].is_matrix:
                    # Retrieve a Cholesky decomposition
                    Cov[i,j].get_cholesky()

                # Get the log of the determinant (log prevents over/under-flow)
                Cov[i,j].get_logdet()

                # Set up the log-likelihood for this order/detector
                # Chi-squared and optimal uncertainty scaling terms still need to be added
                ln_L_ij = -(N_ij/2*np.log(2*np.pi) + 1/2*Cov[i,j].logdet)

                if self.scale_flux and not (i==0 and j==0):
                    # Only scale the flux relative to the first order/detector

                    # Scale the model flux to minimize the chi-squared error
                    m_flux_ij_scaled, f_ij = self.get_flux_scaling(d_flux_ij, m_flux_ij, Cov[i,j])
                    res_ij = (d_flux_ij - m_flux_ij_scaled)

                else:
                    # Without linear scaling of detectors
                    f_ij = 1

                # Chi-squared for the optimal linear scaling
                inv_cov_ij_res_ij = Cov[i,j].solve(res_ij)
                chi_squared_ij_scaled = np.dot(res_ij, inv_cov_ij_res_ij)
                
                if self.scale_err:
                    # Scale the flux uncertainty that maximizes the log-likelihood
                    beta_ij = self.get_err_scaling(chi_squared_ij_scaled, N_ij)
                else:
                    # No additional uncertainty scaling
                    beta_ij = 1

                # Chi-squared for optimal linear scaling and uncertainty scaling
                chi_squared_ij = 1/beta_ij**2 * chi_squared_ij_scaled

                # Add chi-squared and optimal uncertainty scaling terms to log-likelihood
                ln_L_ij += -(N_ij/2*np.log(beta_ij**2) + 1/2*chi_squared_ij)

                # Add to the total log-likelihood and chi-squared
                self.ln_L += ln_L_ij
                #self.chi_squared += chi_squared_ij
                self.chi_squared += np.nansum((res_ij/d_err_ij)**2)

                # Store in the arrays
                self.f[i,j]    = f_ij
                self.beta[i,j] = beta_ij

                # Following Peter McGill's advice
                g_k = 1/beta_ij**2 * inv_cov_ij_res_ij
                sigma_bar_kk = np.diag(
                    1/beta_ij**2 * Cov[i,j].solve(np.eye(N_ij))
                    )

                # Conditional mean and standard deviation
                mu_tilde_k = d_flux_ij - g_k/sigma_bar_kk
                sigma_tilde_k = 1/sigma_bar_kk

                # Scale the ln L penalty by the number of good pixels
                self.ln_L_per_pixel[i,j,mask_ij] = ln_L_penalty/N_tot - (
                    1/2*np.log(2*np.pi*sigma_tilde_k) + \
                    1/2*(d_flux_ij - mu_tilde_k)**2/sigma_tilde_k
                    )

                self.chi_squared_per_pixel[i,j,mask_ij] = \
                    (d_flux_ij - mu_tilde_k)**2/sigma_tilde_k

        # Reduced chi-squared
        self.chi_squared_red = self.chi_squared / self.n_dof

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
        m_flux_ij*f_ij : np.ndarray
            Scaled model flux.
        f_ij : 
            Optimal linear scaling factor.
        '''
        
        # Left-hand side
        lhs = np.dot(m_flux_ij, cov_ij.solve(m_flux_ij))
        # Right-hand side
        rhs = np.dot(m_flux_ij, cov_ij.solve(d_flux_ij))

        # Return the scaled model flux
        f_ij = rhs / lhs
        return m_flux_ij * f_ij, f_ij

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
        beta_ij : float
            Optimal uncertainty scaling factor.
        '''

        # Find uncertainty scaling that maximizes log-likelihood
        beta_ij = np.sqrt(1/N_ij * chi_squared_ij_scaled)
        return beta_ij