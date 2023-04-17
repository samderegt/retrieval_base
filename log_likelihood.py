import numpy as np

from covariance import Covariance, GaussianProcesses

class LogLikelihood:

    def __init__(self, d_spec, scale_flux=False, scale_err=False, scale_GP_amp=False):

        self.d_spec = d_spec

        self.scale_flux   = scale_flux
        self.scale_err    = scale_err
        self.scale_GP_amp = scale_GP_amp

    def __call__(self, m_spec, params, ln_L_penalty=0):
        '''
        Evaluate the total log-likelihood given the model spectrum and parameters.

        Input
        -----
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class.
        params : dict
            Dictionary containing free/constant parameters. 
        ln_L_penalty : float
            Penalty term to be added to the total log-likelihood. Default is 0.       
        '''
        
        self.params = params
        #self.m_spec = m_spec

        # Set up the total log-likelihood for this model 
        # (= 0 if there is no penalty)
        self.ln_L = ln_L_penalty
        
        # Arrays to store log-likelihood and chi-squared per pixel in
        self.ln_L_per_pixel        = ln_L_penalty * np.ones_like(self.d_spec.flux)
        self.chi_squared_per_pixel = np.nan * np.ones_like(self.d_spec.flux)

        # Array to store the linear flux-scaling terms
        self.f    = np.ones((self.d_spec.n_orders, self.d_spec.n_dets))
        # Array to store the uncertainty-scaling terms
        self.beta = np.ones((self.d_spec.n_orders, self.d_spec.n_dets))
        
        # Loop over all orders and detectors
        for i in range(self.d_spec.n_orders):
            for j in range(self.d_spec.n_dets):

                # Apply mask to model and data, calculate residuals
                mask_ij = self.d_spec.mask_isfinite[i,j,:]

                # Number of data points
                N_ij = mask_ij.sum()
                if N_ij == 0:
                    continue

                m_flux_ij = m_spec.flux[i,j,:][mask_ij]
                d_flux_ij = self.d_spec.flux[i,j,:][mask_ij]
                d_err_ij  = self.d_spec.err[i,j,:][mask_ij]

                res_ij = (d_flux_ij - m_flux_ij)

                # Set up the covariance matrix
                if self.params['a'][i,j] != 0:

                    # Wavelengths within this order/detector
                    d_wave_ij = self.d_spec.wave[i,j,:][mask_ij]

                    # Wavelength separation between pixels
                    d_delta_wave_ij = np.abs(d_wave_ij[None,:] - d_wave_ij[:,None])

                    # Use Gaussian Processes
                    cov_ij = GaussianProcesses(d_err_ij)

                    if self.scale_GP_amp:
                        # Scale the GP amplitude by flux uncertainty
                        GP_err = d_err_ij
                    else:
                        GP_err = None

                    # Add a radial-basis function kernel
                    cov_ij.add_RBF_kernel(a=self.params['a'][i,j], 
                                          l=self.params['l'][i,j], 
                                          delta_wave=d_delta_wave_ij, 
                                          err=GP_err, 
                                          trunc_dist=5
                                          )

                else:
                    # Use only diagonal terms in covariance matrix
                    cov_ij = Covariance(d_err_ij)

                if self.params['beta'][i,j] != 1:
                    # Scale the flux uncertainty
                    cov_ij.add_data_err_scaling(beta=self.params['beta'][i,j])

                if self.params['beta_tell'] is not None:
                    # Add a scaling separate from the tellurics
                    # TODO: confusing wording...
                    d_transm_ij = self.d_spec.transm[i,j,:][mask_ij]
                    cov_ij.add_model_err(model_err=params['beta_tell']*d_err_ij/d_transm_ij)

                if self.params['x_tol'] is not None:
                    # Add a model uncertainty (Piette et al. 2020)
                    cov_ij.add_model_err(self.params['x_tol']*m_flux_ij)

                if self.params['b'] is not None:
                    # Add a model uncertainty (Line et al. 2015)
                    cov_ij.add_model_err(np.sqrt(10**self.params['b']))

                # Get the log of the determinant (log prevents over/under-flow)
                cov_ij.get_logdet()

                # Set up the log-likelihood for this order/detector
                # Chi-squared and optimal uncertainty scaling terms still need to be added
                ln_L_ij = -(N_ij/2*np.log(2*np.pi) + 1/2*cov_ij.logdet)

                if self.scale_flux and not (i==0 and j==0):
                    # Only scale the flux relative to the first order/detector

                    # Scale the model flux to minimize the chi-squared error
                    m_flux_ij_scaled, f_ij = self.get_flux_scaling(d_flux_ij, m_flux_ij, cov_ij)
                    res_ij_scaled = (d_flux_ij - m_flux_ij_scaled)

                    # Chi-squared for the optimal linear scaling
                    chi_squared_ij_scaled = np.dot(res_ij_scaled, cov_ij.solve(res_ij_scaled))
                else:
                    # Chi-squared without linear scaling of detectors
                    f_ij = 1
                    chi_squared_ij_scaled = np.dot(res_ij, cov_ij.solve(res_ij))
                
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

                # Add to the total log-likelihood
                self.ln_L += ln_L_ij

                # Store in the arrays
                self.f[i,j]    = f_ij
                self.beta[i,j] = beta_ij

                # This is not perfect for off-diagonal elements in covariance matrix
                if cov_ij.is_matrix:
                    self.chi_squared_per_pixel[i,j,mask_ij] = 1/beta_ij**2 * res_ij**2/cov_ij.cov.diagonal()
                else:
                    self.chi_squared_per_pixel[i,j,mask_ij] = 1/beta_ij**2 * res_ij**2/cov_ij.cov
                self.ln_L_per_pixel[i,j,mask_ij] = -(
                    N_ij/2*np.log(2*np.pi) + \
                    N_ij/2*np.log(beta_ij**2) + \
                    1/2*self.chi_squared_per_pixel[i,j,mask_ij]
                    )

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