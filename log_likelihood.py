import numpy as np

from .covariance import Covariance, GaussianProcesses

class LogLikelihood:

    def __init__(self, d_spec, scale_flux=None, scale_err=None):

        self.d_spec = d_spec

        self.scale_flux = scale_flux
        self.scale_err  = scale_err

    def __call__(self, m_spec, params):
        '''
        Input
        -----
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class.
        params : dict
            Dictionary containing free/constant parameters. 
        
        '''
        
        self.params = params
        #self.m_spec = m_spec
        
        # Loop over all orders and detectors
        for i in range(self.d_spec.n_orders):
            for j in range(self.d_spec.n_dets):

                # Apply mask to model and data, calculate residuals
                mask_ij = self.d_spec.mask_isfinite[i,j,:]

                m_flux_ij = m_spec.flux[i,j,:][mask_ij]
                d_flux_ij = self.d_spec.flux[i,j,:][mask_ij]
                d_err_ij  = self.d_spec.err[i,j,:][mask_ij]
                d_delta_wave_ij = self.d_spec.delta_wave[i,j,:,:][mask_ij,mask_ij]

                res_ij = (d_flux_ij - m_flux_ij)
                
                # Number of data points
                N_ij = len(d_flux_ij)

                # Set up the covariance matrix
                if self.params['a'][i,j] != 0:
                    # Use Gaussian Processes
                    cov_ij = GaussianProcesses(d_err_ij)

                    # Add a radial-basis function kernel
                    cov_ij.add_RBF_kernel(a=self.params['a'][i,j], 
                                          l=self.params['tau'][i,j], 
                                          delta_wave=d_delta_wave_ij, 
                                          trunc_dist=3
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


                if self.scale_flux:
                    # Scale the model flux to minimize the chi-squared error
                    self.get_flux_scaling(d_flux_ij, m_flux_ij, cov_ij)
                
                if self.scale_err:
                    # TODO: ...
                    pass
                # ...
                pass


    def get_flux_scaling(self, d_flux_ij, m_flux_ij, cov_ij):
        # TODO: ...
        pass

    def get_err_scaling(self, ):
        # TODO: ...
        pass