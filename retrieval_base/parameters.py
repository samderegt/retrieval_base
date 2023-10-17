import numpy as np
from scipy.stats import invgamma, norm

from petitRADTRANS.retrieval import cloud_cond as fc

from .chemistry import Chemistry

class Parameters:

    # Dictionary of all possible parameters and their default values
    all_params = {

        # Uncertainty scaling
        'a': 0, 'l': 1, 
        'a_f': 0, 'l_f': 1, 
        'beta': 1,  

        # Cloud properties (cloud base from condensation)
        'f_sed': None, 
        'log_K_zz': None, 
        'sigma_g': None, 
    }

    def __init__(
            self, 
            free_params, 
            constant_params, 
            PT_mode='free', 
            n_T_knots=None, 
            enforce_PT_corr=False, 
            chem_mode='free', 
            cloud_mode='gray', 
            cov_mode=None, 
            wlen_settings={
                'J1226': [9,3], 'K2166': [7,3], 
                }
            ):

        # Separate the prior range from the mathtext label
        self.param_priors, self.param_mathtext = {}, {}
        for key_i, (prior_i, mathtext_i) in free_params.items():
            self.param_priors[key_i]   = prior_i
            self.param_mathtext[key_i] = mathtext_i

        self.param_keys = np.array(list(self.param_priors.keys()))
        self.n_params = len(self.param_keys)

        # Create dictionary with constant parameter-values
        self.params = self.all_params.copy()
        self.params.update(constant_params)

        for key_i in list(self.params.keys()):
            if key_i.startswith('log_'):
                self.params = self.log_to_linear(self.params, key_i)

        # Check the used PT profile
        self.PT_mode = PT_mode
        assert(self.PT_mode in ['free', 'free_gradient', 'grid', 'Molliere'])

        self.n_T_knots = n_T_knots
        self.enforce_PT_corr = enforce_PT_corr

        # Check the used chemistry type
        self.chem_mode = chem_mode
        assert(self.chem_mode in ['eqchem', 'free'])
        
        # Check the used cloud type
        self.cloud_mode = cloud_mode
        if cloud_mode == 'grey':
            self.cloud_mode = 'gray'
        assert(self.cloud_mode in ['gray', 'MgSiO3', None])

        # Check the used covariance definition
        self.cov_mode = cov_mode
        assert(self.cov_mode in ['GP', None])

        self.wlen_settings = wlen_settings

    def __call__(self, cube, ndim=None, nparams=None):
        '''
        Update values in the params dictionary.

        Input
        -----
        cube : np.ndarray
            Array of values between 0-1 for the free parameters.
        ndim : int or None
            Number of dimensions. (Equal to number of free parameters)
        nparams : int or None
            Number of free parameters.
        '''

        # Convert to numpy array if necessary
        if (ndim is None) and (nparams is None):
            self.cube_copy = cube
        else:
            self.cube_copy = np.array(cube[:ndim])

        # Loop over all parameters
        for i, key_i in enumerate(self.param_keys):

            if key_i.startswith('invgamma_'):
                # Get the two parameters defining the inverse gamma pdf
                invgamma_a, invgamma_b = self.param_priors[key_i]
                
                # Sample from the inverse gamma prior
                cube[i] = invgamma.ppf(cube[i], a=invgamma_a, loc=0, scale=invgamma_b)

            elif key_i.startswith('gaussian_'):
                # Get the two parameters defining the Gaussian pdf
                mu, sigma = self.param_priors[key_i]
                
                # Sample from the Gaussian prior
                cube[i] = norm.ppf(cube[i], loc=mu, scale=sigma)
            
            else:
                # Sample within the boundaries
                low, high = self.param_priors[key_i]
                cube[i] = low + (high-low)*cube[i]

            self.params[key_i] = cube[i]

            if key_i.startswith('log_'):
                self.params = self.log_to_linear(self.params, key_i)

            if key_i.startswith('invgamma_'):
                self.params[key_i.replace('invgamma_', '')] = self.params[key_i]

            if key_i.startswith('gaussian_'):
                self.params[key_i.replace('gaussian_', '')] = self.params[key_i]

        # Read the parameters for the model's segments
        cube = self.read_PT_params(cube)
        self.read_uncertainty_params()
        self.read_chemistry_params()
        self.read_cloud_params()

        if (ndim is None) and (nparams is None):
            return cube
        else:
            return
    
    def read_PT_params(self, cube=None):

        if self.params.get('temperature') is not None:
            return cube

        if self.PT_mode == 'grid':
            return cube

        if (self.PT_mode == 'Molliere') and (cube is not None):

            # Update the parameters in the MultiNest cube 
            # for the Molliere et al. (2020) PT parameterization

            # T_0 is connection temperature at P=0.1 bar
            T_0 = (3/4 * self.params['T_int']**4 * (0.1 + 2/3))**(1/4)

            # Define the prior based on the other knots
            low, high = self.param_priors['T_1']
            idx = np.argwhere(self.param_keys=='T_1').flatten()[0]
            self.params['T_1'] = T_0 * (high - (high-low)*self.cube_copy[idx])
            cube[idx] = self.params['T_1']

            low, high = self.param_priors['T_2']
            idx = np.argwhere(self.param_keys=='T_2').flatten()[0]
            self.params['T_2'] = self.params['T_1'] * (high - (high-low)*self.cube_copy[idx])
            cube[idx] = self.params['T_2']

            low, high = self.param_priors['T_3']
            idx = np.argwhere(self.param_keys=='T_3').flatten()[0]
            self.params['T_3'] = self.params['T_2'] * (high - (high-low)*self.cube[idx])
            cube[idx] = self.params['T_3']

        if self.PT_mode in ['free', 'free_gradient', 'Molliere']:

            # Fill the pressure knots
            self.params['log_P_knots'] = np.array(self.params['log_P_knots'])
            for i in range(self.n_T_knots-1):
                
                if f'd_log_P_{i}{i+1}' in list(self.params.keys()):
                    # Add the difference in log P to the previous knot
                    self.params['log_P_knots'][-(i+1)-1] = \
                        self.params['log_P_knots'][::-1][i] - \
                        self.params[f'd_log_P_{i}{i+1}']
            
            self.params['P_knots']     = 10**self.params['log_P_knots']
            self.params['ln_P_knots']  = np.log(self.params['P_knots'])

        if self.PT_mode in ['free', 'Molliere']:
            
            # Combine the upper temperature knots into an array
            self.params['T_knots'] = []
            for i in range(self.n_T_knots-1):

                T_i = self.params[f'T_{i+1}']

                if (cube is not None) and self.enforce_PT_corr:
                    # Temperature knot is product of previous knots
                    T_i = np.prod([self.params[f'T_{j}'] for j in range(i+2)])
                    
                    idx = np.argwhere(self.param_keys==f'T_{i+1}').flatten()[0]
                    cube[idx] = T_i

                self.params['T_knots'].append(T_i)

            self.params['T_knots'] = np.array(self.params['T_knots'])[::-1]
            
            if 'invgamma_gamma' in self.param_keys:
                self.params['gamma'] = self.params['invgamma_gamma']

        if (self.PT_mode == 'free_gradient'):

            # Combine the upper temperature knots into an array            
            self.params['T_knots'] = [self.params['T_0'], ]
            self.params['dlnT_dlnP_knots'] = np.array([
                self.params[f'dlnT_dlnP_{i}'] for i in range(self.n_T_knots)
                ])
            for i in range(self.n_T_knots-1):

                ln_P_i1 = self.params['ln_P_knots'][::-1][i+1]
                ln_P_i  = self.params['ln_P_knots'][::-1][i]

                T_i1 = np.exp(
                    np.log(self.params['T_knots'][-1]) + \
                    (ln_P_i1 - ln_P_i) * self.params[f'dlnT_dlnP_{i}']
                    )
                self.params['T_knots'].append(T_i1)

            self.params['T_knots'] = np.array(self.params['T_knots'])[::-1]
            self.params['dlnT_dlnP_knots'] = self.params['dlnT_dlnP_knots'][::-1]
            
        return cube

    def read_uncertainty_params(self):
        
        cov_keys = ['beta', 'a', 'l', 'a_f', 'l_f']

        for w_set, (n_orders, n_dets) in self.wlen_settings.items():
            
            for key_i in cov_keys:

                # Make specific for each wlen setting
                if f'{key_i}_{w_set}' not in self.param_keys:
                    self.params[f'{key_i}_{w_set}'] = self.params[f'{key_i}']

                # Reshape to values for each order and detector
                self.params[f'{key_i}_{w_set}'] = \
                    np.ones((n_orders, n_dets)) * self.params[f'{key_i}_{w_set}']
                
                # Loop over the orders
                for i in range(n_orders):

                    # Replace the constant with a free parameter
                    if f'{key_i}_{i+1}' in self.param_keys:
                        self.params[f'{key_i}_{w_set}'][i,:] = \
                            self.params[f'{key_i}_{i+1}']
                        
                    if f'{key_i}_{w_set}_{i+1}' in self.param_keys:
                        self.params[f'{key_i}_{w_set}'][i,:] = \
                            self.params[f'{key_i}_{w_set}_{i+1}']

        '''
        for key_i in cov_keys:

            if self.params.get(key_i) is None:


        # Reshape to values for each order and detector
        self.params['a']    = np.ones((self.n_orders, self.n_dets)) * self.params['a']
        self.params['l']    = np.ones((self.n_orders, self.n_dets)) * self.params['l']
        self.params['a_f']  = np.ones((self.n_orders, self.n_dets)) * self.params['a_f']
        self.params['l_f']  = np.ones((self.n_orders, self.n_dets)) * self.params['l_f']
        self.params['beta'] = np.ones((self.n_orders, self.n_dets)) * self.params['beta']

        # Make a copy of the global values
        a, l     = np.copy(self.params['a']), np.copy(self.params['l'])
        a_f, l_f = np.copy(self.params['a_f']), np.copy(self.params['l_f'])
        beta = np.copy(self.params['beta'])
        
        for i in range(self.n_orders):
            for j in range(self.n_dets):

                # Replace the constants with the free parameters
                if f'a_{i+1}' in self.param_keys:
                    a[i,:] = self.params[f'a_{i+1}']
                    self.params['a'] = a
                if f'l_{i+1}' in self.param_keys:
                    l[i,:] = self.params[f'l_{i+1}']
                    self.params['l'] = l

                if f'a_f_{i+1}' in self.param_keys:
                    a_f[i,:] = self.params[f'a_f_{i+1}']
                    self.params['a_f'] = a_f
                if f'l_f_{i+1}' in self.param_keys:
                    l_f[i,:] = self.params[f'l_f_{i+1}']
                    self.params['l_f'] = l_f

                if f'beta_{i+1}' in self.param_keys:
                    beta[i,:] = self.params[f'beta_{i+1}']
                    self.params['beta'] = beta
        '''

    def read_chemistry_params(self):

        if self.chem_mode == 'eqchem':
            # Use chemical equilibrium
            self.VMR_species = None

        elif self.chem_mode == 'free':
            # Use free chemistry
            self.params['C/O'], self.params['Fe/H'] = None, None

            # Loop over all possible species
            self.VMR_species = {}
            for species_i in Chemistry.species_info.keys():

                # If multiple VMRs are given
                for j in range(3):
                    if f'log_{species_i}_{j}' in self.param_keys:
                        self.VMR_species[f'{species_i}_{j}'] = self.params[f'{species_i}_{j}']

                if f'log_{species_i}' in self.param_keys:
                    self.VMR_species[f'{species_i}'] = self.params[f'{species_i}']
                    continue

                if species_i == '13CO' and ('log_13C/12C_ratio' in self.param_keys):
                    # Use isotope ratio to retrieve the VMR
                    self.VMR_species[species_i] = self.params['13C/12C_ratio'] * self.params['12CO']

                if species_i == '13CH4' and ('log_13C/12C_ratio' in self.param_keys):
                    # Use isotope ratio to retrieve the VMR
                    self.VMR_species[species_i] = self.params['13C/12C_ratio'] * self.params['CH4']

                if species_i == 'C18O' and ('log_18O/16O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['18O/16O_ratio'] * self.params['12CO']
                if species_i == 'C17O' and ('log_17O/16O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['17O/16O_ratio'] * self.params['12CO']

                if species_i == 'H2O_181' and ('log_18O/16O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['18O/16O_ratio'] * self.params['H2O']
                if species_i == 'H2O_171' and ('log_17O/16O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['17O/16O_ratio'] * self.params['H2O']
        
    def read_cloud_params(self, pressure=None, temperature=None):

        if (self.cloud_mode == 'MgSiO3') and (self.chem_mode == 'eqchem'):
            # Return the eq.-chem. mass fraction of MgSiO3
            X_eq_MgSiO3 = fc.return_XMgSiO3(self.params['Fe/H'], self.params['C/O'])
            # Pressure at the cloud base
            # TODO: this doesn't work, temperature is not yet given
            self.params['P_base_MgSiO3'] = fc.simple_cdf_MgSiO3(pressure, temperature, 
                                                                self.params['Fe/H'], 
                                                                self.params['C/O']
                                                                )

            # Log mass fraction at the cloud base
            self.params['log_X_cloud_base_MgSiO3'] = np.log10(self.params['X_MgSiO3'] * X_eq_MgSiO3)
            self.params['X_cloud_base_MgSiO3'] = 10**self.params['log_X_cloud_base_MgSiO3']

    @classmethod
    def log_to_linear(cls, param_dict, key_log, key_lin=None, verbose=False):

        if not isinstance(key_log, (list, tuple, np.ndarray)):
            key_log = [key_log]
            
        if not isinstance(key_lin, (list, tuple, np.ndarray)):
            key_lin = [key_lin]
        
        for key_log_i, key_lin_i in zip(key_log, key_lin):

            if key_lin_i is None:
                key_lin_i = key_log_i.replace('log_', '')

            # Value of the logarithmic parameter
            val_log = param_dict[key_log_i]

            if isinstance(val_log, (float, int, np.ndarray)):
                # Convert from log to linear if float or integer
                param_dict[key_lin_i] = 10**val_log
            elif isinstance(val_log, list):
                param_dict[key_lin_i] = 10**np.array(val_log)

            elif val_log is None:
                # Set linear parameter to None as well
                param_dict[key_lin_i] = None

        return param_dict