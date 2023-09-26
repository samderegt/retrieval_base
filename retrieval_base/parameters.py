import numpy as np
from scipy.stats import invgamma

from petitRADTRANS.retrieval import cloud_cond as fc

from .spectrum import Spectrum
from .chemistry import Chemistry

class Parameters:

    # Dictionary of all possible parameters and their default values
    all_params = {

        # Uncertainty scaling
        'log_a': -np.inf, 
        'log_l': 0, 
        'beta': 1,  
        'beta_tell': None, 
        'x_tol': None, 
        'b': None, 
        'ls1': None,
        'ls2': None,


        # General properties
        'R_p': 1.0, 
        'log_g': 5.0, 
        'epsilon_limb': 0.6, 
        'parallax': 250, # mas


        # Velocities
        'vsini': 10.0, 
        'rv': 0.0, 


        # Cloud properties (specify cloud base)
        'log_X_cloud_base_MgSiO3': None, 
        'log_P_base_MgSiO3': None, 

        # Cloud properties (cloud base from condensation)
        'log_X_MgSiO3': None, 
        'f_sed': None, 
        'log_K_zz': None, 
        'sigma_g': None, 

        # Cloud properties (gray cloud)
        'log_opa_base_gray': None, 
        'log_P_base_gray': None, 
        'f_sed_gray': None, 
        'cloud_slope': 0, 


        # Chemistry (chemical equilibrium)
        'C/O': None, 
        'Fe/H': None, 
        'log_P_quench': -8, 

        # Chemistry (free)
        'log_12CO': -np.inf, 
        'log_H2O': -np.inf, 
        'log_CH4': -np.inf, 
        'log_NH3': -np.inf, 
        'log_HCN': -np.inf, 
        'log_CO2': -np.inf, 
        'log_C_ratio': -np.inf, 
        'log_O_ratio': -np.inf, 


        # PT profile (free)
        'log_gamma': None, 

        'T_0': 4000, 
        'T_1': 2000, 
        'T_2': 1800, 
        'T_3': 1500, 
        'T_3': 1400, 
        # etc.
        'log_P_knots': np.array([-6,-2,0,1,2], dtype=np.float64), 

        # PT profile (Molliere et al. 2020)
        'log_P_phot': 0.0, 
        'alpha': 1.0, 
        'T_int': 1800, 

        # PT profile (SONORA grid)
        'T_eff': 1300, 

    }

    def __init__(self, free_params, constant_params, n_orders=7, n_dets=3, enforce_PT_corr=False):

        # Separate the prior range from the mathtext label
        self.param_priors, self.param_mathtext = {}, {}
        for key_i, (prior_i, mathtext_i) in free_params.items():
            self.param_priors[key_i]   = prior_i
            self.param_mathtext[key_i] = mathtext_i

        self.param_keys = np.array(list(self.param_priors.keys()))
        self.n_params = len(self.param_keys)

        # Count the number of temperature knots above the base
        for i in range(1,100):
            if (f'T_{i}' not in self.param_keys) and (f'dlnT_dlnP_{i-1}' not in self.param_keys):
                self.n_T_knots = i-1
                break

        # Create dictionary with constant parameter-values
        self.params = self.all_params.copy()
        self.params.update(constant_params)


        # Check if Molliere et al. (2020) PT-profile is used
        Molliere_param_keys = ['log_P_phot', 'alpha', 'T_int', 'T_1', 'T_2', 'T_3']
        PT_grid_param_keys = ['log_g', 'T_eff']
        free_PT_gradient_param_keys = ['dlnT_dlnP_0']

        if np.isin(Molliere_param_keys, self.param_keys).all():
            self.PT_mode = 'Molliere'
        elif np.isin(PT_grid_param_keys, self.param_keys).all():
            self.PT_mode = 'grid'
        elif np.isin(free_PT_gradient_param_keys, self.param_keys).all():
            self.PT_mode = 'free_gradient'
        else:
            self.PT_mode = 'free'

        # Check if equilibrium- or free-chemistry is used
        eqchem_param_keys = ['C/O', 'Fe/H']
       
        if np.isin(eqchem_param_keys, self.param_keys).all():
            self.chem_mode = 'eqchem'
        else:
            self.chem_mode = 'free'

        # Check if Gaussian processes are used
        GP_param_keys = ['a', 'log_a', 'a_1', 'log_a_1']
        if np.isin(GP_param_keys, self.param_keys).any():
            self.cov_mode = 'GP'
        else:
            self.cov_mode = None

        # Check if clouds (and which type) are included
        self.cloud_mode = None
        MgSiO3_param_keys = ['log_X_MgSiO3', 'f_sed', 'log_K_zz', 'sigma_g']
        gray_cloud_param_keys = ['log_opa_base_gray', 'log_P_base_gray', 'f_sed_gray']
        
        if np.isin(MgSiO3_param_keys, self.param_keys).all():
            self.cloud_mode = 'MgSiO3'
        elif np.isin(gray_cloud_param_keys, self.param_keys).all():
            self.cloud_mode = 'gray'

        self.n_orders = n_orders
        self.n_dets   = n_dets

        self.enforce_PT_corr = enforce_PT_corr

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
            
            else:
                # Sample within the boundaries
                low, high = self.param_priors[key_i]
                cube[i] = low + (high-low)*cube[i]

            self.params[key_i] = cube[i]

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

        if self.PT_mode in ['free', 'Molliere']:

            # Fill the pressure knots
            self.params['P_knots'] = 10**np.array(
                self.params['log_P_knots'], dtype=np.float64
                )[[0,-1]]
            for i in range(self.n_T_knots-1):
                
                if f'd_log_P_{i}{i+1}' in list(self.params.keys()):
                    # Add the difference in log P to the previous knot
                    log_P_i = np.log10(self.params['P_knots'][1]) - \
                        self.params[f'd_log_P_{i}{i+1}']
                else:
                    # Use the stationary knot
                    log_P_i = self.params['log_P_knots'][-i-2]

                # Insert each pressure knot into the array
                self.params['P_knots'] = np.insert(self.params['P_knots'], 1, 10**log_P_i)

            self.params['log_P_knots'] = np.log10(self.params['P_knots'])
            
            # Combine the upper temperature knots into an array
            self.params['T_knots'] = []
            for i in range(self.n_T_knots):

                T_i = self.params[f'T_{i+1}']

                if (cube is not None) and self.enforce_PT_corr:
                    # Temperature knot is product of previous knots
                    T_i = np.prod([self.params[f'T_{j}'] for j in range(i+2)])
                    
                    idx = np.argwhere(self.param_keys==f'T_{i+1}').flatten()[0]
                    cube[idx] = T_i

                self.params['T_knots'].append(T_i)

            self.params['T_knots'] = np.array(self.params['T_knots'])[::-1]

            # Convert from logarithmic to linear scale
            self.params = self.log_to_linear(self.params, 'log_gamma', 'gamma')
            
            if 'invgamma_gamma' in self.param_keys:
                self.params['gamma'] = self.params['invgamma_gamma']

        if (self.PT_mode == 'free_gradient'):

            # Fill the pressure knots
            self.params['P_knots'] = 10**np.array(
                self.params['log_P_knots'], dtype=np.float64
                )[[0,-1]]
            for i in range(self.n_T_knots-2):
                
                if f'd_log_P_{i}{i+1}' in list(self.params.keys()):
                    # Add the difference in log P to the previous knot
                    log_P_i = np.log10(self.params['P_knots'][1]) - \
                        self.params[f'd_log_P_{i}{i+1}']
                else:
                    # Use the stationary knot
                    log_P_i = self.params['log_P_knots'][-i-2]

                # Insert each pressure knot into the array
                self.params['P_knots'] = np.insert(self.params['P_knots'], 1, 10**log_P_i)

            self.params['log_P_knots'] = np.log10(self.params['P_knots'])
            self.params['ln_P_knots']  = np.log(self.params['P_knots'])

            # Combine the upper temperature knots into an array
            T_i = self.params['T_0']
            self.params['ln_P_knots'] = np.log(self.params['P_knots'])
            
            self.params['T_knots'] = [T_i, ]
            self.params['dlnT_dlnP_knots'] = [
                self.params[f'dlnT_dlnP_{i}'] for i in range(self.n_T_knots)
                ]
            for i in range(1,self.n_T_knots):

                T_i = np.exp(
                    np.log(T_i) + self.params[f'dlnT_dlnP_{i-1}'] * \
                    (self.params['ln_P_knots'][-i-1] - self.params['ln_P_knots'][-i]) 
                    )
                self.params['T_knots'].append(T_i)

            self.params['T_knots'] = np.array(self.params['T_knots'])[::-1]
            self.params['dlnT_dlnP_knots'] = np.array(self.params['dlnT_dlnP_knots'])
            
        return cube

    def read_uncertainty_params(self):

        # Convert the logarithmic value to linear scale
        if 'a' not in self.param_keys:
            self.params['a'] = 10**self.params['log_a']
        if 'l' not in self.param_keys:
            self.params['l'] = 10**self.params['log_l']

        if 'a_f' not in self.param_keys:
            self.params['a_f'] = 10**self.params['log_a_f']
        if 'l_f' not in self.param_keys:
            self.params['l_f'] = 10**self.params['log_l_f']

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
                if f'log_a_{i+1}' in self.param_keys:
                    a[i,:] = 10**self.params[f'log_a_{i+1}']
                    self.params['a'] = a
                if f'a_{i+1}' in self.param_keys:
                    a[i,:] = self.params[f'a_{i+1}']
                    self.params['a'] = a
                if f'log_l_{i+1}' in self.param_keys:
                    l[i,:] = 10**self.params[f'log_l_{i+1}']
                    self.params['l'] = l
                if f'l_{i+1}' in self.param_keys:
                    l[i,:] = self.params[f'l_{i+1}']
                    self.params['l'] = l

                if f'log_a_f_{i+1}' in self.param_keys:
                    a_f[i,:] = 10**self.params[f'log_a_f_{i+1}']
                    self.params['a'] = a_f
                if f'a_f_{i+1}' in self.param_keys:
                    a_f[i,:] = self.params[f'a_f_{i+1}']
                    self.params['a_f'] = a_f
                if f'log_l_f_{i+1}' in self.param_keys:
                    l_f[i,:] = 10**self.params[f'log_l_f_{i+1}']
                    self.params['l_f'] = l_f
                if f'l_f_{i+1}' in self.param_keys:
                    l_f[i,:] = self.params[f'l_f_{i+1}']
                    self.params['l_f'] = l_f

                if f'beta_{i+1}' in self.param_keys:
                    beta[i,:] = self.params[f'beta_{i+1}']
                    self.params['beta'] = beta

    def read_chemistry_params(self):

        # Convert from logarithmic to linear scale
        self.params = self.log_to_linear(self.params, 
                                         ['log_C_ratio', 'log_O_ratio', 'log_P_quench'], 
                                         ['C_ratio', 'O_ratio', 'P_quench'], 
                                         )

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
                        self.VMR_species[f'{species_i}_{j}'] = 10**self.params[f'log_{species_i}_{j}']

                if f'log_{species_i}' in self.param_keys:
                    self.VMR_species[f'{species_i}'] = 10**self.params[f'log_{species_i}']
                elif species_i == '13CO' and ('log_C_ratio' in self.param_keys):
                    # Use isotope ratio to retrieve the VMR
                    self.VMR_species[species_i] = self.params['C_ratio'] * 10**self.params['log_12CO']
                elif species_i == 'C18O' and ('log_O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['O_ratio'] * 10**self.params['log_12CO']
                elif species_i == 'H2O_181' and ('log_O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['O_ratio'] * 10**self.params['log_H2O']
        
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
            self.params['log_X_cloud_base_MgSiO3'] = np.log10(10**self.params['log_X_MgSiO3'] * X_eq_MgSiO3)

        # Convert the cloud parameters from log to linear scale
        self.params = self.log_to_linear(self.params, 
            key_log=['log_K_zz', 'log_P_base_MgSiO3', 'log_X_cloud_base_MgSiO3', 'log_P_base_gray', 'log_opa_base_gray'], 
            key_lin=['K_zz', 'P_base_MgSiO3', 'X_cloud_base_MgSiO3', 'P_base_gray', 'opa_base_gray'], 
            )

    @classmethod
    def log_to_linear(cls, param_dict, key_log, key_lin, verbose=True):

        if not isinstance(key_log, (list, tuple, np.ndarray)):
            key_log = [key_log]
            
        if not isinstance(key_lin, (list, tuple, np.ndarray)):
            key_lin = [key_lin]
        
        for key_log_i, key_lin_i in zip(key_log, key_lin):

            if key_log_i not in list(param_dict.keys()):
                # key_log_i is not in the supplied dictionary
                param_dict[key_log_i] = None

                if verbose:
                    print(f'\nWarning: key_log_i={key_log_i} not found in dictionary, set to None.')

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