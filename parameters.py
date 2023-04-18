import numpy as np

from petitRADTRANS.retrieval import cloud_cond as fc

from spectrum import Spectrum
from chemistry import Chemistry

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
        'log_P_knots': np.array([-6,-2,0,1,2]), 

        # PT profile (Molliere et al. 2020)
        'log_P_phot': 0.0, 
        'alpha': 1.0, 
        'T_int': 1800, 

    }

    def __init__(self, free_params, constant_params, n_orders=7, n_dets=3):

        # Separate the prior range from the mathtext label
        self.param_priors, self.param_mathtext = {}, {}
        for key_i, (prior_i, mathtext_i) in free_params.items():
            self.param_priors[key_i]   = prior_i
            self.param_mathtext[key_i] = mathtext_i

        self.param_keys = np.array(list(self.param_priors.keys()))
        self.n_params = len(self.param_keys)

        # Count the number of temperature knots above the base
        for i in range(1,100):
            if f'T_{i}' not in self.param_keys:
                self.n_T_knots = i-1
                break

        #self.constant_params = constant_params
        # Create dictionary with constant parameter-values
        self.params = self.all_params.copy()
        self.params.update(constant_params)

        # Check if Molliere et al. (2020) PT-profile is used
        Molliere_param_keys = ['log_P_phot', 'alpha', 'T_int', 'T_1', 'T_2', 'T_3']
        if all([(param_i in self.param_keys) for param_i in Molliere_param_keys]):
            self.PT_mode = 'Molliere'
        else:
            self.PT_mode = 'free'

        # Check if equilibrium- or free-chemistry is used
        eqchem_param_keys = ['C/O', 'Fe/H']
        if all([(param_i in self.param_keys) for param_i in eqchem_param_keys]):
            self.chem_mode = 'eqchem'
        else:
            self.chem_mode = 'free'

        # Check if clouds (and which type) are included
        self.cloud_mode = None
        MgSiO3_param_keys = ['log_X_MgSiO3', 'f_sed', 'log_K_zz', 'sigma_g']
        gray_cloud_param_keys = ['log_opa_base_gray', 'log_P_base_gray', 'f_sed_gray']
        if all([(param_i in self.param_keys) for param_i in MgSiO3_param_keys]):
            self.cloud_mode = 'MgSiO3'
        elif all([(param_i in self.param_keys) for param_i in gray_cloud_param_keys]):
            self.cloud_mode = 'gray'

        self.n_orders = n_orders
        self.n_dets   = n_dets

    def __call__(self, cube, ndim, nparams):
        '''
        Update values in the params dictionary.

        Input
        -----
        cube : np.ndarray
            Array of values between 0-1 for the free parameters.
        ndim : int
            Number of dimensions. (Equal to number of free parameters)
        nparams : int
            Number of free parameters.
        '''

        # Update values in the params-dictionary
        #cube = np.array(cube[:ndim])
        self.cube_copy = np.array(cube[:ndim])

        # Loop over all parameters
        for i, key_i in enumerate(self.param_keys):

            # Sample within the boundaries
            low, high = self.param_priors[key_i]
            cube[i] = low + (high-low)*cube[i]

            self.params[key_i] = cube[i]

        # Read the parameters for the model's segments
        cube = self.read_PT_params(cube)
        self.read_uncertainty_params()
        self.read_chemistry_params()
        self.read_cloud_params()

        return
    
    def read_PT_params(self, cube=None):

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

        # Combine the upper temperature knots into an array
        self.params['T_knots'] = np.array(
            [self.params[f'T_{i+1}'] \
             for i in range(self.n_T_knots)]
            )[::-1]

        # Convert from logarithmic to linear scale
        self.params = self.log_to_linear(self.params, 
            ['log_gamma', 'log_P_knots'], 
            ['gamma', 'P_knots']
            )

        return cube

    def read_uncertainty_params(self):

        # Convert the global value to linear scale
        self.params = self.log_to_linear(self.params, ['log_a', 'log_l'], ['a', 'l'])

        # Reshape if only one value is given
        if isinstance(self.params['a'], (float, int)):
            self.params['a'] = np.ones((self.n_orders, self.n_dets)) * self.params['a']
        if isinstance(self.params['l'], (float, int)):
            self.params['l'] = np.ones((self.n_orders, self.n_dets)) * self.params['l']
        if isinstance(self.params['beta'], (float, int)):
            self.params['beta'] = np.ones((self.n_orders, self.n_dets)) * self.params['beta']

        # Make a copy of the global values
        a, l, beta = np.copy(self.params['a']), np.copy(self.params['l']), np.copy(self.params['beta'])
        
        for i in range(self.n_orders):
            for j in range(self.n_dets):

                # Replace the constants with the free parameters
                if f'log_a_{i+1}' in self.param_keys:
                    a[i,:] = 10**self.params[f'log_a_{i+1}']
                    self.params['a'] = a
                if f'log_l_{i+1}' in self.param_keys:
                    l[i,:] = 10**self.params[f'log_l_{i+1}']
                    self.params['l'] = l
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

                if f'log_{species_i}' in self.param_keys:
                    self.VMR_species[f'{species_i}'] = 10**self.params[f'log_{species_i}']
                elif species_i == '13CO' and ('log_C_ratio' in self.param_keys):
                    # Use isotope ratio to retrieve the VMR
                    self.VMR_species[species_i] = self.params['C_ratio'] * 10**self.params['log_12CO']
                elif species_i == 'C18O' and ('log_O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['O_ratio'] * 10**self.params['log_12CO']
                elif species_i == 'H2O_181' and ('log_O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['O_ratio'] * 10**self.params['log_H2O']
        
    def read_cloud_params(self):

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