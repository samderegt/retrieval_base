import numpy as np
from scipy.stats import invgamma, norm

from petitRADTRANS.retrieval import cloud_cond as fc

from .chemistry import Chemistry

class ParameterWaveSetting:

    @classmethod
    def log_to_linear(cls, param_dict, key_log, key_lin=None):

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
                param_dict[key_log_i] = np.array(val_log, dtype=np.float64)
                param_dict[key_lin_i] = 10**np.array(val_log, dtype=np.float64)

            elif val_log is None:
                # Set linear parameter to None as well
                param_dict[key_lin_i] = None

        return param_dict

    def __init__(self, 
            free_params, 
            constant_params, 
            m_set, 
            n_orders=7, 
            n_dets=3, 
            PT_kwargs={}, 
            chem_kwargs={}, 
            cov_kwargs={}, 
            cloud_kwargs={}, 
            **kwargs, 
            ):
        
        self.m_set    = m_set
        self.n_orders = n_orders
        self.n_dets   = n_dets
        
        # Separate the prior range from the mathtext label
        self.param_priors, self.param_mathtext, self.param_idx = {}, {}, {}
        for key_i, (prior_i, mathtext_i, idx_i) in free_params.items():
            self.param_priors[key_i]   = prior_i
            self.param_mathtext[key_i] = mathtext_i
            self.param_idx[key_i]      = idx_i

        self.param_keys = np.array(list(self.param_priors.keys()))
        self.n_params = len(self.param_keys)

        # Create dictionary with constant parameter-values
        self.params = constant_params.copy()

        for key_i in list(self.params.keys()):
            if key_i.startswith('log_'):
                self.params = self.log_to_linear(self.params, key_i)

        # Check the used PT profile
        self.PT_kwargs = PT_kwargs
        self.PT_mode   = self.PT_kwargs.get('PT_mode', 'free_gradient')
        self.n_T_knots = self.PT_kwargs.get('n_T_knots')
        assert(self.PT_mode in ['free', 'free_gradient', 'grid'])

        # Check the used chemistry type
        self.chem_kwargs = chem_kwargs
        self.chem_mode   = self.chem_kwargs.get('chem_mode', 'free')
        assert(self.chem_mode in ['pRT_table', 'free', 'fastchem', 'fastchem_table'])
        
        # Check the used cloud type
        self.cloud_kwargs = cloud_kwargs
        self.cloud_mode   = self.cloud_kwargs.get('cloud_mode', None)
        if self.cloud_mode == 'grey':
            self.cloud_mode = 'gray'
        assert(self.cloud_mode in ['gray', 'EddySed', None])

        # Check the used covariance definition
        self.cov_kwargs = cov_kwargs
        self.cov_mode   = self.cov_kwargs.get('cov_mode', None)
        assert(self.cov_mode in ['GP', None])

    def read_PT_params(self, cube=None):

        if self.params.get('temperature') is not None:
            return cube

        if self.PT_mode == 'grid':
            return cube

        # If free, or free-gradient, define the pressure knots
        if self.params.get('log_P_knots') is None:
            self.params['log_P_knots'] = np.zeros(self.PT_kwargs.get('n_T_knots'), dtype=np.float64)
        else:
            self.params['log_P_knots'] = np.array(self.params.get('log_P_knots'), dtype=np.float64)

        if self.params.get('d_log_P_01') is not None:
            # Separations are given relative to bottom knot
            log_P_knots = [self.params['log_P_range'][1], ]

            for i in range(self.n_T_knots-1):

                up_i = self.param.get(f'd_log_P_{i}{i+1}')
                if up_i is not None:
                    # Add knot above previous knot
                    log_P_knots.append(log_P_knots[-1]-up_i)
            
            log_P_knots.append(self.params['log_P_range'][0])
            
            self.params['log_P_knots'] = np.sort(np.array(log_P_knots))

        if self.params.get('log_P_phot') is not None:
            # Separations are given relative to photospheric knot
            log_P_phot  = self.params['log_P_phot']
            log_P_knots = [
                log_P_phot, self.params['log_P_range'][1], 
                self.params['log_P_range'][0], 
                ]

            for i in range(1, self.n_T_knots-1):
                up_i  = self.params.get(f'd_log_P_phot+{i}')
                low_i = self.params.get(f'd_log_P_phot-{i}')
                if self.PT_kwargs.get('symmetric_around_P_phot', False):
                    low_i = up_i
                
                if up_i is not None:
                    # Add knot above P_phot-knot
                    log_P_knots.append(log_P_phot-up_i)
                if low_i is not None:
                    # Add knot below P_phot-knot
                    log_P_knots.append(log_P_phot+low_i)

            self.params['log_P_knots'] = np.sort(np.array(log_P_knots))

        self.params['P_knots']     = 10**self.params['log_P_knots']
        self.params['ln_P_knots']  = np.log(self.params['P_knots'])

        if self.PT_mode == 'free':

            # Following De Regt et al. (2024)
            # TODO: this doesn't work with the photospheric-knot method
            
            # Combine the upper temperature knots into an array
            self.params['T_knots'] = []
            for i in range(self.n_T_knots-1):

                T_i = self.params[f'T_{i+1}']

                if (cube is not None) and self.PT_kwargs.get('enforce_PT_corr', False):
                    # Temperature knot is product of previous knots
                    T_i = np.prod([self.params[f'T_{j}'] for j in range(i+2)])
                    
                    idx = np.argwhere(self.param_keys==f'T_{i+1}').flatten()[0]
                    cube[idx] = T_i

                self.params['T_knots'].append(T_i)

            self.params['T_knots'] = np.array(self.params['T_knots'])[::-1]

        if self.PT_mode == 'free_gradient':

            # Following Zhang et al. (2023)

            # Define the temperature gradients at each knot
            self.params['dlnT_dlnP_knots'] = np.array([
                self.params[f'dlnT_dlnP_{i}'] for i in range(self.n_T_knots)
                ])
            self.params['dlnT_dlnP_knots'] = self.params['dlnT_dlnP_knots'][::-1]
            
        return cube
    
    def read_uncertainty_params(self):

        for key_i in ['beta', 'a', 'l']:

            if (self.params.get(key_i) is None) and (self.params.get(f'{key_i}_1') is None):
                continue

            # Reshape to values for each order and detector
            self.params[key_i] *= np.ones((self.n_orders, self.n_dets))

            for j in range(self.n_orders):
                
                if self.params.get(f'{key_i}_{j+1}') is None:
                    continue

                # Replace the constant with a free parameter for each order
                self.params[key_i][j,:] = self.params[f'{key_i}_{j+1}']

    def read_chemistry_params(self):

        if self.chem_mode in ['pRT_table', 'fastchem', 'fastchem_table']:
            # Use chemical equilibrium
            self.VMR_species = None

        elif self.chem_mode == 'free':
            # Use free chemistry
            self.params['C/O'], self.params['Fe/H'] = None, None

            # Loop over all possible species
            self.VMR_species = {}
            for species_i in Chemistry.species_info.index:

                '''
                # If multiple VMRs are given
                for j in range(3):
                    if f'log_{species_i}_{j}' in self.param_keys:
                        self.VMR_species[f'{species_i}_{j}'] = self.params[f'{species_i}_{j}']
                '''

                if f'log_{species_i}' in self.param_keys:
                    self.VMR_species[f'{species_i}'] = self.params[f'{species_i}']
                    continue

                '''
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
                '''

    def __call__(self, cube, apply_prior=True):

        # Loop over all parameters
        for key_i, i in self.param_idx.items():

            if apply_prior:
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

        if (self.params.get('M_p') is not None) and (self.params.get('R_p') is not None):
            G = 6.6743e-8 # cm^3 g^-1 s^-2
            M_p = self.params['M_p'] * 1.899e30 # g
            R_p = self.params['R_p'] * 7.149e9 # cm

            g = G*M_p/R_p**2
            self.params['log_g'] = np.log10(g)

        idx = list(self.param_idx.values())
        return cube[idx]

class Parameters:

    @classmethod
    def expand_per_setting(cls, d, model_settings, return_indices=False):

        is_dict = isinstance(d, dict)
        if is_dict:
            new_d = {m_set: {} for m_set in model_settings}
            items = d.items()
        else:
            new_d = {m_set: [] for m_set in model_settings}
            if not isinstance(d, (np.ndarray, list, tuple)):
                # Only a single value given
                items = [d]
            else:
                # Multiple values given
                items = d
        
        unique_d = []
        for res_i in items:

            # Other items
            val_i = res_i

            if is_dict:
                # Dictionary items
                key_i, val_i = res_i

                if key_i not in model_settings:
                    # Parameter applies to all settings
                    for m_set in model_settings:
                        new_d[m_set][key_i] = val_i
                    unique_d.append(f'{key_i}')
                    continue
                else:
                    # Parameter for specific setting
                    if not isinstance(val_i, dict):
                        # Only one value for this setting (should only 
                        # occur for modes, not free/constant parameters)
                        new_d[key_i] = val_i
                        continue

                    for key_j, val_j in val_i.items():
                        # val_i is dictionary too
                        # Add to existing dictionary for this setting
                        new_d[key_i][key_j] = val_j

                        # Is a unique free-parameter
                        unique_d.append(f'{key_j}_{key_i}')
                    continue
            
            # Parameter applies to all settings
            for m_set in model_settings:
                # Add to each setting-dictionary
                new_d[m_set] = val_i

        if not return_indices:
            return new_d

        # All free-parameters to be retrieved
        unique_d = np.array(unique_d)
        n_params = len(unique_d)
        
        # Assign an index to each parameter
        new_idx_d = new_d.copy()
        for i, (m_set, val_i) in enumerate(new_d.items()):

            for key_i in val_i.keys():

                # Parameter applies to all settings
                mask = (unique_d == key_i)
                if mask.any() and (i != 0):
                    continue

                if not mask.any():
                    # Parameter is specific to one setting
                    mask = (unique_d == f'{key_i}_{m_set}')

                idx = np.argwhere(mask).flatten()[0]
                new_idx_d[m_set][key_i].append(idx)

        return new_idx_d, n_params, unique_d

    def __init__(
            self, 
            free_params, 
            constant_params, 
            model_settings, 
            **kwargs, 
            ):
        
        self.apply_prior = True
        
        self.model_settings = model_settings
        self.model_settings_names = list(self.model_settings.keys())

        # Re-structure dictionaries so that each setting has necessary parameters
        free_params, self.n_params, self.unique_param_keys = self.expand_per_setting(
            free_params, self.model_settings_names, return_indices=True
            )
        constant_params = self.expand_per_setting(
            constant_params, self.model_settings_names
            )
        kwargs = {
            key_i: self.expand_per_setting(val_i, self.model_settings_names) \
            for key_i, val_i in kwargs.items()
            }
        
        # Create ParamWaveSetting instances per wavelength-setting
        self.Params_m_set = {}
        self.param_mathtext = np.array([None]*self.n_params)
        self.param_keys     = np.array([None]*self.n_params)
        for m_set in self.model_settings_names:

            # Give only kwargs for this setting
            kwargs_i = {
                key_i: val_i[m_set] for key_i, val_i in kwargs.items()
                }

            self.Params_m_set[m_set] = ParameterWaveSetting(
                free_params[m_set], 
                constant_params[m_set], 
                m_set=m_set,
                n_orders=self.model_settings[m_set][0], 
                n_dets=self.model_settings[m_set][1], 
                **kwargs_i
                )
            
            idx = list(self.Params_m_set[m_set].param_idx.values())
            keys     = self.Params_m_set[m_set].param_mathtext.keys()
            mathtext = self.Params_m_set[m_set].param_mathtext.values()
            self.param_mathtext[idx] = list(mathtext)
            self.param_keys[idx]     = list(keys)
                    
        self.PT_kwargs    = kwargs.get('PT_kwargs')
        self.chem_kwargs  = kwargs.get('chem_kwargs')
        self.cov_kwargs   = kwargs.get('cov_kwargs')
        self.cloud_kwargs = kwargs.get('cloud_kwargs')

    def get_sorted_kwargs(self):

        list_to_return = [
            self.PT_kwargs.copy(), self.chem_kwargs.copy(), 
            self.cov_kwargs.copy(), self.cloud_kwargs.copy(),
            ]
        del self.PT_kwargs, self.chem_kwargs, self.cov_kwargs, self.cloud_kwargs
        return list_to_return
            
    def __call__(self, cube, ndim=None, nparams=None):

        # Convert to numpy array
        cube_copy = np.array(cube[:ndim])
        cube_init = cube_copy.copy()

        # Loop over each wavelength-setting
        for Param_i in self.Params_m_set.values():

            # Select indices relevant for this setting
            idx = list(Param_i.param_idx.values())

            # Run through priors and update the dictionaries
            cube_copy[idx] = Param_i(cube_init.copy(), apply_prior=self.apply_prior)

        for i, cube_i in enumerate(cube_copy):
            # Keep 'cube' as c-type array
            cube[i] = cube_i
        
        self.cube = cube_copy
    
        if (ndim is None) and (nparams is None):
            return cube
        else:
            return
