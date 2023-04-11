import numpy as np

from petitRADTRANS.retrieval import cloud_cond as fc

from .spectrum import Spectrum
from .chemistry import Chemistry

class Parameters:

    def __init__(self, param_priors, param_constant, param_mathtext):

        self.param_priors   = param_priors
        self.param_constant = param_constant
        self.param_mathtext = param_mathtext

        self.param_keys = np.array(list(self.param_priors.keys()))

        # Create dictionary with constant parameter-values
        self.params = self.param_constant.copy()

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

    def __call__(self, cube, ndim, nparams):

        # Update values in the params-dictionary

        # Loop over all parameters
        for i, key_i in enumerate(self.param_keys):

            # Sample within the boundaries
            low, high = self.param_priors[key_i]
            self.params[key_i] = low + (high-low)*cube[i]

        # PT profile parameterization from Molliere et al. (2020)
        if self.PT_mode == 'Molliere':

            # T_0 is connection temperature at P=0.1 bar
            T_0 = (3/4 * self.params['T_int']**4 * (0.1 + 2/3))**(1/4)

            # Define the prior based on the other knots
            low, high = self.param_priors['T_1']
            self.params['T_1'] = T_0 * (high - (high-low)*cube[self.param_keys=='T_1'])

            low, high = self.param_priors['T_2']
            self.params['T_2'] = self.params['T_1'] * (high - (high-low)*cube[self.param_keys=='T_2'])

            low, high = self.param_priors['T_3']
            self.params['T_3'] = self.params['T_2'] * (high - (high-low)*cube[self.param_keys=='T_3'])

        # Use same naming-scheme for each temperature knot
        if 'T_bottom' in self.param_keys:
            self.params['T_0'] = self.params['T_bottom']

        # Convert from logarithmic to linear scale
        if 'log_gamma' in self.param_keys:
            self.params['gamma'] = 10**params['log_gamma']

        self.read_uncertainty_params()
        self.read_chemistry_params()
        self.read_cloud_params()

        return

    def read_uncertainty_params(self):

        # Convert the global value to linear scale
        self.params['a']   = 10**self.params['log_a']
        self.params['tau'] = 10**self.params['log_tau']

        if not isinstance(self.params['a'], (np.ndarray, list)):
            self.params['a'] = [self.params['a']] * len(Spectrum.order_wlen_ranges)
        if not isinstance(self.params['tau'], (np.ndarray, list)):
            self.params['tau'] = [self.params['tau']] * len(Spectrum.order_wlen_ranges)

        # Make a copy of the global values
        a, tau, beta = np.copy(self.params['a']), np.copy(self.params['tau']), np.copy(self.params['beta'])
        for i in range(Spectrum.n_orders):
            # Replace the constants with the free parameters
            if f'log_a_{i+1}' in self.param_keys:
                a[i*3:i*3+3] = 10**self.params[f'log_a_{i+1}']
                self.params['a'] = a
            if f'log_tau_{i+1}' in self.param_keys:
                tau[i*3:i*3+3] = 10**self.params[f'log_tau_{i+1}']
                self.params['tau'] = tau
            if f'beta_{i+1}' in self.param_keys:
                beta[i*3:i*3+3] = self.params[f'beta_{i+1}']
                self.params['beta'] = beta

    def read_chemistry_params(self):

        # Convert the isotope ratios to linear scale
        self.params['C_ratio'] = 10**self.params['log_C_ratio']
        self.params['O_ratio'] = 10**self.params['log_O_ratio']

        if 'C/O' in param_keys:
            # Use chemical equilibrium
            self.VMR_species = None
        else:
            # Use free chemistry
            self.params['C/O'], self.params['Fe/H'] = None, None

            # Loop over all possible species
            self.VMR_species = {}
            for species_i in Chemistry.species_info.keys():

                if f'log_{species_i}' in self.param_keys:
                    self.VMR_species[species_i] = 10**self.params[f'log_{species_i}']
                elif species_i == '13CO' and ('log_C_ratio' in self.param_keys):
                    # Use isotope ratio to retrieve the VMR
                    self.VMR_species[species_i] = self.params['C_ratio'] * 10**self.params['log_12CO']
                elif species_i == 'C18O' and ('log_O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['O_ratio'] * 10**self.params['log_12CO']
                elif species_i == 'H2O_181' and ('log_O_ratio' in self.param_keys):
                    self.VMR_species[species_i] = self.params['O_ratio'] * 10**self.params['log_H2O']

    def read_cloud_params(self):

        if 'log_K_zz' in self.param_keys:
            self.params['K_zz'] = 10**self.params['log_K_zz']
        else:
            self.params['K_zz'] = None

        if ('log_X_MgSiO3' in self.param_keys) and ('C/O' in self.param_keys):
            # Return the eq.-chem. mass fraction of MgSiO3
            X_eq_MgSiO3 = fc.return_XMgSiO3(self.params['Fe/H'], self.params['C/O'])
            # Pressure at the cloud base
            # TODO: this doesn't work, temperature is not yet given
            self.params['P_base_MgSiO3'] = fc.simple_cdf_MgSiO3(pressure, temperature, self.params['Fe/H'], self.params['C/O'])

            # Log mass fraction at the cloud base
            self.params['log_X_cloud_base_MgSiO3'] = np.log10(10**self.params['log_X_MgSiO3'] * X_eq_MgSiO3)
            self.params['X_cloud_base_MgSiO3'] = 10**self.params['log_X_cloud_base_MgSiO3']
        
        elif 'log_X_cloud_base_MgSiO3' in self.param_keys:
            # Not using chem-eq, fitting for cloud base VMR and pressure
            self.params['P_base_MgSiO3'] = 10**self.params['log_P_base_MgSiO3']

            self.params['X_cloud_base_MgSiO3'] = 10**self.params['log_X_cloud_base_MgSiO3']

        else:
            self.params['P_base_MgSiO3'] = None

        # Convert from logarithmic to linear scale
        if 'log_P_base_gray' in self.param_keys:
            self.params['P_base_gray'] = 10**self.params['log_P_base_gray']
        if 'log_opa_base_gray' in self.param_keys:
            self.params['opa_base_gray'] = 10**self.params['log_opa_base_gray']