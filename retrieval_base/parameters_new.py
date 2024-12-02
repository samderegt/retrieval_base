import scipy.stats as stats
import pandas as pd
import numpy as np

class Parameter:

    def __init__(self, name, prior_type, prior_params, mathtext, m_set):
        
        self.name  = name
        self.m_set = m_set
        self.mathtext = mathtext
        
        # Configure prior
        self.prior_type   = prior_type
        self.prior_params = prior_params
        self.set_prior()

    def set_prior(self):

        # Prior range or loc/scale
        loc, scale = self.prior_params

        if self.prior_type in ['U', 'uniform']:
            # Uniform prior
            scale = self.prior_params[1] - self.prior_params[0]
            self.prior = stats.uniform(loc=loc, scale=scale)
        elif self.prior_type in ['N', 'G', 'normal', 'gaussian']:
            # Gaussian prior
            self.prior = stats.norm(loc=loc, scale=scale)
        else:
            raise ValueError(f'prior_type={self.prior_type} not recognized')

        # Function to apply the prior
        self.apply_prior = self.prior.ppf

    def __call__(self, val_01):
        
        # Convert from [0,1] to the parameter space
        self.val_01 = val_01
        self.val    = self.apply_prior(self.val_01)

        return self.val
    
class ParameterTable:

    def get(self, name, value=None):

        query = f'm_set in ["all","{self.queried_m_set}"] and name=="{name}"'
        queried_table = self.table.query(query)
        
        if queried_table.empty:
            # Parameter not found
            return value
        
        return queried_table['val'].values[0]

    def expand_per_model_setting(self, d, is_kwargs=False):
        
        # Expand dictionary to have all model settings
        d_expanded = {'all': {}}

        if is_kwargs:
            # Make (empty) entry for each model setting
            for m_set in self.model_settings:
                d_expanded[m_set] = {}

        for name, val in d.items():
            if not isinstance(val, dict):
                # Is a single parameter
                d_expanded['all'][name] = val
                continue

            # Set of parameters for a model setting
            d_expanded[name] = val

        if not is_kwargs:
            # No need to adopt default values
            return d_expanded
        
        # Use the default parameters from the main model setting
        for m_set, val in d_expanded.items():
            d_expanded[m_set].update(d_expanded['all'])

        # Remove the main model setting
        d_expanded.pop('all')

        return d_expanded
    
    def add_params(self, params, is_free=False):

        cols = list(self.table.columns)
        idx  = len(self.table)

        params_expanded = self.expand_per_model_setting(params)
        for m_set, dictionary in params_expanded.items():
            
            # Set of parameters for a model setting
            for name, val in dictionary.items():
                
                Param = None
                if is_free:
                    # Is a free parameter
                    Param = Parameter(name, *val, m_set)
                    val = np.nan

                self.table.loc[idx,cols] = name, m_set, Param, val
                idx += 1

                if name.startswith('log_') and isinstance(val, (float, int)):
                    # Add the linear parameter
                    name = name.replace('log_','')
                    self.table.loc[idx,cols] = name, m_set, None, 10**val
                    idx += 1

    def __init__(self, free_params, constant_params, model_settings, PT_kwargs={}, chem_kwargs={}, cloud_kwargs={}, cov_kwargs={}, **kwargs):

        self.model_settings = model_settings

        # Create the table
        cols = ['name', 'm_set', 'Param', 'val']
        self.table = pd.DataFrame(columns=cols)

        # Add parameters to the table
        self.add_params(constant_params, is_free=False)
        self.add_params(free_params, is_free=True)

        # Add indices for free parameters
        self.table['idx_free'] = None
        mask = ~pd.isna(self.table['Param'])
        self.table.loc[mask,'idx_free'] = range(mask.sum())

        # Set queried model setting
        self.queried_m_set = 'all'

        # Expand the kwargs to each model setting
        self.PT_kwargs    = self.expand_per_model_setting(PT_kwargs, is_kwargs=True)
        self.chem_kwargs  = self.expand_per_model_setting(chem_kwargs, is_kwargs=True)
        self.cloud_kwargs = self.expand_per_model_setting(cloud_kwargs, is_kwargs=True)
        self.cov_kwargs   = self.expand_per_model_setting(cov_kwargs, is_kwargs=True)

    def __call__(self, cube, ndim=None, nparams=None):

        # Convert to a numpy array
        cube_np = np.array(cube[:ndim])
        cube_init = cube_np.copy()

        # Run unit cube through the priors
        for idx, (idx_free) in enumerate(self.table['idx_free']):
            if pd.isna(idx_free):
                # Not a free parameter
                continue

            # Apply prior of parameter
            Param = self.table.loc[idx]['Param']
            val = Param(cube_init[idx_free])

            # Update the cube and table
            cube_np[idx_free] = val
            self.table.loc[idx,'val'] = val

            # Keep cube as c-type array
            cube[idx_free] = val

            name = self.table.loc[idx,'name']
            if name.startswith('log_'):
                # Update the linear parameter too
                self.table.loc[idx+1,'val'] = 10**val

        # Make attribute
        self.cube = cube_np

        if (ndim is None) and (nparams is None):
            return cube