import scipy.stats as stats

import pandas as pd
import numpy as np

from .utils import sc

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

    def get(self, name, value=None, key='val'):

        queried_m_set = list(np.array(self.queried_m_set).flatten())
        query = 'm_set=={} and name=={}'.format(queried_m_set, [name])
        queried_table = self.table.query(query)
        
        if queried_table.empty:
            # Parameter not found
            return value
        
        return queried_table[key].values[0]
        
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
    
    def add_param(self, **kwargs):

        m_set = kwargs['m_set']
        name  = kwargs['name']

        # Check if parameter already exists
        query = f'm_set=="{m_set}" and name=="{name}"'
        queried_table = self.table.query(query)

        if queried_table.empty:
            # Add parameter to the table
            idx = len(self.table)            
        else:
            # Update the parameter in the table
            idx = queried_table.index[0]

        cols = list(kwargs.keys())

        self.table.loc[idx,:] = None
        self.table.loc[idx,cols] = list(kwargs.values())

        # Add linear parameter if log
        if not name.startswith('log_'):
            return
        
        val = kwargs.get('val')        
        if not isinstance(val, (float, int)):
            # Cannot take power 10
            return
        
        # Add the linear parameter
        cols = ['name', 'm_set', 'val']
        self.table.loc[idx+1,:] = None
        self.table.loc[idx+1,cols] = name.replace('log_',''), m_set, 10**val
    
    def add_params_dictionary(self, params, is_free=False):

        # Expand dictionary to have all model settings
        params_expanded = self.expand_per_model_setting(params)
        for m_set, dictionary in params_expanded.items():
            
            # Set of parameters for a model setting
            for name, val in dictionary.items():

                Param = None
                if is_free:
                    # Is a free parameter
                    Param = Parameter(name, *val, m_set)
                    val = np.nan

                # Add the parameter to the table
                self.add_param(name=name, m_set=m_set, Param=Param, val=val)

    def __init__(self, free_params, constant_params, model_settings, all_model_kwargs):
                 
        #PT_kwargs={}, chem_kwargs={}, cloud_kwargs={}, cov_kwargs={}, pRT_Radtrans_kwargs={}, **kwargs):

        self.model_settings = model_settings

        # Create the table
        cols = ['name', 'm_set', 'Param', 'val']
        self.table = pd.DataFrame(columns=cols)

        # Convert parallax to distance
        parallax = constant_params.get('parallax')
        if parallax is not None:
            constant_params['distance'] = 1e3/parallax

        # Add parameters to the table
        self.add_params_dictionary(constant_params, is_free=False)
        self.add_params_dictionary(free_params, is_free=True)

        # Add indices for free parameters
        self.table['idx_free'] = None
        mask = ~pd.isna(self.table['Param'])
        self.table.loc[mask,'idx_free'] = range(mask.sum())

        # Set queried model setting
        self.queried_m_set = 'all'

        # Add model kwargs
        self.add_model_kwargs(all_model_kwargs)

    def add_model_kwargs(self, all_model_kwargs):

        kwarg_keys = [
            'PT_kwargs', 'chem_kwargs', 'cloud_kwargs', 
            'cov_kwargs', 'loglike_kwargs',
            'rotation_kwargs', 'pRT_Radtrans_kwargs', 
            ]
        for key in kwarg_keys:
            model_kwargs = all_model_kwargs.get(key)
            if model_kwargs is None:
                raise ValueError(f'{key} not found in all_model_kwargs')
            
            # Expand the kwargs to each model setting
            model_kwargs = self.expand_per_model_setting(model_kwargs, is_kwargs=True)
            
            # Make attribute of ParameterTable class
            setattr(self, key, model_kwargs)

    def update_log_g(self, m_set):

        if self.get('log_g', key='idx_free') is not None:
            # log_g is a free parameter
            return
        
        M_p = self.get('M_p')
        R_p = self.get('R_p')
        if None in [M_p, R_p]:
            return
        
        # Compute the surface gravity
        M_p *= (sc.m_jup*1e3)
        R_p *= (sc.r_jup_mean*1e2)

        g = (sc.G*1e3) * M_p/R_p**2
        log_g = np.log10(g)
        
        # Add the parameter to the table
        self.add_param(name='log_g', m_set=m_set, val=log_g)

    def update_secondary_params(self, m_set):
        self.update_log_g(m_set)

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

        # Update secondary parameters, consider shared parameters initially
        for m_set in ['all', *self.model_settings]:
            # Query parameters specific to or shared between model settings
            self.queried_m_set = [m_set, 'all']
            self.update_secondary_params(m_set)
        self.queried_m_set = 'all'

        # Make attribute
        self.cube = cube_np

        if (ndim is None) and (nparams is None):
            return cube