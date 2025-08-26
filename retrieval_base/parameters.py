import scipy.stats as stats

import pandas as pd
import numpy as np

import warnings

from .utils import sc

class Parameter:
    """Represents a parameter with a prior distribution."""

    def __init__(self, name, prior_type, prior_params, mathtext, m_set):
        """
        Args:
            name (str): Name of the parameter.
            prior_type (str): Type of the prior distribution.
            prior_params (tuple): Parameters of the prior distribution.
            mathtext (str): Mathematical representation of the parameter.
            m_set (str): Model setting associated with the parameter.
        """
        
        self.name  = name
        self.m_set = m_set
        self.mathtext = mathtext
        
        # Configure prior
        self.prior_type   = prior_type
        self.prior_params = prior_params
        self._set_prior()

        print(f'{name}: {prior_type}({prior_params[0]:.2f},{prior_params[1]:.2f})')

        self.apply_prior = True

    def __call__(self, val_01):
        """
        Applies the prior to a value in [0,1] and returns the parameter value.

        Args:
            val_01 (float): Value in the range [0,1].

        Returns:
            float: Parameter value after applying the prior.
        """
        # val_01 is in [0,1]
        self.val_01 = val_01

        if not self.apply_prior:
            # val_01 is already in parameter space
            self.val_01 = self.prior.cdf(self.val_01)

        # Apply the prior
        self.val = self.prior.ppf(self.val_01)

        return self.val
    
    def _set_prior(self):
        """Sets the prior distribution based on the prior type and parameters."""
        # Prior range or loc/scale
        loc, scale = self.prior_params

        if self.prior_type in ['U', 'uniform']:
            # Uniform prior
            if self.prior_params[1] < self.prior_params[0]:
                raise ValueError('Uniform priors should be defined as (min, max)')
            scale = self.prior_params[1] - self.prior_params[0]
            self.prior = stats.uniform(loc=loc, scale=scale)
        elif self.prior_type in ['N', 'G', 'normal', 'gaussian']:
            # Gaussian prior
            self.prior = stats.norm(loc=loc, scale=scale)
        else:
            raise ValueError(f'prior_type={self.prior_type} not recognized')
    
class ParameterTable:
    """Manages a table of parameters and their values."""

    def __init__(self, free_params, constant_params, model_settings, all_model_kwargs):
        """
        Initializes the ParameterTable.

        Args:
            free_params (dict): Dictionary of free parameters.
            constant_params (dict): Dictionary of constant parameters.
            model_settings (list): List of model settings.
            all_model_kwargs (dict): Dictionary of model keyword arguments.
        """
        self.model_settings = model_settings
        self.model_settings_linked = all_model_kwargs.get('model_settings_linked', {})

        # Create the table
        cols = ['name', 'm_set', 'Param', 'val']
        self.table = pd.DataFrame(columns=cols)

        # Convert parallax to distance
        parallax = constant_params.get('parallax')
        if parallax is not None:
            constant_params['distance'] = 1e3/parallax

        # Add parameters to the table
        self._add_params_dictionary(constant_params, is_free=False)
        self._add_params_dictionary(free_params, is_free=True)

        # Add indices for free parameters
        self.table['idx_free'] = None
        mask = ~pd.isna(self.table['Param'])
        self.n_free_params = mask.sum()

        self.table.loc[mask,'idx_free'] = range(self.n_free_params)

        # Set queried model setting
        self.set_queried_m_set('all')

        # Add model kwargs
        self._add_model_kwargs(all_model_kwargs)

    def __call__(self, cube, ndim=None, nparams=None):
        """
        Applies the priors to a unit cube and updates the parameter table.

        Args:
            cube (list): List of parameter values in the unit cube.
            ndim (int, optional): Number of dimensions.
            nparams (int, optional): Number of parameters.

        Returns:
            list: Updated cube with parameter values after applying priors.
        """
        # Reset whether the parameters are physical
        self.is_physical = True

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
            self.set_queried_m_set(m_set,'all')
            self._update_secondary_params(m_set)
        self.set_queried_m_set('all')

        # Make attribute
        self.cube = cube_np

        if (ndim is None) and (nparams is None):
            return cube
        
    def get(self, name, value=None, key='val'):
        """
        Retrieves the value of a parameter from the table.

        Args:
            name (str): Name of the parameter.
            value (optional): Default value if the parameter is not found.
            key (str): Column name to retrieve the value from.

        Returns:
            The value of the parameter or the default value if not found.
        """
        queried_table = self.queried_table.loc[self.queried_table.name == name]
        
        if queried_table.empty:
            # Parameter not found
            return value
            
        return queried_table[key].values[-1]
    
    def get_mathtext(self):
        """
        Retrieves the mathtext representations of free parameters.

        Returns:
            list: List of mathtext strings for free parameters.
        """
        mathtext = []
        for idx, (idx_free) in enumerate(self.table['idx_free']):
            if pd.isna(idx_free):
                # Not a free parameter
                continue

            # Get the mathtext of parameter
            mathtext.append(self.table.loc[idx]['Param'].mathtext)

        return mathtext

    def set_apply_prior(self, apply_prior=True):
        """
        Sets whether to apply the prior distribution to parameters.

        Args:
            apply_prior (bool): Whether to apply the prior distribution.
        """
        for idx, (idx_free) in enumerate(self.table['idx_free']):
            if pd.isna(idx_free):
                # Not a free parameter
                continue

            Param = self.table.loc[idx]['Param']
            Param.apply_prior = apply_prior

    def set_queried_m_set(self, *m_set, add_linked_m_set=False):
        """
        Sets the model setting to query parameters for.

        Args:
            m_set (str): Model setting to query.
            add_linked_m_set (bool): Whether to add linked model settings.
        """
        self.queried_m_set = list(m_set)
        if add_linked_m_set:
            for m_set_i in m_set:
                m_set_linked_i = np.atleast_1d(
                    self.model_settings_linked.get(m_set_i, [])
                )
                self.queried_m_set += list(m_set_linked_i)

        self.queried_table = self.table.loc[
            self.table.m_set.isin(self.queried_m_set)
            ]

    def _expand_dictionary_per_model_setting(self, d, is_kwargs=False):
        """
        Expands a dictionary to include all model settings.

        Args:
            d (dict): Dictionary to expand.
            is_kwargs (bool): Whether the dictionary contains keyword arguments.

        Returns:
            dict: Expanded dictionary with all model settings.
        """
        # Expand dictionary to have all model settings
        d_expanded = {'all': {}}

        if is_kwargs:
            # Make (empty) entry for each model setting
            for m_set in self.model_settings:
                d_expanded[m_set] = {}

        for name, val in d.items():
            if name in self.model_settings:
                # Set of parameters for a model setting
                d_expanded[name] = val
                continue

            # Applies to all model settings
            d_expanded['all'][name] = val

        if not is_kwargs:
            # No need to adopt default values
            return d_expanded
        
        # Use the default parameters from the main model setting
        for m_set, val in d_expanded.items():
            d_expanded[m_set].update(d_expanded['all'])

        # Remove the main model setting
        d_expanded.pop('all')

        return d_expanded
    
    def _add_model_kwargs(self, all_model_kwargs):
        """
        Adds model keyword arguments to the table.

        Args:
            all_model_kwargs (dict): Dictionary of all model keyword arguments.
        """
        kwarg_keys = [
            'PT_kwargs', 'chem_kwargs', 'cloud_kwargs', 
            'line_opacity_kwargs', 
            'rotation_kwargs', 'pRT_Radtrans_kwargs', 
            'cov_kwargs', 'loglike_kwargs', 
            ]
        for key in kwarg_keys:
            
            model_kwargs = all_model_kwargs.get(key, {})
            
            if len(model_kwargs) == 0:
                warnings.warn(f'{key} not found in all_model_kwargs')
            
            if key not in ['cov_kwargs', 'loglike_kwargs']:
                # Expand the kwargs to each model setting
                model_kwargs = self._expand_dictionary_per_model_setting(model_kwargs, is_kwargs=True)
                
            # Make attribute of ParameterTable class
            setattr(self, key, model_kwargs)

    def _add_param(self, **kwargs):
        """
        Adds or updates a parameter in the table.

        Args:
            **kwargs: Parameter attributes to add or update.
        """
        m_set = kwargs['m_set']
        name  = kwargs['name']

        # Check if parameter already exists
        query = f'm_set=="{m_set}" and name=="{name}"'
        queried_table = self.table.query(query)

        if queried_table.empty:
            # Add parameter to the table
            idx = len(self.table)
            self.table.loc[idx,:] = None
        else:
            # Update the parameter in the table
            idx = queried_table.index[0]

        cols = list(kwargs.keys())
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
    
    def _add_params_dictionary(self, params, is_free=False):
        """
        Adds parameters from a dictionary to the table.

        Args:
            params (dict): Dictionary of parameters to add.
            is_free (bool): Whether the parameters are free parameters.
        """
        if is_free:
            print('\n'+'='*50+'\nFree parameters and priors')

        # Expand dictionary to have all model settings
        params_expanded = self._expand_dictionary_per_model_setting(params)
        for m_set, dictionary in params_expanded.items():
            
            # Set of parameters for a model setting
            for name, val in dictionary.items():

                Param = None
                if is_free:
                    # Is a free parameter
                    Param = Parameter(name, *val, m_set)
                    val = np.nan

                # Add the parameter to the table
                self._add_param(name=name, m_set=m_set, Param=Param, val=val)

    def _update_log_g(self, m_set):
        """
        Updates the surface gravity parameter for a model setting.

        Args:
            m_set (str): Model setting to update.
        """
        if self.get('log_g', key='idx_free') is not None:
            # log_g is a free parameter
            return

        if (self.get('log_g') is not None) and (m_set != 'all'):
            # log_g is already set for all model settings
            return
        
        M_p = self.get('M_p')
        R_p = self.get('R_p')
        if None in [M_p, R_p]:
            return
        
        # Compute the surface gravity
        M_p *= (sc.m_jup*1e3)
        R_p *= (sc.r_jup_mean*1e2)

        g = (sc.G*1e3) * M_p/R_p**2
        if g <= 0.:
            # Invalid surface gravity
            self.is_physical = False
            return
        log_g = np.log10(g)
        
        # Add the parameter to the table
        self._add_param(name='log_g', m_set=m_set, val=log_g)

    def _update_coverage_fraction(self, m_set):
        """
        Updates the coverage fraction parameter for a model setting.

        Args:
            m_set (str): Model setting to update.
        """
        if self.get('coverage_fraction', key='idx_free') is None:
            # coverage_fraction is not a free parameter, no need to update
            return

        # Check the given coverage_fraction
        cf = self.get('coverage_fraction', 1.)
        if cf == 1.:
            # Full coverage (default, e.g. binary)
            return

        # Partial coverage
        if (m_set == self.model_settings[0]) and (len(self.model_settings) == 2):
            # Add the remainder to the second model setting
            self._add_param(name='coverage_fraction', m_set=self.model_settings[1], val=1.-cf)

    def _update_secondary_params(self, m_set):
        """
        Updates secondary parameters for a model setting.

        Args:
            m_set (str): Model setting to update.
        """
        # Update the surface gravity of this model setting
        self._update_log_g(m_set)
        
        # Update the coverage fraction of this model setting
        self._update_coverage_fraction(m_set)