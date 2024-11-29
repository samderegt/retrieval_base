import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.interpolate import make_interp_spline

import petitRADTRANS.poor_mans_nonequ_chem as pm

def get_PT_profile_class(pressure, PT_mode='free_gradient', **kwargs):
    """
    Get the PT profile class based on the mode.

    Parameters:
    pressure (array): 
        Pressure levels.
    PT_mode (str): 
        Mode for the PT profile.
    **kwargs: 
        Additional keyword arguments.

    Returns:
    PT_profile: 
        Instance of the PT profile class.
    """
    if PT_mode == 'free':
        return PT_profile_free(pressure, **kwargs)
    if PT_mode == 'free_gradient':
        return PT_profile_free_gradient(pressure, **kwargs)
    if PT_mode == 'grid':
        return PT_profile_SONORA(pressure, **kwargs)
    if PT_mode == 'static':
        return PT_profile(pressure, **kwargs)

class PT_profile:
    """
    Base class for PT profiles.
    """

    def __init__(self, pressure):
        """
        Initialize the PT_profile class.

        Parameters:
        pressure (array): 
            Pressure levels.
        """
        self.pressure = pressure
        self.temperature_envelopes = None

    def __call__(self, params):
        """
        Retrieve the temperature profile.

        Parameters:
        params (dict): 
            Parameters for the PT profile.

        Returns:
        array or float: 
            Temperature profile or -np.inf if failed.
        """
        self.temperature = params.get('temperature')
        if self.temperature is not None:
            return self.temperature
        
        return -np.inf

class PT_profile_SONORA(PT_profile):
    """
    Class for SONORA PT profiles.
    """

    def __init__(self, pressure, path_to_SONORA_PT='./data/SONORA_PT_structures', **kwargs):
        """
        Initialize the PT_profile_SONORA class.

        Parameters:
        pressure (array): 
            Pressure levels.
        path_to_SONORA_PT (str): 
            Path to the SONORA PT structures.
        **kwargs: 
            Additional keyword arguments.
        """
        # Give arguments to the parent class
        super().__init__(pressure)

        import glob, os
        all_paths = glob.glob(os.path.join(path_to_SONORA_PT, '*/t*.dat'))
        all_paths = np.sort(all_paths)

        # T_eff, log_g, C/O, Fe/H
        self.T_eff_grid = np.nan * np.ones(all_paths.shape)
        self.log_g_grid = np.nan * np.ones(all_paths.shape)
        self.CO_grid = np.nan * np.ones(all_paths.shape)
        self.FeH_grid = np.nan * np.ones(all_paths.shape)

        for i, path_i in enumerate(all_paths):
            # Read from the file headers
            T_eff_i, g_i, FeH_i, CO_i = \
                np.loadtxt(path_i, max_rows=1, usecols=(0,1,5,6))
            T_eff_i = np.around(T_eff_i, -1)

            # Convert to cgs and take log10
            log_g_i = np.log10(g_i*1e2)
            log_g_i = np.around(log_g_i, 2)
            
            # Convert from Solar to actual C/O
            CO_i = CO_i*0.6

            self.T_eff_grid[i] = T_eff_i
            self.log_g_grid[i] = log_g_i
            self.CO_grid[i]    = CO_i
            self.FeH_grid[i]   = FeH_i

        # Get only one of each entry
        self.T_eff_grid = np.sort(np.unique(self.T_eff_grid))
        self.log_g_grid = np.sort(np.unique(self.log_g_grid))
        self.CO_grid    = np.sort(np.unique(self.CO_grid))
        self.FeH_grid   = np.sort(np.unique(self.FeH_grid))

        self.temperature_grid = np.nan * np.ones(
            (len(self.T_eff_grid), len(self.log_g_grid), 
             len(self.CO_grid), len(self.FeH_grid), 50)
            )

        self.rad_conv_boundary_grid = np.nan * np.ones(
            (len(self.T_eff_grid), len(self.log_g_grid), 
             len(self.CO_grid), len(self.FeH_grid))
            )
        
        for i, path_i in enumerate(all_paths):
            # Read from the file headers
            T_eff_i, g_i, FeH_i, CO_i = \
                np.loadtxt(path_i, max_rows=1, usecols=(0,1,5,6))
            T_eff_i = np.around(T_eff_i, -1)

            # Convert to cgs and take log10
            log_g_i = np.log10(g_i*1e2)
            log_g_i = np.around(log_g_i, 2)
            
            # Convert from Solar to actual C/O
            CO_i = CO_i*0.6

            mask_T_eff = (self.T_eff_grid == T_eff_i)
            mask_log_g = (self.log_g_grid == log_g_i)
            mask_CO    = (self.CO_grid == CO_i)
            mask_FeH   = (self.FeH_grid == FeH_i)

            # Load the PT profile
            pressure_i, temperature_i, ad_gradient_i, rad_gradient_i = np.genfromtxt(
                path_i, skip_header=1, usecols=(1,2,4,5), delimiter=(3,12,10,11,8,8)
                ).T

            # Schwarzschild criterion
            self.rad_conv_boundary_grid[mask_T_eff,mask_log_g,mask_CO,mask_FeH] = \
                pressure_i[(ad_gradient_i < rad_gradient_i)].min()

            self.temperature_grid[mask_T_eff,mask_log_g,mask_CO,mask_FeH,:] = interp1d(
                pressure_i, temperature_i, kind='cubic', fill_value='extrapolate'
                )(self.pressure)

        # Set-up an function to conduct 4D-interpolation
        points = ()
        if len(self.T_eff_grid) > 1:
            points += (self.T_eff_grid, )
        if len(self.log_g_grid) > 1:
            points += (self.log_g_grid, )
        if len(self.CO_grid) > 1:
            points += (self.CO_grid, )
        if len(self.FeH_grid) > 1:
            points += (self.FeH_grid, )

        self.temperature_grid = np.squeeze(self.temperature_grid)
        self.rad_conv_boundary_grid = np.squeeze(self.rad_conv_boundary_grid)

        self.temperature_interp_func = RegularGridInterpolator(
            points=points, values=self.temperature_grid
            )
        
        self.rad_conv_boundary_interp_func = RegularGridInterpolator(
            points=points, values=self.rad_conv_boundary_grid
            )
        
    def __call__(self, params):
        """
        Retrieve the temperature profile.

        Parameters:
        params (dict): 
            Parameters for the PT profile.

        Returns:
        array: 
            Temperature profile.
        """
        # Interpolate the 4D grid onto the requested point
        point = ()
        if len(self.T_eff_grid) > 1:
            point += (params['T_eff'], )
        if len(self.log_g_grid) > 1:
            point += (params['log_g'], )
        if len(self.CO_grid) > 1:
            point += (params['C/O'], )
        if len(self.FeH_grid) > 1:
            point += (params['Fe/H'], )

        self.temperature = self.temperature_interp_func(point)
        self.RCB = self.rad_conv_boundary_interp_func(point)

        return self.temperature
    
class PT_profile_free(PT_profile):
    """
    Class for free PT profiles.
    """

    def __init__(self, pressure, ln_L_penalty_order=3, PT_interp_mode='log', **kwargs):
        """
        Initialize the PT_profile_free class.

        Parameters:
        pressure (array): 
            Pressure levels.
        ln_L_penalty_order (int): 
            Order of the log-likelihood penalty.
        PT_interp_mode (str): 
            Interpolation mode for PT profile.
        **kwargs: 
            Additional keyword arguments.
        """
        # Give arguments to the parent class
        super().__init__(pressure)

        self.ln_L_penalty_order = ln_L_penalty_order
        self.PT_interp_mode = PT_interp_mode

    def __call__(self, params):
        """
        Retrieve the temperature profile.

        Parameters:
        params (dict): 
            Parameters for the PT profile.

        Returns:
        array: 
            Temperature profile.
        """
        # Combine all temperature knots
        self.T_knots = np.concatenate((params['T_knots'], 
                                       [params['T_0']]
                                       ))

        self.P_knots = params['P_knots']

        if self.P_knots is None:
            # Use evenly-spaced (in log) pressure knots
            self.P_knots = np.logspace(np.log10(self.pressure.min()), 
                                       np.log10(self.pressure.max()), 
                                       num=len(self.T_knots))

        # Log-likelihood penalty scaling factor
        self.gamma = params['gamma']

        # Cubic spline interpolation over all layers
        self.spline_interp()
        # Compute the log-likelihood penalty
        if self.gamma is not None:
            self.get_ln_L_penalty()
            
        return self.temperature

    def spline_interp(self):
        """
        Perform spline interpolation for the PT profile.
        """
        if self.PT_interp_mode == 'log':
            y = np.log10(self.T_knots)
        elif self.PT_interp_mode == 'lin':
            y = self.T_knots

        # Spline interpolation over a number of knots
        #spl = make_interp_spline(np.log10(self.P_knots), y, bc_type=([(3,0)],[(3,0)]))
        spl = make_interp_spline(np.log10(self.P_knots), y, bc_type='not-a-knot')

        if self.PT_interp_mode == 'log':
            self.temperature = 10**spl(np.log10(self.pressure))
        elif self.PT_interp_mode == 'lin':
            self.temperature = spl(np.log10(self.pressure))

        self.coeffs = spl.c
        self.knots  = spl.t

        # Remove padding zeros
        self.coeffs = self.coeffs[:len(self.knots)-4]

    def get_ln_L_penalty(self):
        """
        Compute the log-likelihood penalty.
        """
        # Do not apply a ln L penalty
        if self.ln_L_penalty_order == 0:
            self.ln_L_penalty = 0
            return

        # Compute the log-likelihood penalty based on the wiggliness
        # (Inverted) weight matrices, scaling the penalty of small/large segments
        inv_W_1 = np.diag(1/(1/3 * np.array([self.knots[i+3]-self.knots[i] \
                                             for i in range(1, len(self.knots)-4)]))
                          )
        inv_W_2 = np.diag(1/(1/2 * np.array([self.knots[i+2]-self.knots[i] \
                                             for i in range(2, len(self.knots)-4)]))
                          )
        inv_W_3 = np.diag(1/(1/1 * np.array([self.knots[i+1]-self.knots[i] \
                                             for i in range(3, len(self.knots)-4)]))
                          )

        # Fundamental difference matrix
        delta = np.zeros((len(inv_W_1), len(inv_W_1)+1))
        delta[:,:-1] += np.diag([-1]*len(inv_W_1))
        delta[:,1:]  += np.diag([+1]*len(inv_W_1))

        # 1st, 2nd, 3rd order general difference matrices
        D_1 = np.dot(inv_W_1, delta)
        D_2 = np.dot(inv_W_2, np.dot(delta[1:,1:], D_1))
        D_3 = np.dot(inv_W_3, np.dot(delta[2:,2:], D_2))

        # General difference penalty, computed with L2-norm
        if self.ln_L_penalty_order == 1:
            gen_diff_penalty = np.nansum(np.dot(D_1, self.coeffs)**2)
        elif self.ln_L_penalty_order == 2:
            gen_diff_penalty = np.nansum(np.dot(D_2, self.coeffs)**2)
        elif self.ln_L_penalty_order == 3:
            gen_diff_penalty = np.nansum(np.dot(D_3, self.coeffs)**2)

        self.ln_L_penalty = -(1/2*gen_diff_penalty/self.gamma + \
                              1/2*np.log(2*np.pi*self.gamma)
                              )

class PT_profile_free_gradient(PT_profile):
    """
    Class for free gradient PT profiles.
    """

    def __init__(self, pressure, PT_interp_mode='quadratic', **kwargs):
        """
        Initialize the PT_profile_free_gradient class.

        Parameters:
        pressure (array): 
            Pressure levels.
        PT_interp_mode (str): 
            Interpolation mode for PT profile.
        **kwargs: 
            Additional keyword arguments.
        """
        # Give arguments to the parent class
        super().__init__(pressure)

        self.flipped_ln_pressure = np.log(self.pressure)[::-1]

        self.PT_interp_mode = PT_interp_mode

    def __call__(self, params):
        """
        Retrieve the temperature profile.

        Parameters:
        params (dict): 
            Parameters for the PT profile.

        Returns:
        array: 
            Temperature profile.
        """
        self.P_knots = params['P_knots']

        # Perform interpolation over dlnT/dlnP gradients
        interp_func = interp1d(
            params['log_P_knots'], params['dlnT_dlnP_knots'], 
            kind=self.PT_interp_mode
            )
        dlnT_dlnP_array = interp_func(np.log10(self.pressure))[::-1]

        # Compute the temperatures relative to a base pressure
        P_base = params.get('P_phot', self.pressure.max())
        T_base = params.get('T_phot', params.get('T_0'))

        mask_above = (self.pressure[::-1] <= P_base)
        mask_below = (self.pressure[::-1] > P_base)

        ln_P_above      = self.flipped_ln_pressure[mask_above]
        dlnT_dlnP_above = dlnT_dlnP_array[mask_above]
        T_above = [T_base, ]
        for i, ln_P_up_i in enumerate(ln_P_above):

            if i == 0:
                # Use base knot initially
                ln_P_low_i = np.log(P_base)
                ln_T_low_i = np.log(T_base)
            else:
                # Use previous layer
                ln_P_low_i = ln_P_above[i-1]
                ln_T_low_i = np.log(T_above[-1])

            # Compute the temperatures based on the gradient
            ln_T_up_i = ln_T_low_i + (ln_P_up_i - ln_P_low_i)*dlnT_dlnP_above[i]
            T_above.append(np.exp(ln_T_up_i))

        self.temperature = np.array(T_above)[1:] # Remove the base knot

        if mask_below.any():
            
            # Compute the temperatures below the photospheric knot
            ln_P_below      = self.flipped_ln_pressure[mask_below][::-1]
            dlnT_dlnP_below = dlnT_dlnP_array[mask_below][::-1]
            T_below = [T_base, ]
            for i, ln_P_low_i in enumerate(ln_P_below):

                if i == 0:
                    # Use base knot initially
                    ln_P_up_i = np.log(P_base)
                    ln_T_up_i = np.log(T_base)
                else:
                    # Use previous layer
                    ln_P_up_i = ln_P_below[i-1]
                    ln_T_up_i = np.log(T_below[-1])

                # Compute the temperatures based on the gradient
                ln_T_low_i = ln_T_up_i - (ln_P_up_i - ln_P_low_i)*dlnT_dlnP_below[i]
                T_below.append(np.exp(ln_T_low_i))

            T_below = np.array(T_below)[1:] # Remove the base knot

            # Flip the below-array and combine with the above-layers
            self.temperature = np.concatenate((T_below[::-1], self.temperature))

        # Flip temperatures so high-altitude is at first indices
        self.temperature = self.temperature[::-1]

        return self.temperature