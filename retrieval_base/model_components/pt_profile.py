import numpy as np
from scipy.interpolate import interp1d

def get_class(PT_mode='free_gradient', **kwargs):
    """
    Get the PT profile class based on the mode.

    Args:
        PT_mode (str): Mode for the PT profile.
        **kwargs: Additional keyword arguments.

    Returns:
        PT_profile: Instance of the PT profile class.
    """
    if PT_mode == 'free_gradient':
        return PT_profile_free_gradient(**kwargs)
    elif PT_mode in ['static','constant']:
        return PT_profile(**kwargs)
    else:
        raise ValueError(f'PT mode {PT_mode} not recognized.')

class PT_profile:
    """
    Base class for PT profiles.
    """

    @staticmethod
    def get_dlnT_dlnP(temperature, pressure):
        """
        Get the temperature gradient with respect to pressure.

        Args:
            temperature (array): Temperature array.
            pressure (array): Pressure array.

        Returns:
            tuple: Temperature gradient and mean log pressure.
        """
        # Log pressure between layers
        log_pressure = np.log10(pressure)
        mean_log_pressure = 1/2*(log_pressure[1:] + log_pressure[:-1])

        # Temperature gradient
        dlnT_dlnP = np.diff(np.log(temperature)) / np.diff(np.log(pressure))

        return dlnT_dlnP, 10**mean_log_pressure
    
    def __init__(self, **kwargs):
        """
        Initialize the PT_profile class.

        Args:
            **kwargs: Additional keyword arguments.
        """
        # Set the pressure levels
        self._set_pressures(**kwargs)
        self.n_atm_layers = len(self.pressure)

        self.log_pressure = np.log10(self.pressure)

    def __call__(self, ParamTable):
        """
        Update the PT profile.

        Args:
            ParamTable (dict): Parameter table.
        """
        if ParamTable.get('temperature') is not None:
            # Get a constant temperature profile
            self.temperature = ParamTable.get('temperature')
        
        if (self.temperature < 0.).any():
            # Not a valid temperature profile
            return -np.inf
        
    def _set_pressures(self, **kwargs):
        """
        Set the pressure levels.

        Args:
            **kwargs: Additional keyword arguments.
        """
        self.pressure = kwargs.get('pressure')
        if self.pressure is not None:
            return
        
        log_P_range  = kwargs.get('log_P_range')
        n_atm_layers = kwargs.get('n_atm_layers')
        if None not in [log_P_range, n_atm_layers]:
            self.pressure = np.logspace(*log_P_range, n_atm_layers, dtype=float)
            self.pressure = np.sort(self.pressure)
            return

        raise ValueError('Could not set pressure levels.')
                    
class PT_profile_free_gradient(PT_profile):
    """
    Class for free gradient PT profiles.
    """
    def __init__(self, interp_mode='linear', symmetric_around_P_phot=False, n_knots=2, **kwargs):
        """
        Initialize the PT_profile_free_gradient class.

        Args:
            interp_mode (str): Interpolation mode.
            symmetric_around_P_phot (bool): Flag for symmetry around photospheric pressure.
            n_knots (int): Number of knots.
            **kwargs: Additional keyword arguments.
        """
        # Give arguments to the parent class
        super().__init__(**kwargs)

        self.ln_pressure = np.log(self.pressure)
        self.flipped_ln_pressure = self.ln_pressure[::-1]

        self.interp_mode = interp_mode
        self.symmetric_around_P_phot = symmetric_around_P_phot
        self.n_knots = n_knots

    def __call__(self, ParamTable):
        """
        Set the parameters for the PT profile.

        Args:
            ParamTable (dict): Parameter table.
        """
        self._set_pressure_knots(ParamTable)
        self.P_knots     = 10**self.log_P_knots
        self.ln_P_knots  = np.log(self.P_knots)
        
        self.n_knots = len(self.P_knots)

        self._set_temperature_gradients(ParamTable)
        self._get_temperature(ParamTable)

        super().__call__(ParamTable)
        
    def _set_pressure_knots(self, ParamTable):
        """
        Set the pressure knots for the PT profile.

        Args:
            ParamTable (dict): Parameter table.
        """
        self.log_P_knots = ParamTable.get('log_P_knots')
        if self.log_P_knots is not None:
            # Constant knots
            self.log_P_knots = np.sort(self.log_P_knots)
            return
        #n_knots = ParamTable.get('n_knots', 2)

        # Equally-spaced knots
        self.log_P_knots = np.linspace(
            self.log_pressure.min(), self.log_pressure.max(), self.n_knots
            )
        
        log_P_base = ParamTable.get('log_P_phot')
        if log_P_base is not None:
            # Relative to photospheric knot
            self.log_P_knots = [self.log_P_knots[0], log_P_base, self.log_P_knots[-1]]

        if (log_P_base is None) and (ParamTable.get('d_log_P_0+1') is not None):
            # Relative to base of atmosphere
            log_P_base = self.log_pressure.max()
            self.log_P_knots = [self.log_P_knots[0], self.log_P_knots[-1]]

        if log_P_base is None:
            return
        
        for i in range(1, self.n_knots):
            # Upper and lower separations
            up_i = ParamTable.get(f'd_log_P_phot+{i}')
            if up_i is None:
                up_i = ParamTable.get(f'd_log_P_0+{i}')

            low_i = ParamTable.get(f'd_log_P_phot-{i}')
            if (up_i is None) and (low_i is None):
                break
            
            if self.symmetric_around_P_phot:
                low_i = up_i

            if up_i is not None:
                self.log_P_knots.append(log_P_base-up_i)
            if low_i is not None:
                self.log_P_knots.append(log_P_base+low_i)

        # Ascending pressure
        self.log_P_knots = np.sort(np.array(self.log_P_knots))

    def _set_temperature_gradients(self, ParamTable):
        """
        Get the temperature gradients for the PT profile.

        Args:
            ParamTable (dict): Parameter table.
        """
        # Get the temperature gradients at each knot
        self.dlnT_dlnP_knots = np.array([
            ParamTable.get(f'dlnT_dlnP_{i}') for i in range(self.n_knots)
            ])
        # Ascending in pressure
        self.dlnT_dlnP_knots = self.dlnT_dlnP_knots[::-1]

        # Interpolate onto each pressure level
        interp_func = interp1d(
            self.log_P_knots, self.dlnT_dlnP_knots, kind=self.interp_mode
            )
        # Ascending in pressure
        self.dlnT_dlnP = interp_func(self.log_pressure)

    def _get_temperature(self, ParamTable):
        """
        Get the temperature profile.

        Args:
            ParamTable (dict): Parameter table.
        """
        # Compute the temperatures relative to a base pressure
        P_base = ParamTable.get('P_phot', self.pressure.max())
        T_base = ParamTable.get('T_phot', ParamTable.get('T_0'))

        # Mask for above and below the base pressure
        mask_above = (self.pressure <= P_base)
        mask_below = (self.pressure > P_base)

        self.temperature = np.zeros_like(self.pressure)
        for mask in [mask_above, mask_below]:
            if not mask.any():
                continue

            #dlnT_dlnP = self.dlnT_dlnP[::-1][mask]
            dlnT_dlnP = self.dlnT_dlnP[mask]
            ln_P = np.log(self.pressure)[mask]
            
            # Sort relative to base pressure
            idx  = np.argsort(np.abs(ln_P-np.log(P_base)))
            sorted_ln_P = ln_P[idx]
            sorted_dlnT_dlnP = dlnT_dlnP[idx]

            ln_T = []
            for i, (ln_P_i, dlnT_dlnP_i) in enumerate(zip(sorted_ln_P, sorted_dlnT_dlnP)):

                if i==0:
                    # Compare to the base pressure
                    dln_P_i = ln_P_i - np.log(P_base)
                    ln_T_previous = np.log(T_base)
                else:
                    # Compare to the previous pressure level
                    dln_P_i = ln_P_i - sorted_ln_P[i-1]
                    ln_T_previous = ln_T[-1]

                # T_j = T_{j-1} * (P_j/P_{j-1})^dlnT_dlnP_j
                ln_T_i = ln_T_previous + dln_P_i*dlnT_dlnP_i
                ln_T.append(ln_T_i)

            # Sort by ascending pressure
            idx = np.argsort(sorted_ln_P)
            self.temperature[mask] = np.exp(np.array(ln_T)[idx])