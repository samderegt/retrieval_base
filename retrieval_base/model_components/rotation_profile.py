import numpy as np
from scipy.interpolate import interp1d

from ..utils import sc

def get_class(rotation_mode='convolve', **kwargs):
    """
    Get the rotation profile class based on the mode.

    Args:
        rotation_mode (str): Mode for the rotation profile.
        **kwargs: Additional keyword arguments.

    Returns:
        RotationProfile: Instance of the rotation profile class.
    """
    if rotation_mode == 'convolve':
        return ConvolveRotationProfile(**kwargs)
    elif rotation_mode == 'integrate':
        return IntegrateRotationProfile(**kwargs)
    else:
        raise ValueError(f'Rotation profile mode "{rotation_mode}" not recognized.')
    
class ConvolveRotationProfile:
    """
    Class for convolve rotation profile.
    """
    def __init__(self, **kwargs):
        """
        Initialize the ConvolveRotationProfile class.

        Args:
            **kwargs: Additional keyword arguments.
        """
        from PyAstronomy.pyasl import fastRotBroad
        self.broad_func = fastRotBroad

    def __call__(self, ParamTable, **kwargs):
        """
        Set the parameters for the rotation profile.

        Args:
            ParamTable (dict): Parameter table.
            **kwargs: Additional keyword arguments.
        """
        self.vsini        = ParamTable.get('vsini', 0.)
        self.epsilon_limb = ParamTable.get('epsilon_limb', 0.)

    def broaden(self, wave, flux, **kwargs):
        """
        Apply rotational broadening to the flux.

        Args:
            wave (array): Wavelength array.
            flux (array): Flux array.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: Broadened wavelength and flux arrays.
        """
        wave = wave[0]
        flux = flux[0]
        if (self.vsini == 0.) or (self.epsilon_limb == 0.):
            return wave, flux # Avoid interpolation
        
        # Evenly spaced wavelength grid
        wave_even = np.linspace(wave.min(), wave.max(), len(wave))
        flux_even = np.interp(wave_even, xp=wave, fp=flux)

        # Rotational broadening
        flux_rot_broad = self.broad_func(
            wave_even, flux_even, vsini=self.vsini, 
            epsilon=self.epsilon_limb
            )
        return wave_even, flux_rot_broad
    
class SurfaceMap:
    """
    Class for surface map.
    """
    def __init__(self, inclination=0, lon_0=0, n_mu=3, n_c=None, n_theta=150, is_inside_patch=False, is_outside_patch=False, **kwargs):
        """
        Initialize the SurfaceMap class.

        Args:
            inclination (float): Inclination angle in degrees.
            lon_0 (float): Longitude in degrees.
            n_mu (int): Number of incidence angle segments. If given, use Gaussian quadrature.
            n_c (int): Number of angular distance segments.
            n_theta (int): Number of angular segments.
            is_inside_patch (bool): Flag for inside patch.
            is_outside_patch (bool): Flag for outside patch.
            **kwargs: Additional keyword arguments.
        """
        self.inclination = np.deg2rad(inclination)
        self.lon_0 = np.deg2rad(lon_0)

        # Define grid to integrate over
        self._set_coords(n_mu, n_c, n_theta)

        # Which segments to include for this model setting
        self.is_inside_patch  = is_inside_patch
        self.is_outside_patch = is_outside_patch

    def __call__(self, ParamTable, **kwargs):
        """
        Set the parameters for the surface map.

        Args:
            ParamTable (dict): Parameter table.
            **kwargs: Additional keyword arguments.
        """
        # Brightness and velocity maps
        self._get_brightness_and_velocity_maps(ParamTable)
        
        # Get the segments included in the current patch
        self._get_included_segments()

    def _set_coords(self, n_mu, n_c, n_theta):
        """
        Set the coordinates for the surface map.

        Args:
            n_mu (int): Number of incidence angle segments.
            n_c (int): Number of angular distance segments.
            n_theta (int): Number of angular segments.
        """
        if n_mu is not None:
            # Gaussian quadrature over incidence angles
            #self.integration_method = 'quadrature'
            x, w = np.polynomial.legendre.leggauss(n_mu)
            self.unique_mu = (1-0)/2*x + (1+0)/2 # Change integration limits
            self.unique_dmu = (1-0)/2*w

            # Angular distance grid
            self.unique_c = np.arccos(self.unique_mu)

        elif n_c is not None:
            # Trapezoidal rule over angular distance
            #self.integration_method = 'trapezoidal'
            dc = (np.pi/2)/n_c
            self.unique_c = np.arange(dc/2, np.pi/2, dc)
            
            # Incidence angle grid
            self.unique_mu = np.cos(self.unique_c)
            self.unique_dmu = np.sin(self.unique_c)*dc

        else:
            raise ValueError('Either n_mu or n_c must be given.')
        
        # Radial distance grid
        self.unique_r = np.sqrt(1-self.unique_mu**2)

        self.r_grid, self.theta_grid = [], []
        self.c_grid, self.mu_grid, self.dmu_grid = [], [], []
        self.weight_grid = []
        for r_i, c_i, mu_i, dmu_i in zip(self.unique_r, self.unique_c, self.unique_mu, self.unique_dmu):
            # Reduce number of angular segments close to centre
            n_theta_i = int(np.ceil(n_theta*r_i))
            d_theta_i = (2*np.pi)/n_theta_i

            # Angular coordinates
            theta_i = np.arange(d_theta_i/2, 2*np.pi, d_theta_i)
            self.theta_grid.append(theta_i)

            # Integration weight over azimuthal angle and incidence angle
            weight_i = d_theta_i * mu_i * dmu_i
            self.weight_grid.append(weight_i*np.ones_like(theta_i))
            
            self.r_grid.append(r_i*np.ones_like(theta_i))
            self.c_grid.append(c_i*np.ones_like(theta_i))
            self.mu_grid.append(mu_i*np.ones_like(theta_i))
            self.dmu_grid.append(dmu_i*np.ones_like(theta_i))

        # Coordinate for each segment
        self.theta_grid = np.concatenate(self.theta_grid)

        self.r_grid  = np.concatenate(self.r_grid)
        self.c_grid  = np.concatenate(self.c_grid)
        self.mu_grid = np.concatenate(self.mu_grid)
        self.dmu_grid = np.concatenate(self.dmu_grid)

        self.weight_grid = np.concatenate(self.weight_grid)
        # self.integrated_area = np.sum(self.weight_grid)

        self._set_cartesian_coords()
        self._set_latlon_coords()

    def _set_cartesian_coords(self):
        """
        Set the Cartesian coordinates for the surface map.
        """
        # 2D Cartesian coordinates
        self.x_grid = self.r_grid * np.sin(self.theta_grid)
        self.y_grid = self.r_grid * np.cos(self.theta_grid)

    def _set_latlon_coords(self):
        """
        Set the latitude and longitude coordinates for the surface map.
        """
        # Latitude and longitude from orthographic projection
        self.lat_grid = np.arcsin(
            np.cos(self.c_grid)*np.sin(self.inclination) + \
                np.cos(self.theta_grid)*np.sin(self.c_grid)*np.cos(self.inclination)
        )
        self.lon_grid = self.lon_0 + np.arctan2(
            self.x_grid*np.sin(self.c_grid), 
            self.r_grid * np.cos(self.c_grid)*np.cos(self.inclination) - \
                self.y_grid * np.sin(self.c_grid)*np.sin(self.inclination)
            )
        
    def _add_zonal_band(self, ParamTable, suffix=''):
        """
        Get a zonal band for the surface map.

        Args:
            ParamTable (dict): Parameter table.
            suffix (str): Suffix for the parameters.

        Returns:
            array: Mask for the zonal band.
        """
        # Band parameters
        lat_band_cen   = ParamTable.get(f'lat_band_cen{suffix}', 0.) # Equator by default
        lat_band_width = ParamTable.get(f'lat_band_width{suffix}')
        band_contrast  = ParamTable.get(f'band_contrast{suffix}', 1.)
        band_velocity  = ParamTable.get(f'band_velocity{suffix}', self.vsini)

        if None in [lat_band_width]:
            return np.zeros_like(self.r_grid, dtype=bool)

        # Mask for latitude band
        lat_band_cen   = np.deg2rad(lat_band_cen)
        lat_band_width = np.deg2rad(lat_band_width)

        lat_band_min = lat_band_cen - lat_band_width/2
        lat_band_max = lat_band_cen + lat_band_width/2

        lat_band_mask = (self.lat_grid >= lat_band_min) & (self.lat_grid <= lat_band_max)

        # Apply contrast to brightness map
        self.brightness[lat_band_mask] *= band_contrast

        # Differential rotation
        self.velocity[lat_band_mask] *= band_velocity/self.vsini

        return lat_band_mask
    
    def _add_latlon_circular_spot(self, ParamTable, suffix=''):
        """
        Get a circular spot for the surface map.

        Args:
            ParamTable (dict): Parameter table.
            suffix (str): Suffix for the parameters.

        Returns:
            array: Mask for the circular spot.
        """
        # Spot parameters
        lat_spot = ParamTable.get(f'lat_spot{suffix}')
        lon_spot = ParamTable.get(f'lon_spot{suffix}')
        spot_radius   = ParamTable.get(f'spot_radius{suffix}')
        spot_contrast = ParamTable.get(f'spot_contrast{suffix}', 1.)
        
        if None in [lat_spot, lon_spot, spot_radius]:
            return np.zeros_like(self.r_grid, dtype=bool)

        # Mask for circular spot
        lat_spot = np.deg2rad(lat_spot)
        lon_spot = np.deg2rad(lon_spot)
        spot_radius = np.deg2rad(spot_radius)

        # Haversine formula
        distance_from_spot = 2 * np.arcsin((1/2*(
            1 - np.cos(self.lat_grid - lat_spot) + \
            np.cos(lat_spot)*np.cos(self.lat_grid)*(1-np.cos(self.lon_grid-lon_spot))
            ))**(1/2))

        spot_mask = (distance_from_spot <= spot_radius)

        # Apply contrast to brightness map
        self.brightness[spot_mask] *= spot_contrast

        return spot_mask
    
    def _get_brightness_and_velocity_maps(self, ParamTable, n_max_features=5):
        """
        Get the brightness and velocity maps for the surface map.

        Args:
            ParamTable (dict): Parameter table.
            n_max_features (int): Maximum number of features.
        """
        # Set default brightness map
        self.brightness = np.ones_like(self.r_grid)
        
        # Set default velocity map
        self.vsini    = ParamTable.get('vsini', 0.)
        self.velocity = self.vsini * self.r_grid * np.sin(self.theta_grid)

        self.patch_mask = np.zeros_like(self.r_grid, dtype=bool)

        for idx in [None, *range(n_max_features)]:
            # Add multiple spots or bands
            suffix = f'_{idx+1}' if idx is not None else ''

            spot_mask = self._add_latlon_circular_spot(ParamTable, suffix=suffix)
            band_mask = self._add_zonal_band(ParamTable, suffix=suffix)

            self.patch_mask[spot_mask|band_mask] = True

        self.integrated_brightness = np.sum(self.brightness)

    def _get_included_segments(self):
        """
        Get the included segments for the surface map.
        """
        # Include all segments by default
        self.included_segments = np.ones_like(self.r_grid, dtype=bool)

        if self.is_inside_patch:
            # Ignore segments outside of the patches
            self.included_segments[~self.patch_mask] = False
        if self.is_outside_patch:
            # Ignore segments inside of the patches
            self.included_segments[self.patch_mask] = False

        # Update the incidence angles included in current patch
        self.unique_mu_included, indices = \
            np.unique(self.mu_grid[self.included_segments], return_index=True)
        self.unique_dmu_included = self.dmu_grid[indices]

class IntegrateRotationProfile(SurfaceMap):
    """
    Class for integrate rotation profile.
    """
    def __init__(self, **kwargs):
        """
        Initialize the IntegrateRotationProfile class.

        Args:
            **kwargs: Additional keyword arguments.
        """
        # Initialize the surface map
        super().__init__(**kwargs)

    def broaden(self, wave, flux, get_integrated_flux=False, **kwargs):
        """
        Apply rotational broadening to the flux.

        Args:
            wave (array): Wavelength array.
            flux (array): Flux array.
            get_integrated_flux (bool): Flag to get integrated flux.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: Broadened wavelength and flux arrays.
        """
        self.get_integrated_flux = get_integrated_flux

        # Store global flux, integrated over the surface
        flux_rot_broad = np.zeros_like(wave[0])

        # Integrate over all incidence angles
        for mu, flux_mu in zip(self.unique_mu_included, flux):
            
            # Integrate over all velocities in annulus
            flux_rot_broad += self._integrate_over_velocity(wave[0], flux_mu, mu)

        # Normalise to account for the number of segments
        flux_rot_broad *= len(self.brightness) / self.integrated_brightness

        return wave[0], flux_rot_broad
    
    def _integrate_over_wavelength(self, wave, flux_shifted, idx_segments):
        """
        Integrate the flux over the wavelength.

        Args:
            wave (array): Wavelength array.
            flux_shifted (array): Shifted flux array.
            idx_segments (array): Index segments array.
        """
        if not hasattr(self, 'integrated_flux'):
            self.integrated_flux = np.nan * np.ones_like(self.r_grid)

        for i, idx in enumerate(idx_segments):
            
            flux_i = flux_shifted[i]
            
            mask = np.isfinite(flux_i)
            wave_i = wave[mask]
            flux_i = flux_i[mask]

            # Integrate over wavelengths
            self.integrated_flux[idx] = np.trapz(x=wave_i, y=flux_i) / np.trapz(x=wave_i, y=wave_i)

    def _integrate_over_velocity(self, wave, flux_mu, mu):
        """
        Integrate the flux over the velocity.

        Args:
            wave (array): Wavelength array.
            flux_mu (array): Flux array for each incidence angle.
            mu (float): Incidence angle.

        Returns:
            array: Integrated flux array.
        """
        # Included in current patch + matching incidence angle
        mask = self.included_segments & (self.mu_grid == mu)
        idx_segments = np.arange(len(self.r_grid))[mask]
        
        # Expand over all velocities
        interp_func = interp1d(
            wave, flux_mu, kind='linear', bounds_error=False, fill_value=np.nan
            )
        velocity = self.velocity[idx_segments]
        wave_shifted = wave[None,:] * (1 - velocity[:,None]/(sc.c*1e-3))

        # Interpolate flux at shifted wavelengths
        flux_shifted = interp_func(wave_shifted)

        # Apply brightness map
        flux_shifted *= self.brightness[idx_segments][:,None]

        # Apply area weighting
        flux_shifted *= self.weight_grid[idx_segments][:,None]

        if self.get_integrated_flux:
            self._integrate_over_wavelength(wave, flux_shifted, idx_segments)

        # Integrate over all velocities
        integrated_flux_mu = np.sum(flux_shifted, axis=0)

        return integrated_flux_mu