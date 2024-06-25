import numpy as np
import petitRADTRANS.nat_cst as nc

def get_Rotation_class(mode='convolve', **kwargs):
    """
    Function to create a rotation profile based on the mode.

    Args:
    - mode (str): Either 'integrate' or 'convolve' to choose the type of rotation profile.
    - **kwargs: Additional keyword arguments passed to the rotation profile classes.

    Returns:
    - Instance of IntRotationProfile or ConvRotationProfile based on the mode.
    """
    if mode is None:
        return ConvRotationProfile(**kwargs)
    if mode == 'convolve':
        return ConvRotationProfile(**kwargs)
    if mode == 'integrate':
        return IntRotationProfile(**kwargs)

class ConvRotationProfile:
    """
    Class for convolutional rotation profile.

    Attributes:
    - fastRotBroad: Fast rotational broadening function from PyAstronomy.
    """
    def __init__(self, **kwargs):
        """
        Initializes ConvRotationProfile.

        Args:
        - **kwargs: Additional keyword arguments.
        """
        from PyAstronomy import pyasl
        self.fastRotBroad = pyasl.fastRotBroad

    def __call__(self, wave, flux, params, **kwargs):
        """
        Applies rotational broadening to a given spectrum.

        Args:
        - wave (numpy.ndarray): Wavelength grid.
        - flux (numpy.ndarray): Flux values corresponding to the wavelength grid.
        - params (dict): Parameters for rotational broadening.
        - **kwargs: Additional keyword arguments.

        Returns:
        - wave_even (numpy.ndarray): Evenly spaced wavelength grid.
        - flux_rot_broad (numpy.ndarray): Rotational broadened flux.
        """
        # Evenly space the wavelength grid
        wave_even = np.linspace(wave.min(), wave.max(), wave.size)
        flux_even = np.interp(wave_even, xp=wave, fp=flux)

        epsilon_limb = params.get('epsilon_limb', 0)
        vsini = params.get('vsini', 0)

        if (vsini == 0) or (epsilon_limb == 0):
            return wave_even, flux_even

        # Rotational broadening of the model spectrum
        flux_rot_broad = self.fastRotBroad(
            wave_even, flux_even, 
            epsilon=epsilon_limb, vsini=vsini
            )
        
        return wave_even, flux_rot_broad

class IntRotationProfile:
    """
    Class for integrated rotation profile.

    Attributes:
    - inc (float): Inclination angle in radians.
    - lon_0 (float): Longitude angle in radians.
    - n_c (int): Number of longitudinal grid points.
    - n_theta (int): Number of latitudinal grid points.
    - Various grid-related attributes for integration.
    """
    def __init__(self, inc=0, lon_0=0, n_c=15, n_theta=150, **kwargs):
        """
        Initializes IntRotationProfile.

        Args:
        - inc (float): Inclination angle in degrees.
        - lon_0 (float): Longitude angle in degrees.
        - n_c (int): Number of longitudinal grid points.
        - n_theta (int): Number of latitudinal grid points.
        - **kwargs: Additional keyword arguments.
        """
        # (Partially) adopted from Carvalho & Johns-Krull (2023)
        self.inc   = np.deg2rad(inc)
        self.lon_0 = np.deg2rad(lon_0)
        
        # Define grid to integrate over
        self.n_c     = n_c
        self.n_theta = n_theta

        # Define the grid with the regular spacing
        self.get_coords()

    def get_coords(self, sampling_factor=1):
        """
        Computes coordinates on a grid for integration.

        Args:
        - sampling_factor (float): Factor to adjust grid sampling density.
        """
        # Compute coordinates on a finer grid
        n_c     = self.n_c * int(sampling_factor)
        n_theta = self.n_theta * int(sampling_factor)

        # Equidistant grid in angular distance
        self.dc = (np.pi/2) / n_c
        self.unique_c = np.arange(self.dc/2, np.pi/2, self.dc)

        self.unique_mu = np.cos(self.unique_c)
        self.unique_r  = np.sqrt(1-self.unique_mu**2)

        self.unique_w_gauss_mu = np.ones_like(self.unique_mu)
        self.unique_w_gauss_mu /= np.sum(self.unique_w_gauss_mu)
        
        self.mu_grid, self.w_gauss_mu = [], []
        self.r_grid, self.theta_grid  = [], []
        self.area_per_segment = []

        # Radial grid spacing
        for i, mu_i in enumerate(self.unique_mu):

            # Corresponding radius
            r_i = self.unique_r[i]
            c_i = self.unique_c[i]

            # Reduce number of angular segments close to centre
            n_theta_r_i = int(n_theta*r_i)
                
            # Projected area
            area_ij = 2*np.pi / n_theta_r_i
            area_ij *= mu_i * np.sin(c_i) * self.dc

            for j in range(n_theta_r_i):
                th_ij = (np.pi + j*2*np.pi)/n_theta_r_i
                
                self.mu_grid.append(mu_i)
                self.w_gauss_mu.append(self.unique_w_gauss_mu[i])

                self.r_grid.append(r_i)
                self.theta_grid.append(th_ij)

                self.area_per_segment.append(area_ij)

        self.mu_grid    = np.array(self.mu_grid)
        self.w_gauss_mu = np.array(self.w_gauss_mu)

        self.r_grid     = np.array(self.r_grid)
        self.theta_grid = np.array(self.theta_grid)
        
        # Normalize the projected area
        self.area_per_segment = np.array(self.area_per_segment)
        self.disk_area_tot    = np.sum(self.area_per_segment)

        self.c_grid = np.arccos(self.mu_grid) # Angular distance
        
        self.idx_mu = np.zeros_like(self.mu_grid, dtype=int)
        for unique_i, unique_mu_i in enumerate(self.unique_mu):
            self.idx_mu[(self.mu_grid==unique_mu_i)] = unique_i

        # 2D cartesian coordinates
        self.x_grid = self.r_grid * np.sin(self.theta_grid)
        self.y_grid = self.r_grid * np.cos(self.theta_grid)

        # Get the latitude and longitude coordinates
        self.get_latlon()

    def get_latlon(self):
        """
        Computes latitude and longitude coordinates for the grid.
        """
        if not hasattr(self, 'x_grid'):
            # 2D cartesian coordinates
            self.x_grid = self.r_grid * np.sin(self.theta_grid)
            self.y_grid = self.r_grid * np.cos(self.theta_grid)
        
        # Latitudes + longitudes
        self.lat_grid = np.arcsin(
            np.cos(self.c_grid)*np.sin(self.inc) + \
            np.cos(self.theta_grid)*np.sin(self.c_grid)*np.cos(self.inc)
            )
        self.lon_grid = self.lon_0 + np.arctan2(
            self.x_grid*np.sin(self.c_grid), 
            self.r_grid * np.cos(self.c_grid)*np.cos(self.inc) - \
                self.y_grid * np.sin(self.c_grid)*np.sin(self.inc)
            )
    
    def _add_longitudinal_band(self, params, band_suffix=''):
        """
        Adds a longitudinal band of brightness scaling.

        Args:
        - params (dict): Parameters for the band.
        - band_suffix (str): Suffix for distinguishing multiple bands.
        """
        lon_band_upper = params.get(f'lon_band_upper{band_suffix}')
        lon_band_lower = params.get(f'lon_band_lower{band_suffix}')
        
        upper_and_lower = (lon_band_upper is not None) & (lon_band_lower is not None)
        if not upper_and_lower:

            if params.get('lon_band_1') is None:
                # Only a single band
                lon_band_cen = np.deg2rad(params.get('lon_band_0', 0))
            else:
                # Multiple bands
                lon_band_cen = np.deg2rad(params.get(f'lon_band_cen{band_suffix}', 0))

            # Band width
            lon_band = np.deg2rad(params.get(f'lon_band{band_suffix}'))

            lon_band_upper = lon_band_cen + lon_band
            lon_band_lower = lon_band_cen - lon_band
        else:
            lon_band_upper = np.deg2rad(lon_band_upper)
            lon_band_lower = np.deg2rad(lon_band_lower)

        # Brightness-scaling factor
        epsilon_band = params.get(f'epsilon_band{band_suffix}', 1)

        # Add a band at some longitude
        mask_band = (self.lon_grid < lon_band_upper) & (self.lon_grid > lon_band_lower)
    
        if epsilon_band < 0:
            epsilon_band = 1 + epsilon_band

        # Change the flux between some latitudes
        if self.relative_scaling:
            self.brightness[mask_band] *= epsilon_band
        else:
            self.brightness[mask_band] = epsilon_band

        if params.get('is_in_band'):
            # Ignore segments outside of band
            self.included_segments[~mask_band] = False
        if params.get('is_not_in_band'):
            # Ignore segments inside band 
            self.included_segments[mask_band] = False

    def _add_latitudinal_band(self, params, band_suffix=''):
        """
        Adds a latitudinal band of brightness scaling.

        Args:
        - params (dict): Parameters for the band.
        - band_suffix (str): Suffix for distinguishing multiple bands.
        """
        lat_band_upper = params.get(f'lat_band_upper{band_suffix}')
        lat_band_lower = params.get(f'lat_band_lower{band_suffix}')
        
        upper_and_lower = (lat_band_upper is not None) & (lat_band_lower is not None)
        if not upper_and_lower:
            # Central latitude of the band
            lat_band_cen = np.deg2rad(params.get(f'lat_band_cen{band_suffix}', 0))

            # Band width/height
            lat_band = np.deg2rad(params.get(f'lat_band{band_suffix}'))

            lat_band_upper = lat_band_cen + lat_band
            lat_band_lower = lat_band_cen - lat_band
        else:
            # Define upper and lower edges directly
            lat_band_upper = np.deg2rad(lat_band_upper)
            lat_band_lower = np.deg2rad(lat_band_lower)

        # Brightness-scaling factor
        epsilon_band = params.get(f'epsilon_band{band_suffix}', 1)

        # Add a band at some latitude
        mask_band = (self.lat_grid < lat_band_upper) & (self.lat_grid > lat_band_lower)
    
        if epsilon_band < 0:
            epsilon_band = 1 + epsilon_band

        # Change the flux between some latitudes
        if self.relative_scaling:
            self.brightness[mask_band] *= epsilon_band
        else:
            self.brightness[mask_band] = epsilon_band

        if params.get('is_in_band'):
            # Ignore segments outside of band
            self.included_segments[~mask_band] = False
        if params.get('is_not_in_band'):
            # Ignore segments inside band 
            self.included_segments[mask_band] = False

        vsini_band = params.get(f'vsini_band{band_suffix}')
        if vsini_band is not None:
            # Differential velocities
            self.v_grid[mask_band] = vsini_band * self.r_grid * np.sin(self.theta_grid)

    def _add_projected_spot(self, params, spot_suffix=''):
        """
        Adds a projected spot on the grid.

        Args:
        - params (dict): Parameters for the spot.
        - spot_suffix (str): Suffix for distinguishing multiple spots.
        """
        # Add a spot in polar/Cartesian coordinates
        r_spot      = params.get(f'r_spot{spot_suffix}')
        theta_spot  = np.deg2rad(params.get(f'theta_spot{spot_suffix}'))
        radius_spot = params.get(f'radius_spot{spot_suffix}')

        epsilon_spot = params.get(f'epsilon_spot{spot_suffix}', 1)

        # Ensure that entire spot is on the disk
        r_spot = (1 - radius_spot) * r_spot

        # Cartesian coordinates
        x_spot = r_spot * np.sin(theta_spot)
        y_spot = r_spot * np.cos(theta_spot)

        distance_from_spot = (
            (self.x_grid - x_spot)**2 + (self.y_grid - y_spot)**2
            )**(1/2)
        mask_spot = (distance_from_spot <= radius_spot)

        # Change the flux at the spot
        if self.relative_scaling:
            self.brightness[mask_spot] *= epsilon_spot
        else:
            self.brightness[mask_spot] = epsilon_spot

        if params.get(f'is_in_spot{spot_suffix}') or params.get('is_within_patch'):
            # Ignore segments outside of spot
            self.included_segments[mask_spot]  = True
            self.included_segments[~mask_spot] = False
        if params.get(f'is_not_in_spot{spot_suffix}') or params.get('is_outside_patch'):
            # Ignore segments inside spot
            self.included_segments[mask_spot] = False

    def _add_latlon_spot(self, params, spot_suffix=''):
        """
        Adds a spot defined by latitude and longitude.

        Args:
        - params (dict): Parameters for the spot.
        - spot_suffix (str): Suffix for distinguishing multiple spots.
        """
        # Revert to a single spot
        epsilon_spot = params.get(f'epsilon_spot{spot_suffix}', 1)

        lat_spot = np.deg2rad(params.get(f'lat_spot{spot_suffix}'))
        lon_spot = np.deg2rad(params.get(f'lon_spot{spot_suffix}'))

        if params.get(f'radius_spot{spot_suffix}') is not None:
            # Add a circular spot
            radius_spot = np.deg2rad(radius_spot)

            # Haversine formula
            distance_from_spot = 2 * np.arcsin((1/2*(
                1 - np.cos(self.lat_grid - lat_spot) + \
                np.cos(lat_spot)*np.cos(self.lat_grid)*(1-np.cos(self.lon_grid-lon_spot))
                ))**(1/2))
            mask_spot = (distance_from_spot <= radius_spot)

        else:
            # Add an elliptical spot
            a_spot = np.deg2rad(params.get(f'a_spot{spot_suffix}'))
            b_spot = np.deg2rad(params.get(f'b_spot{spot_suffix}'))

            # Use simple Cartesian? coordinates, results in changing
            # spot-size with changing latitude
            distance_lon = np.abs(self.lon_grid - lon_spot)
            # Make longitude-distance continuous
            mask_invalid = (distance_lon > np.pi)
            distance_lon[mask_invalid] = np.pi - distance_lon[mask_invalid] % np.pi

            # Latitude-distance
            distance_lat = np.abs(self.lat_grid - lat_spot)

            # Ellipse equation
            mask_spot = (
                (distance_lon/a_spot)**2 + (distance_lat/b_spot)**2 <= 1
                )

        # Scale the brightness
        if self.relative_scaling:
            self.brightness[mask_spot] *= epsilon_spot
        else:
            self.brightness[mask_spot] = epsilon_spot

        if params.get(f'is_in_spot{spot_suffix}') or params.get('is_within_patch'):
            # Ignore segments outside of spot
            self.included_segments[mask_spot]  = True
            self.included_segments[~mask_spot] = False
        if params.get(f'is_not_in_spot{spot_suffix}') or params.get('is_outside_patch'):
            # Ignore segments inside spot
            self.included_segments[mask_spot] = False

    def get_brightness(self, params, max_features=5):
        """
        Computes the brightness distribution over the grid.

        Args:
        - params (dict): Parameters for brightness computation.
        - max_features (int): Maximum number of features (spots or bands) to add.
        """
        # Compute velocity-grid
        vsini = params.get('vsini', 0)
        self.v_grid = vsini * self.r_grid * np.sin(self.theta_grid)
        # Scale by the inclination, turns vsini into v_eq
        #self.v_grid *= np.abs(np.sin(np.deg2rad(90)-self.inc))

        # Compute brightness-grid
        self.relative_scaling  = params.get('relative_scaling', True)
        
        self.brightness        = np.ones_like(self.r_grid)
        self.included_segments = np.ones_like(self.r_grid, dtype=bool)

        if params.get('epsilon_limb') is not None:
            # Linear limb-darkening for integrated flux
            epsilon_limb = params.get('epsilon_limb')
            self.brightness *= (
                1 - epsilon_limb + epsilon_limb*np.sqrt(1-self.r_grid**2)
                )

        if params.get('epsilon_lat') is not None:
            # Latitude-darkening
            epsilon_lat = params.get('epsilon_lat')
            self.brightness *= (1 - epsilon_lat * np.sin(self.lat_grid)**2)

        if (params.get('lon_band') is not None) or (params.get('lon_band_upper') is not None):
            # Add a band at some longitude
            self._add_longitudinal_band(params)

        if (params.get('lat_band') is not None) or (params.get('lat_band_upper') is not None):
            # Add a band at some latitude
            self._add_latitudinal_band(params)
            
        if (params.get('r_spot') is not None) and (params.get('theta_spot') is not None):
            # Add a spot in the projected coordinates
            self._add_projected_spot(params)

        if (params.get('lat_spot') is not None) and (params.get('lon_spot') is not None):
            # Add a spot in the latitude/longitude coordinates
            self._add_latlon_spot(params)

        for idx in range(max_features):
            
            # Add multiple spots or bands
            suffix = f'_{idx}'

            if (params.get(f'lat_band{suffix}') is not None) or \
                (params.get(f'lat_band_upper{suffix}') is not None):
                # Add a band at some latitude
                self._add_latitudinal_band(params, band_suffix=suffix)

            if (params.get(f'r_spot{suffix}') is not None) and \
                (params.get(f'theta_spot{suffix}') is not None):
                # Add a spot in the projected coordinates
                self._add_projected_spot(params, spot_suffix=suffix)

            if (params.get(f'lat_spot{suffix}') is not None) and \
                (params.get(f'lon_spot{suffix}') is not None):
                # Add a spot in the latitude/longitude coordinates
                self._add_latlon_spot(params, spot_suffix=suffix)

        # Update the incidence angles included in this patch
        self.unique_mu_included = np.unique(
            self.mu_grid[self.included_segments]
            )

        # Integrate the brightness map
        self.int_brightness = np.sum(self.brightness * self.area_per_segment)

    def __call__(self, wave, flux, params, get_scaling=False, **kwargs):
        """
        Broadens the input spectrum due to rotational effects.

        Args:
        - wave (array): Wavelength array of the input spectrum.
        - flux (array): Flux array of the input spectrum.
        - params (dict): Parameters for rotational broadening.
        - get_scaling (bool): If True, also compute and return the scaling factor.

        Returns:
        - wave (array): The input wavelength array.
        - flux_rot_broad (array): Rotational-broadened flux array.
        """
        # Flux-scaling grid
        #self.get_brightness(params)

        if (self.brightness == 0).all():
            return wave, 0*wave

        if get_scaling:
            # Store the integrated flux
            self.int_flux = np.ones_like(self.brightness) * np.nan

        # Store global flux, integrated over incidence angles
        flux_rot_broad = np.zeros(flux.shape[-1])
        for i, v_i in enumerate(self.v_grid):

            if not self.included_segments[i]:
                continue

            if flux.ndim > 1:
                # Select the correct intensity
                mu_i = self.mu_grid[i]
                idx_mu_i = (self.unique_mu_included == mu_i)
                
                flux_i = flux[idx_mu_i][0]
            else:
                flux_i = flux
            
            # Apply Doppler-shift
            wave_shifted_i = wave * (1 + v_i/(nc.c*1e-5))
            flux_shifted_i = np.interp(
                wave, wave_shifted_i, flux_i, left=np.nan, right=np.nan
                )

            # Scale by (limb)-darkening and angular area
            f_i    = self.brightness[i]
            area_i = self.area_per_segment[i]
            
            # Add to the global spectrum
            flux_rot_broad += f_i * area_i * flux_shifted_i

            if get_scaling:
                # Integrate over wavelengths, store integrated flux of this segment
                mask_isnan = np.isnan(flux_shifted_i)
                self.int_flux[i] = \
                    np.trapz((flux_shifted_i*wave)[~mask_isnan], wave[~mask_isnan]) / \
                    np.trapz(wave[~mask_isnan], wave[~mask_isnan])
                
                # Apply brightness map
                self.int_flux[i] *= f_i / self.int_brightness
        
        # Normalize to account for any over/under-estimation of total flux
        flux_rot_broad *= self.disk_area_tot / self.int_brightness
            
        return wave, flux_rot_broad