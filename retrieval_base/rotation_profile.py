import numpy as np
import petitRADTRANS.nat_cst as nc

def get_Rotation_class(mode='convolve', **kwargs):

    if mode == 'integrate':
        return IntRotationProfile(**kwargs)
    if mode == 'convolve':
        return ConvRotationProfile(**kwargs)

class ConvRotationProfile:
    
    def __init__(self, **kwargs):

        from PyAstronomy import pyasl
        self.fastRotBroad = pyasl.fastRotBroad

    def __call__(self, wave, flux, params, **kwargs):

        # Evenly space the wavelength grid
        wave_even = np.linspace(wave.min(), wave.max(), wave.size)
        flux_even = np.interp(wave_even, xp=wave, fp=flux)

        epsilon_limb = params.get('epsilon_limb', 0)
        vsini = params.get('vsini', 0)

        if vsini == 0:
            return wave_even, flux_even

        # Rotational broadening of the model spectrum
        flux_rot_broad = self.fastRotBroad(
            wave_even, flux_even, 
            epsilon=epsilon_limb, vsini=vsini
            )
        
        return wave_even, flux_rot_broad

class IntRotationProfile:

    def __init__(self, inc=0, lon_0=0, n_c=15, n_theta=150, **kwargs):
        
        # (Partially) adopted from Carvalho & Johns-Krull (2023)
        self.inc   = np.deg2rad(inc)
        self.lon_0 = np.deg2rad(lon_0)
        
        # Define grid to integrate over
        self.n_c     = n_c
        self.n_theta = n_theta

        # Define the grid with the regular spacing
        self.get_coords()

    def get_coords(self, sampling_factor=1):

        # Compute coordinates on a finer grid?
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

        # 2D cartesian coordinates
        x = self.r_grid * np.sin(self.theta_grid)
        y = self.r_grid * np.cos(self.theta_grid)
        self.c_grid = np.arccos(self.mu_grid) # Angular distance

        # Latitudes + longitudes
        self.lat_grid = np.arcsin(
            np.cos(self.c_grid)*np.sin(self.inc) + \
            np.cos(self.theta_grid)*np.sin(self.c_grid)*np.cos(self.inc)
            )
        self.lon_grid = self.lon_0 + np.arctan2(
            x*np.sin(self.c_grid), self.r_grid * np.cos(self.c_grid)*np.cos(self.inc) - \
                                y * np.sin(self.c_grid)*np.sin(self.inc)
            )
        
        self.idx_mu = np.zeros_like(self.mu_grid, dtype=int)
        for unique_i, unique_mu_i in enumerate(self.unique_mu):
            self.idx_mu[(self.mu_grid==unique_mu_i)] = unique_i

    def get_brightness(self, params, N_spot_max=5):
        
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

        if params.get('lat_band') is not None:
            # Add a band at some latitude
            lat_band = np.deg2rad(params.get('lat_band'))
            epsilon_band = params.get('epsilon_band', 1)

            # Change the flux between some latitudes
            mask_patch = (np.abs(self.lat_grid) < lat_band)
            if epsilon_band < 0:
                epsilon_band = 1 + epsilon_band
            self.brightness[mask_patch] *= epsilon_band

        if (params.get('lon_band') is not None) and \
            (params.get('lon_band_width') is not None):
            # Add a band at some longitude with a certain width
            lon_band       = np.deg2rad(params.get('lon_band'))
            lon_band_width = np.deg2rad(params.get('lon_band_width'))

            lon_band_upper = lon_band + lon_band_width/2
            lon_band_lower = lon_band - lon_band_width/2

            if (lon_band_lower > -np.pi) and (lon_band_upper < np.pi):
                mask_patch = (self.lon_grid >= lon_band_lower) & \
                    (self.lon_grid <= lon_band_upper)
            elif lon_band_upper > np.pi:
                mask_patch = (self.lon_grid >= lon_band_lower) | \
                    (self.lon_grid <= lon_band_upper-2*np.pi)
            elif lon_band_lower < -np.pi:
                mask_patch = (self.lon_grid >= lon_band_lower+2*np.pi) | \
                    (self.lon_grid <= lon_band_upper)

        # Loop over multiple spots
        for i in range(N_spot_max):

            epsilon_spot = params.get(f'epsilon_spot_{i}')

            lon_spot = params.get(f'lon_spot_{i}')
            lat_spot = params.get(f'lat_spot_{i}')
            
            radius_spot  = params.get(f'radius_spot_{i}') # Circular spot
            a_spot       = params.get(f'a_spot_{i}') # Semi-major axis of ellipse
            b_spot       = params.get(f'b_spot_{i}') # Semi-minor axis of ellipse

            no_size = np.all([s_i is None for s_i in [radius_spot,a_spot,b_spot]])

            if (lon_spot is None) and (lat_spot is None) and no_size and (i == 0):
                # Revert to a single spot
                epsilon_spot = params.get('epsilon_spot')

                lon_spot = params.get('lon_spot')
                lat_spot = params.get('lat_spot')

                radius_spot  = params.get('radius_spot') # Circle
                a_spot       = params.get('a_spot') # Ellipse
                b_spot       = params.get('b_spot')

            no_size = np.all([s_i is None for s_i in [radius_spot,a_spot,b_spot]])

            if (lon_spot is None) or (lat_spot is None) or no_size:
                # No spot found in params-dictionary, break loop
                break
            
            lon_spot = np.deg2rad(lon_spot)
            lat_spot = np.deg2rad(lat_spot)

            is_circle = (radius_spot is not None)

            if is_circle:
                # Add a circular spot
                radius_spot = np.deg2rad(radius_spot)

                # Haversine formula
                distance_from_spot = 2 * np.arcsin((1/2*(
                    1 - np.cos(self.lat_grid - lat_spot) + \
                    np.cos(lat_spot)*np.cos(self.lat_grid)*(1-np.cos(self.lon_grid-lon_spot))
                    ))**(1/2))
                mask_patch = (distance_from_spot <= radius_spot)
            else:
                # Add an elliptical spot
                a_spot = np.deg2rad(a_spot)
                b_spot = np.deg2rad(b_spot)

                # Use simple Cartesian? coordinates, results in changing
                # spot-size with changing latitude
                distance_lon = np.abs(self.lon_grid - lon_spot)
                # Make longitude-distance continuous
                mask_invalid = (distance_lon > np.pi)
                distance_lon[mask_invalid] = np.pi - distance_lon[mask_invalid] % np.pi

                # Latitude-distance
                distance_lat = np.abs(self.lat_grid - lat_spot)

                # Ellipse equation
                mask_patch = (
                    (distance_lon/a_spot)**2 + (distance_lat/b_spot)**2 <= 1
                    )

            if epsilon_spot is not None:
                # Scale the brightness
                self.brightness[mask_patch] *= epsilon_spot

        if params.get('is_within_patch'):
            self.included_segments[mask_patch]  = True
            self.included_segments[~mask_patch] = False
        if params.get('is_outside_patch'):
            self.included_segments[mask_patch]  = False
            self.included_segments[~mask_patch] = True

        # Update the incidence angles included in this patch
        self.unique_mu_included = np.unique(
            self.mu_grid[self.included_segments]
            )

        # Integrate the brightness map
        self.int_brightness = np.sum(self.brightness * self.area_per_segment)

    def __call__(self, wave, flux, params, get_scaling=False, **kwargs):

        # Compute velocity-grid
        vsini = params.get('vsini', 0)
        self.v_grid = vsini * self.r_grid * np.sin(self.theta_grid)
        
        # Scale by the inclination, turns vsini into v_eq
        #self.v_grid *= np.abs(np.sin(np.deg2rad(90)-self.inc))

        if params.get('alpha_diff_rot') is not None:
            # Differential rotation
            alpha_diff_rot = params.get('alpha_diff_rot')
            self.v_grid *= (
                1 - alpha_diff_rot * np.sin(self.lat_grid)**2
                )
        
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