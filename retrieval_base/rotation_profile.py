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

        # Define the grid with a fine spacing
        self.get_coords(sampling_factor=10)
        self.lat_grid_fine = self.lat_grid.copy()
        self.lon_grid_fine = self.lon_grid.copy()

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

    def get_f_grid(self, params):
        
        f_grid = np.ones_like(self.r_grid)

        if params.get('epsilon_limb') is not None:
            # Linear limb-darkening for integrated flux
            epsilon_limb = params.get('epsilon_limb')
            f_grid *= (
                1 - epsilon_limb + epsilon_limb*np.sqrt(1-self.r_grid**2)
                )

        if params.get('epsilon_lat') is not None:
            # Latitude-darkening
            epsilon_lat = params.get('epsilon_lat')
            f_grid *= (1 - epsilon_lat * np.sin(self.lat_grid)**2)

        if params.get('lat_band') is not None:
            # Add a band at some latitude
            lat_band = np.deg2rad(params.get('lat_band'))

            sigma_band   = np.deg2rad(params.get('sigma_band', 0))
            epsilon_band = params.get('epsilon_band', 1)

            if sigma_band != 0:
                # Band with a Gaussian profile
                f_grid *= (
                    1 - epsilon_band * np.exp(-(self.lat_grid-lat_band)**2/(2*sigma_band**2))
                    )
            else:
                # Change the flux between some latitudes
                lat_band_upper = np.deg2rad(params.get('lat_band_upper', 90))
                
                mask_above_band = (np.abs(self.lat_grid) > lat_band)
                if lat_band_upper > lat_band:
                    # Dark band at latitudes above equator
                    mask_above_band = mask_above_band & (np.abs(self.lat_grid) < lat_band_upper)
                else:
                    # Flip the bands, i.e. dark band on equator
                    mask_above_band = mask_above_band | (np.abs(self.lat_grid) < lat_band_upper)

                if params.get('eq_band'):
                    # Cloudy atmosphere on equator
                    f_grid[mask_above_band] *= 0
                elif params.get('above_eq_band'):
                    # Clear atmosphere above equatorial band
                    f_grid[~mask_above_band] *= 0
                else:
                    if epsilon_band < 0:
                        # Dark band on equator
                        f_grid[~mask_above_band] *= (1 - np.abs(epsilon_band))
                    else:
                        # Bright band on equator
                        f_grid[mask_above_band] *= (1 - epsilon_band)

        if params.get('lat_band_low') is not None:
            # Add a band at some latitude
            lat_band = np.deg2rad(params.get('lat_band'))
            epsilon_band = params.get('epsilon_band', 1)

        return f_grid

    def __call__(self, wave, flux, params, f_grid=None, get_scaling=False, **kwargs):

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
        if f_grid is None:
            f_grid = self.get_f_grid(params)

        #f_grid = np.ones_like(self.r_grid)

        if (f_grid == 0).all():
            return wave, 0*wave

        integrated_f_grid = np.sum(f_grid * self.area_per_segment)
        if get_scaling:
            self.f_grid = np.ones_like(f_grid) * np.nan

        # Store global flux, integrated over incidence angles
        flux_rot_broad = np.zeros(flux.shape[-1])
        for i, v_i in enumerate(self.v_grid):

            idx_mu_i = self.idx_mu[i]
            if flux.ndim > 1:
                flux_i = flux[idx_mu_i]
            else:
                flux_i = flux

            # Apply Doppler-shift
            wave_shifted_i = wave * (1 + v_i/(nc.c*1e-5))
            flux_shifted_i = np.interp(
                wave, wave_shifted_i, flux_i, left=np.nan, right=np.nan
                )

            # Scale by (limb)-darkening and angular area
            f_i    = f_grid[i]
            area_i = self.area_per_segment[i]
            
            # Add to the global spectrum
            flux_rot_broad += f_i * area_i * flux_shifted_i

            if get_scaling:
                # Store the flux-scaling of this segment
                self.f_grid[i] = np.nansum(f_i*flux_shifted_i / integrated_f_grid)
        
        # Normalize to account for any over/under-estimation of total flux
        #flux_rot_broad *= self.disk_area_tot / integrated_f_grid
        
        if get_scaling:
            # Integrate over wavelengths to store limb-darkening
            #self.f_grid /= np.nanmax(self.f_grid)
            pass
            
        return wave, flux_rot_broad