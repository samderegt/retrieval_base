import numpy as np

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from .spectrum import Spectrum, ModelSpectrum

class RotationProfile:

    def __init__(self, inc=0, lon_0=0, n_mu=None, n_r=None, n_c=15, n_theta=150, int_method='trapezoid'):
        
        # (Partially) adopted from Carvalho & Johns-Krull (2023)
        self.inc   = np.deg2rad(inc)
        self.lon_0 = np.deg2rad(lon_0)
        
        self.n_mu    = n_mu
        self.n_r     = n_r
        self.n_c     = n_c

        self.n_theta = n_theta

        # Integration method
        self.int_method = int_method

        self.get_coords()
        #print(self.inc)
        #print(self.unique_r, self.unique_mu)
        #print(self.lat_grid)
        #print(self.lon_grid)

    def get_coords(self):

        if self.int_method == 'gauss_quadrature':
            # Gaussian quadrature grid
            self.unique_mu, self.unique_w_gauss_mu = \
                np.polynomial.legendre.leggauss(deg=self.n_mu)
            self.unique_mu         = 1/2*self.unique_mu + 1/2
            self.unique_w_gauss_mu = 1/2*self.unique_w_gauss_mu

            self.unique_r = np.sqrt(1-self.unique_mu**2)

        elif self.int_method == 'trapezoid':

            if self.n_r is not None:
                # Equidistant grid in radius
                dr = 1 / self.n_r

                self.unique_r  = np.arange(dr/2, 1, dr)
                self.unique_mu = np.sqrt(1-self.unique_r**2)

                a = 1.0 - (self.unique_r+dr/2)**2
                b = 1.0 - (self.unique_r-dr/2)**2
                a[a<0] = 0.0
                b[b<0] = 0.0

                self.unique_w_gauss_mu = np.abs(np.sqrt(a) - np.sqrt(b))
                self.unique_w_gauss_mu /= np.sum(self.unique_w_gauss_mu)
                
            elif self.n_mu is not None:
                # Equidistant grid in mu
                dmu = 1 / self.n_mu

                self.unique_mu = np.arange(1-dmu/2, 0, -dmu)
                self.unique_r  = np.sqrt(1-self.unique_mu**2)

                self.unique_w_gauss_mu = np.ones_like(self.unique_mu)
                self.unique_w_gauss_mu /= np.sum(self.unique_w_gauss_mu)

            elif self.n_c is not None:
                # Equidistant grid in angular distance
                # Final incidence angle is not given to pRT
                c = np.linspace(0, np.pi/2, self.n_c+1)

                self.unique_mu = np.cos(c)
                self.unique_r  = np.sqrt(1-self.unique_mu**2)

                self.unique_w_gauss_mu = np.ones_like(self.unique_mu)
                self.unique_w_gauss_mu /= np.sum(self.unique_w_gauss_mu)-1
        
        self.mu_grid, self.w_gauss_mu = [], []
        self.r_grid, self.theta_grid  = [], []
        self.area_per_segment = []

        # Radial grid spacing
        for i, mu_i in enumerate(self.unique_mu):

            # Corresponding radius
            r_i = self.unique_r[i]

            # Reduce number of angular segments close to centre
            n_theta_r_i = int(self.n_theta*r_i)

            if n_theta_r_i in [0,self.n_theta]:
                # Angle-integration not necessary,
                # flux at limb (mu=0) is 0 or vsini is 0
                n_theta_r_i = 1                
                
            # Projected area
            area_ij = 2*np.pi / n_theta_r_i

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
        self.area_per_segment *= len(self.area_per_segment)

        # 2D cartesian coordinates
        x = self.r_grid * np.sin(self.theta_grid)
        y = self.r_grid * np.cos(self.theta_grid)
        c = np.arccos(self.mu_grid) # Angular distance

        # Latitudes + longitudes
        self.lat_grid = np.arcsin(
            np.cos(c)*np.sin(self.inc) + \
            np.cos(self.theta_grid)*np.sin(c)*np.cos(self.inc)
            )
        self.lon_grid = self.lon_0 + np.arctan2(
            x*np.sin(c), self.r_grid * np.cos(c)*np.cos(self.inc) - \
                                y * np.sin(c)*np.sin(self.inc)
            )
        
        self.idx_mu = np.zeros_like(self.mu_grid, dtype=int)        
        for unique_i, unique_mu_i in enumerate(self.unique_mu):
            self.idx_mu[(self.mu_grid==unique_mu_i)] = unique_i
                
    def __call__(self, wave, flux, params, get_scaling=False):

        # Compute velocity-grid
        vsini = params.get('vsini', 0)
        self.v_grid = vsini * self.r_grid * np.sin(self.theta_grid)
        
        # Scale by the inclination, turns vsini into v_eq
        #self.v_grid *= np.abs(np.sin(np.deg2rad(90)-self.inc))

        # Flux-scaling grid
        self.f_grid = np.ones_like(self.r_grid)

        if flux.ndim == 1:
            # Linear limb-darkening for integrated flux
            epsilon_limb = params.get('epsilon_limb', 0)
            self.f_grid *= (1 - epsilon_limb + epsilon_limb*np.sqrt(1-self.r_grid**2))

        if params.get('epsilon_lat') is not None:
            # Latitude-darkening
            epsilon_lat = params.get('epsilon_lat')
            self.f_grid *= (1 - epsilon_lat * np.sin(self.lat_grid)**2)

        if params.get('lat_band') is not None:
            # Add a band at some latitude, with a Gaussian profile
            lat_band     = np.deg2rad(params.get('lat_band'))
            sigma_band   = np.deg2rad(params.get('sigma_band', 0))
            epsilon_band = params.get('epsilon_band', 0)

            self.f_grid *= (
                1 - epsilon_band * np.exp(-(self.lat_grid-lat_band)**2/(2*sigma_band**2))
                )
            
        if params.get('alpha_diff_rot') is not None:
            # Differential rotation
            alpha_diff_rot = params.get('alpha_diff_rot')
            self.v_grid *= (
                1 - alpha_diff_rot * np.sin(self.lat_grid)**2
                )

        self.f_grid /= np.sum(self.f_grid) # Normalize

        # Store intensities per incidence angle
        flux_rot_broad_mu = np.zeros((len(self.unique_mu), flux.shape[-1]))
        for i, v_i in enumerate(self.v_grid):

            idx_mu_i = self.idx_mu[i]
            if flux.ndim > 1:
                if idx_mu_i >= len(flux):
                    # Un-computed flux
                    continue
                flux_i = flux[idx_mu_i]
            else:
                flux_i = flux

            # Apply Doppler-shift
            wave_shifted_i = wave * (1 + v_i/(nc.c*1e-5))
            flux_shifted_i = np.interp(
                wave, wave_shifted_i, flux_i, left=np.nan, right=np.nan
                )

            # Scale by (limb)-darkening and angular area
            f_i    = self.f_grid[i]
            area_i = self.area_per_segment[i] 
            
            # Add to the global spectrum
            flux_rot_broad_mu[idx_mu_i,:] += f_i * area_i * flux_shifted_i

            if get_scaling:
                # Store the flux-scaling of this segment
                self.f_grid[i] = np.nansum(f_i*flux_shifted_i)

        # Integrate over incidence angles
        if self.int_method == 'gauss_quadrature':
            flux_rot_broad = np.sum(
                self.unique_mu[:,None] * flux_rot_broad_mu * self.unique_w_gauss_mu[:,None], axis=0
                )
        elif self.int_method == 'trapezoid':
            flux_rot_broad = np.trapz(
                x=self.unique_mu[::-1], 
                y=(self.unique_mu[:,None] * flux_rot_broad_mu)[::-1], 
                axis=0
                )
        
        if get_scaling:
            # Integrate over wavelengths to store limb-darkening
            self.f_grid /= np.nanmax(self.f_grid)
            
        return flux_rot_broad

class pRT_model:

    def __init__(self, 
                 line_species, 
                 d_spec, 
                 mode='lbl', 
                 lbl_opacity_sampling=3, 
                 cloud_species=None, 
                 rayleigh_species=['H2', 'He'], 
                 continuum_opacities=['H2-H2', 'H2-He'], 
                 log_P_range=(-6,2), 
                 n_atm_layers=50, 
                 cloud_mode=None, 
                 chem_mode='free', 
                 rv_range=(-50,50), 
                 vsini_range=(10,30), 
                 inclination=0, 
                 ):
        '''
        Create instance of the pRT_model class.

        Input
        -----
        line_species : list
            Names of line-lists to include.
        d_spec : DataSpectrum
            Instance of the DataSpectrum class.
        mode : str
            pRT mode to use, can be 'lbl' or 'c-k'.
        lbl_opacity_sampling : int
            Let pRT sample every n-th datapoint.
        cloud_species : list or None
            Chemical cloud species to include. 
        rayleigh_species : list
            Rayleigh-scattering species.
        continuum_opacities : list
            CIA-induced absorption species.
        log_P_range : tuple or list
            Logarithm of modelled pressure range.
        n_atm_layers : int
            Number of atmospheric layers to model.
        cloud_mode : None or str
            Cloud mode to use, can be 'MgSiO3', 'gray' or None.
        chem_mode : str
            Chemistry mode to use for clouds, can be 'free' or 'eqchem'.
        
        '''

        # Create instance of RotationProfile
        self.Rot = RotationProfile(inc=inclination)

        # Read in attributes of the observed spectrum
        self.d_wave          = d_spec.wave
        self.d_mask_isfinite = d_spec.mask_isfinite
        self.d_resolution    = d_spec.resolution
        self.apply_high_pass_filter = d_spec.high_pass_filtered
        self.w_set = d_spec.w_set

        self.line_species = line_species
        self.mode = mode
        self.lbl_opacity_sampling = lbl_opacity_sampling

        self.cloud_species     = cloud_species
        self.rayleigh_species  = rayleigh_species
        self.continuum_species = continuum_opacities

        # Clouds
        if self.cloud_species is None:
            self.do_scat_emis = False
        else:
            self.do_scat_emis = True

        self.cloud_mode = cloud_mode
        self.chem_mode  = chem_mode

        self.rv_max = np.array(list(rv_range))
        self.rv_max[0] -= np.max(list(vsini_range))
        self.rv_max[1] += np.max(list(vsini_range))
        self.rv_max = np.max(np.abs(self.rv_max))

        # Define the atmospheric layers
        if log_P_range is None:
            log_P_range = (-6,2)
        if n_atm_layers is None:
            n_atm_layers = 50
        self.pressure = np.logspace(
            log_P_range[0], log_P_range[1], n_atm_layers, dtype=np.float64
            )

        # Make the pRT.Radtrans objects
        self.get_atmospheres(CB_active=False)

    def get_atmospheres(self, CB_active=False):

        # pRT model is somewhat wider than observed spectrum
        if CB_active:
            self.rv_max = 1000
        wave_pad = 1.1 * self.rv_max/(nc.c*1e-5) * self.d_wave.max()

        self.wave_range_micron = np.concatenate(
            (self.d_wave.min(axis=(1,2))[None,:]-wave_pad, 
             self.d_wave.max(axis=(1,2))[None,:]+wave_pad
            )).T
        self.wave_range_micron *= 1e-3

        self.atm = []
        for wave_range_i in self.wave_range_micron:
            
            # Make a pRT.Radtrans object
            atm_i = Radtrans(
                line_species=self.line_species, 
                rayleigh_species=self.rayleigh_species, 
                continuum_opacities=self.continuum_species, 
                cloud_species=self.cloud_species, 
                wlen_bords_micron=wave_range_i, 
                mode=self.mode, 
                lbl_opacity_sampling=self.lbl_opacity_sampling, 
                do_scat_emis=self.do_scat_emis
                )

            # Set up the atmospheric layers
            atm_i.setup_opa_structure(self.pressure)

            # Modify the incidence angles
            atm_i.mu = self.Rot.unique_mu[:-1]
            
            # Equal weights... not a proper Gaussian quadrature integration
            atm_i.w_gauss_mu = self.Rot.unique_w_gauss_mu[:-1]

            self.atm.append(atm_i)

    def __call__(self, 
                 mass_fractions, 
                 temperature, 
                 params, 
                 get_contr=False, 
                 get_full_spectrum=False, 
                 ):
        '''
        Create a new model spectrum with the given arguments.

        Input
        -----
        mass_fractions : dict
            Species' mass fractions in the pRT format.
        temperature : np.ndarray
            Array of temperatures at each atmospheric layer.
        params : dict
            Parameters of the current model.
        get_contr : bool
            If True, compute the emission contribution function. 

        Returns
        -------
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class. 
        '''

        # Update certain attributes
        self.mass_fractions = mass_fractions
        self.temperature    = temperature
        self.params = params

        if self.params.get('res') is not None:
            self.d_resolution = self.params['res']
        if self.params.get(f'res_{self.w_set}') is not None:
            self.d_resolution = self.params[f'res_{self.w_set}']

        # Add clouds if requested
        self.add_clouds()

        # Generate a model spectrum
        m_spec = self.get_model_spectrum(
            get_contr=get_contr, get_full_spectrum=get_full_spectrum
            )
        return m_spec

    def add_clouds(self):
        '''
        Add clouds to the model atmosphere using the given parameters.
        '''

        self.give_absorption_opacity = None
        self.f_sed  = None
        self.K_zz   = None
        self.sigma_g = None

        if self.cloud_mode == 'MgSiO3':

            # Mask the pressure above the cloud deck
            mask_above_deck = (self.pressure <= self.params['P_base_MgSiO3'])

            # Add the MgSiO3 particles
            self.mass_fractions['MgSiO3(c)'] = np.zeros_like(self.pressure)
            self.mass_fractions['MgSiO3(c)'][mask_above_deck] = self.params['X_cloud_base_MgSiO3'] * \
                (self.pressure[mask_above_deck]/self.params['P_base_MgSiO3'])**self.params['f_sed']
            #self.params['K_zz'] = self.params['K_zz'] * np.ones_like(self.pressure)
            self.K_zz = self.params['K_zz'] * np.ones_like(self.pressure)
            self.sigma_g = self.params['sigma_g']

            self.f_sed = {'MgSiO3(c)': self.params['f_sed']}
        
        elif self.cloud_mode == 'gray':
            
            # Gray cloud opacity
            self.give_absorption_opacity = self.gray_cloud_opacity

    def gray_cloud_opacity(self, wave_micron, pressure):
        '''
        Function to be called by petitRADTRANS. 

        Input
        -----
        wave_micron: np.ndarray
            Wavelength in micron.
        pressure: np.ndarray
            Pressure in bar.

        Output
        ------
        opa_gray_cloud: np.ndarray
            Gray cloud opacity for each wavelength and pressure layer.
        '''

        # Create gray cloud opacity, i.e. independent of wavelength
        opa_gray_cloud = np.zeros((len(wave_micron), len(pressure)))

        # Constant below the cloud base
        #opa_gray_cloud[:,pressure >= params['P_base_gray']] = params['opa_base_gray']
        opa_gray_cloud[:,pressure > self.params['P_base_gray']] = 0

        # Opacity decreases with power-law above the base
        mask_above_deck = (pressure <= self.params['P_base_gray'])
        opa_gray_cloud[:,mask_above_deck] = self.params['opa_base_gray'] * \
            (pressure[mask_above_deck]/self.params['P_base_gray'])**self.params['f_sed_gray']

        if self.params.get('cloud_slope') is not None:
            opa_gray_cloud *= (wave_micron[:,None] / 1)**self.params['cloud_slope']

        return opa_gray_cloud

    def get_model_spectrum(self, get_contr=False, get_full_spectrum=False):
        '''
        Generate a model spectrum with the given parameters.

        Input
        -----
        get_contr : bool
            If True, computes the emission contribution 
            and cloud opacity. Updates the contr_em and 
            opa_cloud attributes.
        
        Returns
        -------
        m_spec : ModelSpectrum class
            Instance of the ModelSpectrum class
        '''

        # Loop over all orders
        wave = np.ones_like(self.d_wave) * np.nan
        flux = np.ones_like(self.d_wave) * np.nan
        
        self.int_contr_em  = np.zeros_like(self.pressure)
        self.int_opa_cloud = np.zeros_like(self.pressure)

        self.int_contr_em_per_order = np.zeros((self.d_wave.shape[0], len(self.pressure)))

        self.CCF, self.m_ACF = [], []
        self.wave_pRT_grid, self.flux_pRT_grid = [], []

        for i, atm_i in enumerate(self.atm):
            
            # Compute the emission spectrum
            atm_i.calc_flux(
                self.temperature, 
                self.mass_fractions, 
                gravity=10**self.params['log_g'], 
                mmw=self.mass_fractions['MMW'], 
                Kzz=self.K_zz, 
                fsed=self.f_sed, 
                sigma_lnorm=self.sigma_g,
                give_absorption_opacity=self.give_absorption_opacity, 
                contribution=get_contr, 
                )
            wave_i = nc.c / atm_i.freq
            if hasattr(atm_i, 'flux_mu'):
                flux_i = atm_i.flux_mu
            else:
                flux_i = atm_i.flux

            # Convert [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
            if flux_i.ndim > 1:
                flux_i *= nc.c / (wave_i[None,:]**2)
            else:
                flux_i *= nc.c / (wave_i**2)

            # Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
            flux_i /= 1e7

            # Convert [cm] -> [nm]
            wave_i *= 1e7
            
            # Apply RV-shift
            wave_i *= (1 + self.params['rv']/(nc.c*1e-5))

            # Apply rotational broadening
            flux_i = self.Rot(wave_i, flux_i, self.params, get_scaling=get_full_spectrum)

            # Convert to observation by scaling with planetary radius
            flux_i *= (
                (self.params['R_p']*nc.r_jup_mean) / \
                (1e3/self.params['parallax']*nc.pc)
                )**2
            
            # Create a ModelSpectrum instance
            m_spec_i = ModelSpectrum(
                wave=wave_i, flux=flux_i, 
                lbl_opacity_sampling=self.lbl_opacity_sampling
                )
            
            # Apply instrumental broadening
            m_spec_i.flux = m_spec_i.instr_broadening(
                m_spec_i.wave, m_spec_i.flux, 
                out_res=self.d_resolution, 
                in_res=m_spec_i.resolution
                )

            '''            
            # Apply radial-velocity shift, rotational/instrumental broadening
            m_spec_i.shift_broaden_rebin(
                rv=self.params['rv'], 
                vsini=self.params['vsini'], 
                epsilon_limb=self.params['epsilon_limb'], 
                epsilon_lat=self.params.get('epsilon_lat', 0), 
                dif_rot_delta=self.params.get('dif_rot_delta', 0), 
                dif_rot_phi=self.params.get('dif_rot_phi', 1), 
                epsilon_band=self.params.get('epsilon_band', 0), 
                lat_band=self.params.get('lat_band'), 
                sigma_band=self.params.get('sigma_band', 0),  
                inclination=self.params.get('inclination', 0), 
                out_res=self.d_resolution, 
                in_res=m_spec_i.resolution, 
                rebin=False, 
                )
            '''
            if get_full_spectrum:
                # Store the spectrum before the rebinning
                self.wave_pRT_grid.append(m_spec_i.wave)
                self.flux_pRT_grid.append(m_spec_i.flux)

            # Rebin onto the data's wavelength grid
            m_spec_i.rebin(d_wave=self.d_wave[i,:], replace_wave_flux=True)

            if self.apply_high_pass_filter:
                # High-pass filter the model spectrum
                m_spec_i.high_pass_filter(
                    removal_mode='divide', 
                    filter_mode='gaussian', 
                    sigma=300, 
                    replace_flux_err=True
                    )

            wave[i,:,:] = m_spec_i.wave
            flux[i,:,:] = m_spec_i.flux
            
            if get_contr:

                # Integrate the emission contribution function and cloud opacity
                self.get_integrated_contr_em_and_opa_cloud(
                    atm_i, m_wave_i=wave_i, 
                    d_wave_i=self.d_wave[i,:], 
                    d_mask_i=self.d_mask_isfinite[i], 
                    m_spec_i=m_spec_i, 
                    order=i
                    )

        # Create a new ModelSpectrum instance with all orders
        m_spec = ModelSpectrum(
            wave=wave, 
            flux=flux, 
            lbl_opacity_sampling=self.lbl_opacity_sampling, 
            multiple_orders=True, 
            high_pass_filtered=self.apply_high_pass_filter, 
            )

        # Convert to arrays
        self.CCF, self.m_ACF = np.array(self.CCF), np.array(self.m_ACF)
        #self.wave_pRT_grid = np.array(self.wave_pRT_grid)
        #self.flux_pRT_grid = np.array(self.flux_pRT_grid)

        # Save memory, same attributes in DataSpectrum
        del m_spec.wave, m_spec.mask_isfinite

        return m_spec

    def get_integrated_contr_em_and_opa_cloud(self, 
                                              atm_i, 
                                              m_wave_i, 
                                              d_wave_i, 
                                              d_mask_i, 
                                              m_spec_i, 
                                              order
                                              ):
        
        # Get the emission contribution function
        contr_em_i = atm_i.contr_em
        new_contr_em_i = []

        # Get the cloud opacity
        if self.cloud_mode == 'gray':
            opa_cloud_i = self.gray_cloud_opacity(m_wave_i*1e-3, self.pressure).T
        elif self.cloud_mode == 'MgSiO3':
            opa_cloud_i = atm_i.tau_cloud.T
        else:
            opa_cloud_i = np.zeros_like(contr_em_i)

        for j, (contr_em_ij, opa_cloud_ij) in enumerate(zip(contr_em_i, opa_cloud_i)):
            
            # Similar to the model flux
            contr_em_ij = ModelSpectrum(
                wave=m_wave_i, flux=contr_em_ij, 
                lbl_opacity_sampling=self.lbl_opacity_sampling
                )
            # Shift, broaden, rebin the contribution
            contr_em_ij.shift_broaden_rebin(
                d_wave=d_wave_i, 
                rv=self.params['rv'], 
                vsini=self.params['vsini'], 
                epsilon_limb=self.params.get('epsilon_limb', 0), 
                epsilon_lat=self.params.get('epsilon_lat', 0), 
                dif_rot_delta=self.params.get('dif_rot_delta', 0), 
                dif_rot_phi=self.params.get('dif_rot_phi', 1), 
                epsilon_band=self.params.get('epsilon_band', 0), 
                lat_band=self.params.get('lat_band'), 
                sigma_band=self.params.get('sigma_band', 0),  
                inclination=self.params.get('inclination', 0), 
                out_res=self.d_resolution, 
                in_res=m_spec_i.resolution, 
                rebin=True, 
                )
            # Compute the spectrally-weighted emission contribution function
            # Integrate and weigh the emission contribution function
            self.int_contr_em_per_order[order,j] = \
                contr_em_ij.spectrally_weighted_integration(
                    wave=d_wave_i[d_mask_i].flatten(), 
                    flux=m_spec_i.flux[d_mask_i].flatten(), 
                    array=contr_em_ij.flux[d_mask_i].flatten(), 
                    )
            self.int_contr_em[j] += self.int_contr_em_per_order[order,j]

            # Similar to the model flux
            opa_cloud_ij = ModelSpectrum(
                wave=m_wave_i, flux=opa_cloud_ij, 
                lbl_opacity_sampling=self.lbl_opacity_sampling
                )
            # Shift, broaden, rebin the cloud opacity
            opa_cloud_ij.shift_broaden_rebin(
                d_wave=d_wave_i, 
                rv=self.params['rv'], 
                vsini=self.params['vsini'], 
                epsilon_limb=self.params.get('epsilon_limb', 0), 
                epsilon_lat=self.params.get('epsilon_lat', 0), 
                dif_rot_delta=self.params.get('dif_rot_delta', 0), 
                dif_rot_phi=self.params.get('dif_rot_phi', 1), 
                epsilon_band=self.params.get('epsilon_band', 0), 
                lat_band=self.params.get('lat_band'), 
                sigma_band=self.params.get('sigma_band', 0),  
                inclination=self.params.get('inclination', 0), 
                out_res=self.d_resolution, 
                in_res=m_spec_i.resolution, 
                rebin=True, 
                )
            # Integrate and weigh the cloud opacity
            self.int_opa_cloud[j] += \
                opa_cloud_ij.spectrally_weighted_integration(
                    wave=d_wave_i[d_mask_i].flatten(), 
                    flux=m_spec_i.flux[d_mask_i].flatten(), 
                    array=opa_cloud_ij.flux[d_mask_i].flatten(), 
                    )
