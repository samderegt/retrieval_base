import numpy as np

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc

from .spectrum import ModelSpectrum
from .rotation_profile import get_Rotation_class
from .clouds import get_Cloud_class
from .opacity import LineOpacity

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
                 rv_range=(-50,50), 
                 vsini_range=(10,30), 
                 rotation_mode='convolve', 
                 inclination=0, 
                 sum_m_spec=False, 
                 do_scat_emis=False, 
                 line_opacity_kwargs=None, 
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
        
        '''

        # Create instance of RotationProfile
        self.rotation_mode = rotation_mode
        self.Rot = get_Rotation_class(
            mode=self.rotation_mode, inc=inclination
            )
        
        # Instrumental broadening should be applied after summing spectra
        self.sum_m_spec = sum_m_spec

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

        self.do_scat_emis = do_scat_emis

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

        # Clouds
        self.cloud_mode = cloud_mode
        self.Cloud = get_Cloud_class(
            mode=self.cloud_mode, pressure=self.pressure, 
            cloud_species=self.cloud_species
            )
        
        # Make the pRT.Radtrans objects
        self.get_atmospheres(CB_active=False)

        # Custom (on-the-fly) line opacities
        if line_opacity_kwargs is not None:
            import copy

            # Extend to the broad wavelength range
            rv_max = 1001
            wave_pad = 1.1 * rv_max/(nc.c*1e-5) * self.d_wave.max()
            wave_range_micron = np.concatenate(
                (self.d_wave.min(axis=(1,2))[None,:]-wave_pad, 
                 self.d_wave.max(axis=(1,2))[None,:]+wave_pad
                )).T
            wave_range_micron *= 1e-3

            wave_micron_broad = []
            for atm_i, wave_range_micron_i in zip(self.atm, wave_range_micron):
                
                # Temporary copy of Radtrans object
                new_atm_i = copy.deepcopy(atm_i)
                new_atm_i.wlen_bords_micron = wave_range_micron_i
                new_atm_i.lbl_opacity_sampling = None
                
                # Obtain pRT frequency/wavelength grid
                freq_i, *_ = new_atm_i._init_line_opacities_parameters()
                
                # Find the starting index
                for idx_0 in range(self.lbl_opacity_sampling):
                    freq_ij = freq_i[idx_0::self.lbl_opacity_sampling]
                    if np.isin(atm_i.freq, freq_ij).all():
                        break

                wave_micron_broad.append(1e4*nc.c/freq_ij)
                del new_atm_i

            wave_micron_broad = np.concatenate(wave_micron_broad)
            wave_micron_broad = np.unique(wave_micron_broad) # Remove duplicates
            
            # Create instance of custom opacities
            self.LineOpacity = LineOpacity(
                pressure=self.pressure, 
                wave_range_micron=wave_range_micron, # Use broad wavelength range
                wave_micron_broad=wave_micron_broad, 
                **line_opacity_kwargs
                )
            
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
            
            # Make a copy, otherwise pRT crashes
            if isinstance(self.cloud_species, list):
                cloud_species_i = self.cloud_species.copy()
            else:
                cloud_species_i = self.cloud_species

            # Make a pRT.Radtrans object
            atm_i = Radtrans(
                line_species=self.line_species, 
                rayleigh_species=self.rayleigh_species, 
                continuum_opacities=self.continuum_species, 
                cloud_species=cloud_species_i, 
                wlen_bords_micron=wave_range_i, 
                mode=self.mode, 
                lbl_opacity_sampling=self.lbl_opacity_sampling, 
                do_scat_emis=self.do_scat_emis
                )

            # Set up the atmospheric layers
            atm_i.setup_opa_structure(self.pressure)

            self.atm.append(atm_i)
        
    def give_absorption_opacity(self):

        include_cloud = (self.Cloud.get_opacity is not None)
        include_line  = hasattr(self, 'LineOpacity')

        if (not include_cloud) and (not include_line):
            return None

        # Define the function
        if include_cloud and (not include_line):
            def get_opacity(wave_micron, pressure):
                return self.Cloud.get_opacity(wave_micron, pressure)
        if include_line and (not include_cloud):
            def get_opacity(wave_micron, pressure):
                return self.LineOpacity.get_line_opacity(wave_micron, pressure)
        if include_cloud and include_line:
            def get_opacity(wave_micron, pressure):
                return self.Cloud.get_opacity(wave_micron, pressure) + \
                    self.LineOpacity.get_line_opacity(wave_micron, pressure)
        return get_opacity

    def __call__(self, 
                 mass_fractions, 
                 temperature, 
                 params, 
                 get_contr=False, 
                 get_full_spectrum=False, 
                 CO=0.59, 
                 FeH=0.0, 
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
        self.mass_fractions = mass_fractions.copy()
        self.temperature    = temperature
        self.params = params

        if self.params.get('res') is not None:
            self.d_resolution = self.params['res']
        if self.params.get(f'res_{self.w_set}') is not None:
            self.d_resolution = self.params[f'res_{self.w_set}']

        # Update the cloud parameters + add abundances if necessary
        self.mass_fractions = self.Cloud(
            params=self.params, mass_fractions=self.mass_fractions, 
            temperature=temperature, CO=CO, FeH=FeH, 
            )
        
        if hasattr(self, 'LineOpacity'):
            # Update the on-the-fly opacity with new parameters
            self.LineOpacity(
                self.params, self.temperature, self.mass_fractions
                )

        # Additional opacity from a cloud or custom line profile
        self.add_opacity = self.give_absorption_opacity()

        # Generate a model spectrum
        m_spec = self.get_model_spectrum(
            get_contr=get_contr, get_full_spectrum=get_full_spectrum
            )

        return m_spec

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

        self.int_contr_em  = np.zeros_like(self.pressure)
        self.int_opa_cloud = np.zeros_like(self.pressure)
        self.int_contr_em_per_order = \
            np.zeros((self.d_wave.shape[0], len(self.pressure)))
        
        # Convert to arrays
        self.CCF, self.m_ACF = np.array([]), np.array([])

        # List to store the preliminary model
        self.pRT_wave, self.pRT_flux = [], []

        # pRT settings
        calc_flux_kwargs = dict(
            temp    = self.temperature, 
            abunds  = self.mass_fractions, 
            gravity = 10**self.params['log_g'], 
            mmw     = self.mass_fractions['MMW'], 
            Kzz     = self.Cloud.K_zz, 
            fsed    = self.Cloud.f_sed, 
            sigma_lnorm = self.Cloud.sigma_g,
            give_absorption_opacity = self.add_opacity, 
            contribution = get_contr, 
            )

        # Loop over all orders        
        for i, atm_i in enumerate(self.atm):

            if self.rotation_mode == 'integrate':
                if i == 0:
                    new_inc = self.params.get('inclination', 0)
                    if new_inc != np.rad2deg(self.Rot.inc):
                        # Update the inclination and lat/lon-coords
                        self.Rot.inc = np.deg2rad(new_inc)
                        self.Rot.get_latlon()

                    # Update the brightness map
                    self.Rot.get_brightness(params=self.params)
                
                if len(self.Rot.unique_mu_included) == 0:
                    self.pRT_wave.append(np.zeros_like(atm_i.freq))
                    self.pRT_flux.append(np.zeros_like(atm_i.freq))
                    continue

                # Only compute for the angles in the patch
                atm_i.mu = self.Rot.unique_mu_included
                # Equal weights... not a proper Gaussian quadrature integration
                atm_i.w_gauss_mu = np.ones_like(self.Rot.unique_mu_included) / \
                    len(self.Rot.unique_mu_included)

                # Return the intensities per incidence angle
                calc_flux_kwargs['return_per_mu'] = True

            # Compute the emission spectrum
            atm_i.calc_flux(**calc_flux_kwargs)

            wave_i = nc.c / atm_i.freq
            if self.rotation_mode == 'integrate' and hasattr(atm_i, 'flux_mu'):
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
            wave_i, flux_i = self.Rot(
                wave_i, flux_i, self.params, get_scaling=get_full_spectrum
                )

            # Convert to observation by scaling with planetary radius
            flux_i *= (
                (self.params.get('R_p', 1)*nc.r_jup_mean) / \
                (1e3/self.params['parallax']*nc.pc)
                )**2

            # Store the model spectrum to be summed or broadened later
            self.pRT_wave.append(wave_i)
            self.pRT_flux.append(flux_i)

        m_spec = None
        if not self.sum_m_spec:
            # Single model spectrum can be instrumentally broadened
            m_spec = self.combine_models(
                get_contr=get_contr, get_full_spectrum=get_full_spectrum, 
                )
        
        return m_spec
    
    def combine_models(
            self, other_pRT_wave=None, other_pRT_flux=None, 
            get_contr=False, get_full_spectrum=False
            ):

        wave = np.ones_like(self.d_wave) * np.nan
        flux = np.ones_like(self.d_wave) * np.nan

        self.wave_pRT_grid, self.flux_pRT_grid = [], []

        if isinstance(other_pRT_flux, (tuple, list)):

            cloud_fraction = self.params.get('cloud_fraction')
            if cloud_fraction is None:
                # Equal contributions (i.e. binary atmospheres)
                f = np.ones(len(other_pRT_flux))
            else:
                # Patchy clouds (only works for 2 models)
                f = np.array([(1-cloud_fraction)])

                # Loop over orders
                self.pRT_flux = [
                    pRT_flux_i*cloud_fraction for pRT_flux_i in self.pRT_flux
                    ]

            # Loop over model-settings
            for f_m_set, pRT_wave_m_set, pRT_flux_m_set in \
                zip(f, other_pRT_wave, other_pRT_flux):
                
                # Loop over orders
                for i in range(len(self.pRT_flux)):

                    if (pRT_flux_m_set[i] == 0).all():
                        continue
                    if (self.pRT_wave[i] == 0).all():
                        # First model-setting had no incidence angles
                        self.pRT_wave[i] = pRT_wave_m_set[i]

                    # Interpolate onto the wavelengths of 1st model
                    mask = np.isfinite(pRT_flux_m_set[i])
                    pRT_flux_m_set_i = np.interp(
                        self.pRT_wave[i], 
                        xp=pRT_wave_m_set[i][mask], fp=pRT_flux_m_set[i][mask], 
                        left=np.nan, right=np.nan
                        )
                    # Combine the spectra
                    self.pRT_flux[i] += f_m_set*pRT_flux_m_set_i
            
            # Clear up some memory
            del other_pRT_wave, other_pRT_flux

        # Loop over orders
        for i, (wave_i, flux_i) in enumerate(zip(self.pRT_wave, self.pRT_flux)):

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

            if get_full_spectrum:
                # Store the spectrum before the rebinning
                self.wave_pRT_grid.append(m_spec_i.wave)
                self.flux_pRT_grid.append(m_spec_i.flux)

            # Rebin onto the data's wavelength grid
            m_spec_i.rebin(d_wave=self.d_wave[i,:], replace_wave_flux=True)

            # Store the instrumentally-broadened and rebinned model
            wave[i,:,:] = m_spec_i.wave
            flux[i,:,:] = m_spec_i.flux

            if get_contr:
                # Integrate the emission contribution function and cloud opacity
                self.get_contr_em_and_opa_cloud(
                    self.atm[i], m_wave_i=wave_i, 
                    d_wave_i=self.d_wave[i,:], 
                    d_mask_i=self.d_mask_isfinite[i], 
                    m_spec_i=m_spec_i, 
                    order=i
                    )

        # Create a new ModelSpectrum instance with all orders
        m_spec = ModelSpectrum(
            wave=wave, flux=flux, 
            lbl_opacity_sampling=self.lbl_opacity_sampling, 
            multiple_orders=True, 
            high_pass_filtered=self.apply_high_pass_filter, 
            )
        
        # Adopt the pixel masking
        m_spec.mask_isfinite = self.d_mask_isfinite

        if self.apply_high_pass_filter:
            # High-pass filter the model spectrum
            m_spec.high_pass_filter(replace_flux_err=True)

        # Clear up some memory, same attributes in DataSpectrum
        del m_spec.wave, m_spec.mask_isfinite

        return m_spec

    def get_contr_em_and_opa_cloud(
            self, atm_i, m_wave_i, d_wave_i, d_mask_i, m_spec_i, order
            ):
        
        # Get the emission contribution function
        contr_em_i = atm_i.contr_em
        new_contr_em_i = []

        # Get the cloud opacity
        if self.cloud_mode == 'gray':
            opa_cloud_i = self.Cloud.get_opacity(m_wave_i*1e-3, self.pressure).T
        elif self.cloud_mode == 'EddySed':
            #opacity_shape = (1, atm_i.freq_len, 1, len(atm_i.press))
            #opa_cloud_i = atm_i.cloud_total_opa_retrieval_check.reshape(opacity_shape)
            #opa_cloud_i = np.nansum(opa_cloud_i.T, axis=(1,3))
            opa_cloud_i = atm_i.cloud_total_opa_retrieval_check.T
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
