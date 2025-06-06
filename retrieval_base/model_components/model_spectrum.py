import numpy as np
from ..utils import sc

def get_class(**kwargs):
    """
    Get the pRT class.

    Args:
        **kwargs: Additional keyword arguments.

    Returns:
        pRT: Instance of the pRT class.
    """
    return pRT(**kwargs)
    
class pRT:
    """
    Class for pRT model spectrum.
    """
    def __init__(self, ParamTable, d_spec, m_set, pressure, evaluation=False, is_first_m_set=True):
        """
        Initialize the pRT class.

        Args:
            ParamTable (dict): Parameter table.
            d_spec (object): Data spectrum object.
            m_set (str): Model set.
            pressure (array): Pressure array.
            evaluation (bool): Flag for evaluation mode.
        """
        self.m_set    = m_set
        self.pressure = pressure

        self.evaluation = evaluation

        # Set some info from the data
        self.d_wave                  = d_spec.wave
        self.d_resolution            = d_spec.resolution
        self.d_wave_ranges_chips     = d_spec.wave_ranges_chips
        self.instrumental_broadening = d_spec.instrumental_broadening

        lbl_opacity_sampling = ParamTable.pRT_Radtrans_kwargs[self.m_set].get(
            'line_by_line_opacity_sampling', 1.
            )
        median_wave = np.nanmedian(self.d_wave)
        if median_wave < 1e3:
            self.m_resolution = 1e6 / lbl_opacity_sampling
        else:
            self.m_resolution = 1/(0.01*(median_wave*1e-7)) / lbl_opacity_sampling

        # Flags to determine if line opacities should be interpolated for each m_set
        self.is_first_m_set = is_first_m_set
        self.shared_line_opacities = ParamTable.pRT_Radtrans_kwargs[self.m_set].get(
            'shared_line_opacities', False
            )
        ParamTable.pRT_Radtrans_kwargs[self.m_set].pop('shared_line_opacities', None)

        # Update the input data path
        ParamTable = self._setup_input_data_path(ParamTable)

        # Set the wavelength ranges of the model
        self._set_wave_ranges(ParamTable)

        # Create Radtrans objects
        self._setup_Radtrans(**ParamTable.pRT_Radtrans_kwargs[self.m_set])

    def __call__(self, ParamTable, Chem, PT, Cloud, Rotation, LineOpacity=None, **kwargs):
        """
        Run the radiative transfer to obtain a model spectrum.

        Args:
            ParamTable (dict): Parameter table.
            Chem (object): Chemistry object.
            PT (object): PT profile object.
            Cloud (object): Cloud object.
            Rotation (object): Rotation object.
            LineOpacity (list): List of line opacity objects.
            **kwargs: Additional keyword arguments.
        """
        # Get the custom opacity functions
        abs_opacity  = self._set_absorption_opacity(Cloud, LineOpacity)
        scat_opacity = self._set_scattering_opacity(Cloud)

        # Update the incidence angles
        self._setup_incidence_angles(Rotation)

        # Update the pRT call kwargs
        pRT_call_kwargs = {
            'temperatures': PT.temperature, 
            'mass_fractions': Chem.mass_fractions, 
            'mean_molar_masses': Chem.mass_fractions['MMW'], 

            'reference_gravity': ParamTable.get('g'), 

            'additional_absorption_opacities_function': abs_opacity, 
            'additional_scattering_opacities_function': scat_opacity, 

            'eddy_diffusion_coefficients': getattr(Cloud, 'K_zz', None),
            'cloud_f_sed': getattr(Cloud, 'f_sed', None),
            'cloud_particle_radius_distribution_std': getattr(Cloud, 'sigma_g', None),

            'return_cloud_contribution': self.evaluation and hasattr(Cloud, 'cloud_species'),
            'return_opacities': self.evaluation and hasattr(Cloud, 'cloud_species'),
            'return_contribution': self.evaluation, 
            'frequencies_to_wavelengths': True,
            'fast_opacity_interpolation': hasattr(self.atm[0], '_interpolate_species_opacities_fast'),
            #'fast_opacity_interpolation': False,
        }
        if hasattr(Rotation, 'unique_mu_included'):
            pRT_call_kwargs['return_per_mu'] = True
        
        self.wave, self.flux, self.flux_binned = [], [], []
        self.contr_per_wave, self.integrated_contr = [], []

        # Loop over all chips
        for i, atm_i in enumerate(self.atm):
            
            # Skip if no incidence angles are set
            if len(atm_i._emission_angle_grid['cos_angles']) == 0:
                self.wave.append(np.zeros_like(atm_i.frequencies))
                self.flux.append(np.zeros_like(atm_i.frequencies))
                continue

            # Compute the emission spectrum
            wave_init_i, flux_i, additional_outputs = atm_i.calculate_flux(**pRT_call_kwargs)

            # Convert to the right units
            wave_init_i = wave_init_i[None,:] * 1e7 # [cm] -> [nm]

            # Get the flux per incidence angle or integrated
            if flux_i.ndim == 1:
                flux_i = flux_i[None,:] # [erg cm^{-2} s^{-1} cm^{-1}]
            flux_i *= 1e-7 # [erg cm^{-2} s^{-1} nm^{-1}]

            # Convert to observation
            wave_i, flux_i, flux_binned_i = self._convert_to_observation(
                ParamTable, Rotation, wave_init_i.copy(), flux_i, self.d_wave[i]
                )

            if self.evaluation:
                # Get the shifted/broadened + integrated emission contribution
                contr_per_wave_i, integrated_contr_i = self._get_emission_contribution(
                    ParamTable, Rotation, wave_init_i.copy(), 
                    additional_outputs['emission_contribution'], 
                    self.d_wave[i], flux_binned_i, 
                    )
                self.contr_per_wave.append(contr_per_wave_i)
                self.integrated_contr.append(integrated_contr_i)

                # Store the cloud opacity
                if 'cloud_opacities' in additional_outputs:
                    _, integrated_cloud_opacity_i = self._get_cloud_opacities(
                        wave_init_i.copy()[0], np.squeeze(additional_outputs['cloud_opacities']).T, 
                        self.d_wave[i], flux_binned_i
                        )
                    Cloud.total_opacity += integrated_cloud_opacity_i
                                
            # Store the model spectrum
            self.wave.append(wave_i)
            self.flux.append(flux_i)
            self.flux_binned.append(flux_binned_i)

        self.flux_binned = np.array(self.flux_binned)

        # Add a blackbody to the atmospheric emission
        self._add_blackbody_emission(ParamTable)

        if self.evaluation:
            self.contr_per_wave   = np.array(self.contr_per_wave)
            self.integrated_contr = np.array(self.integrated_contr)

    def read_line_opacities_from_other_m_spec(self, source_m_spec):
        """
        Read the line opacities from another model spectrum.

        Args:
            source_m_spec (object): Source model spectrum object.
        """
        if not hasattr(self.atm[0], '_interpolate_species_opacities_fast'):
            # Shared opacities are not implemented in default pRT installations
            return
            
        #if not self.shared_line_opacities:
        if not getattr(self, 'shared_line_opacities', False):
            # Interpolate line opacities, no need to store
            # (default in Radtrans._interpolate_species_opacities_fast)
            return
        
        if self.is_first_m_set:
            # Shared, but first model setting so store the interpolated line opacities
            [setattr(atm_i, 'store_line_opacities_as_attr', True) for atm_i in self.atm]
            return        
        
        # Line opacities are shared by source m_spec            
        for i, source_atm_i in enumerate(source_m_spec.atm):
            # Copy the line opacities
            self.atm[i].line_opacities = source_atm_i.line_opacities
            # No need to interpolate
            setattr(self.atm[i], 'interpolate_line_opacities_flag', False)

    def combine_model_settings(self, *other_m_spec, sum_model_settings=False):
        """
        Combine the model settings.

        Args:
            *other_m_spec: Other model spectrum objects.
            sum_model_settings (bool): Flag to sum model settings.

        Returns:
            tuple: Combined wavelength, flux, and binned flux arrays.
        """
        if sum_model_settings:
            # Combine the fluxes of all model settings
            wave = self.wave.copy()
            flux = self.flux.copy()
            flux_binned = self.flux_binned.copy()

            for m_spec in list(other_m_spec):
                
                # Loop over all chips (flux is a list)
                for i, (wave_other_i, flux_other_i) in enumerate(zip(m_spec.wave, m_spec.flux)):
                    # Interpolate to the same wavelengths (flux_binned already is)
                    flux_other_i = np.interp(wave[i], xp=wave_other_i, fp=flux_other_i)
                    flux[i] += flux_other_i

                # flux_binned is an array
                flux_binned += m_spec.flux_binned

            return wave, flux, flux_binned
        
        # Consider all model settings separately
        wave = []
        flux = []
        flux_binned = []
        for m_spec in [self, *other_m_spec]:
            # Loop over all chips
            for i in range(len(m_spec.wave)):
                wave.append(m_spec.wave[i])
                flux.append(m_spec.flux[i])
                flux_binned.append(m_spec.flux_binned[i])
        return wave, flux, flux_binned
    
    def _set_wave_ranges(self, ParamTable):
        """
        Set the wavelength ranges for the model.

        Args:
            ParamTable (dict): Parameter table.
        """
        rv_max = 1001
        if not self.evaluation:
            rv_prior    = ParamTable.get('rv', key='Param').prior_params
            vsini_prior = ParamTable.get('rv', key='Param').prior_params
            
            rv_max = np.array(list(rv_prior))
            rv_max += np.array([-1,1]) * max(vsini_prior)
            rv_max = np.max(np.abs(rv_max))

        wave_pad = 1.1 * rv_max/(sc.c*1e-3) * np.max(self.d_wave_ranges_chips)
        
        # Wavelength ranges of model
        self.wave_ranges = np.array([
            [wave_min-wave_pad, wave_max+wave_pad] \
            for (wave_min, wave_max) in self.d_wave_ranges_chips
            ])
                
    def _setup_input_data_path(self, ParamTable):
        """
        Setup the input data path for the model.

        Args:
            ParamTable (dict): Parameter table.
        
        Returns:
            dict: Updated parameter table.
        """
        from petitRADTRANS.config import petitradtrans_config_parser

        old_input_data_path = petitradtrans_config_parser.get_input_data_path()
        new_input_data_path = ParamTable.pRT_Radtrans_kwargs[self.m_set].get(
            'pRT_input_data_path', old_input_data_path
            )
        if new_input_data_path != old_input_data_path:
            # Set the new input_data path
            petitradtrans_config_parser.set_input_data_path(new_input_data_path)

        # Remove pRT_input_data_path from kwargs if it exists
        ParamTable.pRT_Radtrans_kwargs[self.m_set].pop('pRT_input_data_path', None)
        return ParamTable

    def _setup_Radtrans(self, **pRT_Radtrans_kwargs):
        """
        Setup the Radtrans objects for the model.

        Args:
            **pRT_Radtrans_kwargs: Additional keyword arguments.
        """
        from petitRADTRANS.radtrans import Radtrans

        self.atm = []
        for wave_range_i in self.wave_ranges:
            # Make a Radtrans object per chip
            atm_i = Radtrans(
                pressures=self.pressure, 
                wavelength_boundaries=wave_range_i*1e-3, 
                **pRT_Radtrans_kwargs.copy()
                )
            
            #if not self.shared_line_opacities:
            if not getattr(self, 'shared_line_opacities', False):
                self.atm.append(atm_i)
                continue

            # If shared, interpolate the line opacities onto 
            # the new PT grid, but only for the first model setting
            if self.is_first_m_set:
                # Interpolate onto (new) PT grid
                setattr(atm_i, 'store_line_opacities_as_attr', True)
                setattr(atm_i, 'interpolate_line_opacities_flag', True)
            else:
                # Do not interpolate, but use the already interpolated opacities
                #setattr(atm_i, 'store_line_opacities_as_attr', False)
                setattr(atm_i, 'interpolate_line_opacities_flag', False)
                
            self.atm.append(atm_i)

    def _setup_incidence_angles(self, Rotation):
        """
        Setup the incidence angles for the model.

        Args:
            Rotation (object): Rotation object.
        """
        mu  = getattr(Rotation, 'unique_mu_included', None)
        dmu = getattr(Rotation, 'unique_dmu_included', None)
        if mu is None:
            # Do not update
            return
        
        for atm_i in self.atm:
            # Update the incidence angles to compute
            atm_i._emission_angle_grid['cos_angles'] = mu
            atm_i._emission_angle_grid['weights']    = dmu
            
    def _set_absorption_opacity(self, Cloud, LineOpacity=None):
        """
        Set the absorption opacity for the model.

        Args:
            Cloud (object): Cloud object.
            LineOpacity (list): List of line opacity objects.

        Returns:
            function: Absorption opacity function.
        """
        # Check if custom opacities are set
        cloud_abs_opacity = getattr(Cloud, 'abs_opacity', None)
        
        if LineOpacity is not None:
            def line_abs_opacity(wave_micron, pressure):
                opacity = 0
                for LineOpacity_i in LineOpacity:
                    opacity += LineOpacity_i.abs_opacity(wave_micron, pressure)
                return opacity
        else:
            line_abs_opacity = None

        if (not cloud_abs_opacity) and (not line_abs_opacity):
            # No custom opacities
            return None
        
        if cloud_abs_opacity and (not line_abs_opacity):
            # Only cloud opacity
            return cloud_abs_opacity
        
        if (not cloud_abs_opacity) and line_abs_opacity:
            # Only line opacity
            return line_abs_opacity
        
        # Both cloud and line opacity
        return lambda wave_micron, pressure: \
            cloud_abs_opacity(wave_micron, pressure) + \
            line_abs_opacity(wave_micron, pressure)
    
    def _set_scattering_opacity(self, Cloud):
        """
        Set the scattering opacity for the model.

        Args:
            Cloud (object): Cloud object.

        Returns:
            function: Scattering opacity function.
        """
        # Check if custom opacities are set
        cloud_scat_opacity = getattr(Cloud, 'scat_opacity', None)
        return cloud_scat_opacity

    def _convert_to_observation(self, ParamTable, Rotation, wave, flux, d_wave, apply_scaling=True):
        """
        Convert the model spectrum to observation.

        Args:
            ParamTable (dict): Parameter table.
            Rotation (object): Rotation object.
            wave (array): Wavelength array.
            flux (array): Flux array.
            d_wave (array): Data wavelength array.
            apply_scaling (bool): Flag to apply scaling.

        Returns:
            tuple: Wavelength, flux, and binned flux arrays.
        """
        # Apply rotational broadening
        wave, flux = Rotation.broaden(wave, flux)
        
        # Apply radial-velocity shift
        wave *= (1 + ParamTable.get('rv')/(sc.c*1e-3))

        if apply_scaling:
            # Convert to observation by scaling with the radius
            flux *= (ParamTable.get('R_p',1.)*sc.r_jup_mean / (ParamTable.get('distance')*sc.parsec))**2

            # Apply coverage fraction for this model setting
            flux *= ParamTable.get('coverage_fraction', 1.)

        # Apply instrumental broadening
        flux = self.instrumental_broadening(
            wave, flux, resolution=self.d_resolution, initial_resolution=self.m_resolution
            )
                
        # Rebin to data wavelength grid
        flux_binned = np.interp(d_wave, xp=wave, fp=flux)

        return wave, flux, flux_binned
    
    def _get_emission_contribution(self, ParamTable, Rotation, wave, contr, d_wave, flux_binned):
        """
        Get the emission contribution for the model.

        Args:
            ParamTable (dict): Parameter table.
            Rotation (object): Rotation object.
            wave (array): Wavelength array.
            contr (array): Contribution array.
            d_wave (array): Data wavelength array.
            flux_binned (array): Binned flux array.

        Returns:
            tuple: Contribution per wavelength and integrated contribution arrays.
        """
        contr_per_wave   = np.nan * np.ones((len(self.pressure), len(d_wave)))
        integrated_contr = np.nan * np.ones_like(self.pressure)
        
        # Loop over each pressure level
        for i, contr_i in enumerate(contr):

            # Apply rv-shift and broadening to emission contribution
            wave_i, contr_i, contr_binned_i = self._convert_to_observation(
                ParamTable, Rotation, wave.copy(), contr_i[None,:], d_wave, apply_scaling=False, 
                )
            contr_per_wave[i] = contr_binned_i

            # Spectrally weighted integration
            integrand1 = np.trapz(x=d_wave, y=d_wave*flux_binned*contr_binned_i)
            integrand2 = np.trapz(x=d_wave, y=d_wave*flux_binned)
            integrated_contr[i] = integrand1 / integrand2

        return contr_per_wave, integrated_contr

    def _get_cloud_opacities(self, wave, cloud_opacities, d_wave, flux_binned):

        opacity_per_wave   = np.nan * np.ones((len(self.pressure), len(d_wave)))
        integrated_opacity = np.nan * np.ones_like(self.pressure)

        # Loop over each pressure level
        for i, cloud_opacities_i in enumerate(cloud_opacities):
            # Interpolate onto data wavelength grid
            cloud_opacities_binned_i = np.interp(d_wave, xp=wave, fp=cloud_opacities_i)
            opacity_per_wave[i] = cloud_opacities_binned_i

            # Spectrally weighted integration
            integrand1 = np.trapz(x=d_wave, y=d_wave*flux_binned*cloud_opacities_binned_i)
            integrand2 = np.trapz(x=d_wave, y=d_wave*flux_binned)
            integrated_opacity[i] = integrand1 / integrand2

        return opacity_per_wave, integrated_opacity

    def _add_blackbody_emission(self, ParamTable):
        """
        Add emission from a blackbody.

        Args:
            ParamTable (dict): Parameter table.

        Returns:
            array: Blackbody emission.
        """
        # Check if blackbody should be added
        T_BB = ParamTable.get('T_BB', None)
        if T_BB is None:
            return

        # Area of emission, defaults to planet
        R_BB = ParamTable.get('R_BB', ParamTable.get('R_p',1.))*sc.r_jup_mean

        # Loop over all chips
        self.flux_BB = []
        for i, wave_i in enumerate(self.d_wave):

            # Planck's law
            flux_BB_i = 2*sc.h*sc.c**2/(wave_i*1e-9)**5 * 1/(np.exp(sc.h*sc.c/((wave_i*1e-9)*sc.k*T_BB))-1)
            flux_BB_i *= np.pi*1e-6 # [J s^-1 m^-2 m^-1 sr^-1] -> [W m^-2 um^-1]=[erg s^-1 cm^-2 nm^-1]
            
            # Convert to observation
            flux_BB_i *= (R_BB / (ParamTable.get('distance')*sc.parsec))**2

            # Add to the model spectrum
            self.flux_binned[i] += flux_BB_i
            self.flux_BB.append(flux_BB_i)
