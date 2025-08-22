import numpy as np

def get_class(pressure, cloud_mode=None, **kwargs):
    """
    Factory function to get the appropriate Cloud class based on the cloud_mode.

    Args:
        pressure (np.ndarray): Pressure levels.
        cloud_mode (str, optional): Mode of the cloud. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Cloud: An instance of a Cloud subclass.
    """
    if cloud_mode in [None, 'none']:
        return Cloud(pressure, **kwargs)
    if cloud_mode == 'gray':
        return Gray(pressure, **kwargs)
    if cloud_mode == 'EddySed':
        return EddySed(pressure, **kwargs)
    else:
        raise ValueError(f'Cloud mode {cloud_mode} not recognized.')
    
class Cloud:
    """
    Base class for handling cloud models.
    """
    def __init__(self, pressure, **kwargs):
        """
        Initialize the Cloud class.

        Args:
            pressure (np.ndarray): Pressure levels.
            **kwargs: Additional keyword arguments.
        """
        self.pressure = pressure

    def __call__(self, ParamTable, **kwargs):
        """
        Evaluate the cloud model with given parameters.

        Args:
            ParamTable (dict): Parameters for the model.
            **kwargs: Additional keyword arguments.
        """
        return
    
class EddySed(Cloud):
    """
    Class for handling EddySed cloud models.
    """
    def __init__(self, pressure, cloud_species=['Mg2SiO4(s)_crystalline__DHS'], **kwargs):
        """
        Initialize the EddySed class.

        Args:
            pressure (np.ndarray): Pressure levels.
            cloud_species (list, optional): List of cloud species. Defaults to ['Mg2SiO4(s)_crystalline__DHS'].
            **kwargs: Additional keyword arguments.
        """
        # Give arguments to parent class
        super().__init__(pressure)

        self.cloud_species = cloud_species

    def __call__(self, ParamTable, Chem, PT, **kwargs):
        """
        Evaluate the EddySed cloud model with given parameters.

        Args:
            ParamTable (dict): Parameters for the model.
            Chem (Chemistry): Chemistry object.
            PT (PressureTemperature): Pressure-Temperature profile.
            **kwargs: Additional keyword arguments.
        """
        # Mixing and particle size
        self.K_zz    = ParamTable.get('K_zz') * np.ones_like(self.pressure)
        self.sigma_g = ParamTable.get('sigma_g')
        self.f_sed = {}

        self.mass_fractions = {}
        for species in self.cloud_species:
            self.mass_fractions[species] = \
                self._get_mass_fraction_profile(ParamTable, Chem, PT, species)

        self.total_opacity = 0 # Is updated in model_spectrum.pRT.__call__

    def _get_cloud_base(self, Chem, PT, species):
        """
        Get the cloud base pressure and equilibrium mass fraction.

        Args:
            Chem (Chemistry): Chemistry object.
            PT (PressureTemperature): Pressure-Temperature profile.
            species (str): Cloud species.

        Returns:
            tuple: Base pressure and equilibrium mass fraction.
        """
        from petitRADTRANS.chemistry import clouds

        # Intersection between condensation curve and PT profile
        P_base = clouds.simple_cdf(
            name=species, press=self.pressure, temp=PT.temperature, 
            metallicity=Chem.FeH, co_ratio=Chem.CO, 
            mmw=np.nanmean(Chem.mass_fractions['MMW']), 
            )

        # Equilibrium-chemistry mass fraction
        mf_eq = clouds.return_cloud_mass_fraction(
            name=species, metallicity=Chem.FeH, co_ratio=Chem.CO
            )

        return P_base, mf_eq

    def _get_mass_fraction_profile(self, ParamTable, Chem, PT, species):
        """
        Get the mass fraction profile for a cloud species.

        Args:
            ParamTable (dict): Parameters for the model.
            Chem (Chemistry): Chemistry object.
            PT (PressureTemperature): Pressure-Temperature profile.
            species (str): Cloud species.
        """
        P_base = ParamTable.get(f'P_base_{species}') # Base pressure
        mf_eq  = ParamTable.get(f'X_base_{species}')  # Equilibrium mass fraction

        if None in [P_base, mf_eq]:
            # Get cloud base and equilibrium mass fraction
            P_base, mf_eq = self._get_cloud_base(Chem, PT, species)

        # Scaling factor for mass fraction profile
        X = ParamTable.get(f'X_{species}', 1.)
        
        # Power-law drop-off
        f_sed = ParamTable.get(f'f_sed_{species}')
        if f_sed is None:
            raise ValueError(f'f_sed_{species} not found in ParamTable.')

        # Pressures above the cloud base
        mask_above_base = (self.pressure <= P_base)

        # Mass fraction at each layer
        Chem.mass_fractions[species] = np.zeros_like(self.pressure)
        Chem.mass_fractions[species][mask_above_base] = \
            X * mf_eq * (self.pressure[mask_above_base]/P_base)**f_sed

        self.f_sed[species] = f_sed

        return Chem.mass_fractions[species].copy()

class Gray(Cloud):
    """
    Class for handling Gray cloud models.
    """
    def __init__(self, pressure, **kwargs):
        """
        Initialize the Gray class.

        Args:
            pressure (np.ndarray): Pressure levels.
            **kwargs: Additional keyword arguments.
        """
        # Give arguments to parent class
        super().__init__(pressure)

        # Anchor point for non-gray power-law (1 um)
        self.wave_cloud_0 = kwargs.get('wave_cloud_0', 1.)
        # Maximum number of cloud layers
        self.n_clouds_max = kwargs.get('n_clouds_max', 10)

    def __call__(self, ParamTable, mean_wave_micron=None, **kwargs):
        """
        Evaluate the Gray cloud model with given parameters.

        Args:
            ParamTable (dict): Parameters for the model.
            mean_wave_micron (float, optional): Mean wavelength in microns. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.P_base      = []
        self.opa_base    = []
        self.f_sed_gray  = []
        self.cloud_slope = []

        for i in [None, *range(self.n_clouds_max)]:

            suffix = f'_{i}' if i!=None else ''
            P_base_i     = ParamTable.get(f'P_base_gray{suffix}')   # Base pressure
            opa_base_i   = ParamTable.get(f'opa_base_gray{suffix}') # Opacity at the base
            f_sed_gray_i = ParamTable.get(f'f_sed_gray{suffix}')    # Power-law drop-off

            if (None in [P_base_i, opa_base_i, f_sed_gray_i]):
                if i is not None:
                    break
                continue

            # Non-gray cloud slope
            cloud_slope_i = ParamTable.get(f'cloud_slope{suffix}', 0.)

            self.P_base.append(P_base_i)
            self.opa_base.append(opa_base_i)
            self.f_sed_gray.append(f_sed_gray_i)
            self.cloud_slope.append(cloud_slope_i)

        # Single-scattering albedo
        self.omega = ParamTable.get('omega', 0.)

        if mean_wave_micron is None:
            return
        self.total_opacity = self.abs_opacity(mean_wave_micron, self.pressure) + \
            self.scat_opacity(mean_wave_micron, self.pressure)
        self.total_opacity = np.squeeze(self.total_opacity)

    def abs_opacity(self, wave_micron, pressure):
        """
        Calculate the absorption opacity.

        Args:
            wave_micron (np.ndarray): Wavelengths in microns.
            pressure (np.ndarray): Pressure levels.

        Returns:
            np.ndarray: Absorption opacity.
        """
        wave_micron = np.atleast_1d(wave_micron)

        # Create gray cloud opacity, i.e. independent of wavelength
        opacity = np.zeros((len(wave_micron), len(pressure)), dtype=np.float64)

        # Loop over multiple cloud layers
        iterables = zip(
            self.P_base, 
            self.opa_base, 
            self.f_sed_gray, 
            self.cloud_slope, 
            )
        for P_base_i, opa_base_i, f_sed_gray_i, cloud_slope_i in iterables:

            if f_sed_gray_i is None:
                # No opacity change with altitude, assume deck below P_base_i
                mask_P = (pressure >= P_base_i)
                slope_pressure = 1.
            else:
                # Pressures above the cloud base
                mask_P = (pressure <= P_base_i)
                slope_pressure = (pressure[mask_P]/P_base_i)**f_sed_gray_i

            # Non-gray cloud model
            #slope_wave = 1 / (1+(wave_micron/self.wave_cloud_0)**cloud_slope_i)
            slope_wave = (wave_micron/self.wave_cloud_0)**cloud_slope_i

            # Opacity decreases with power-law above the base
            opacity[:,mask_P] += opa_base_i * slope_wave[:,None] * slope_pressure

        return opacity * (1-self.omega)

    def scat_opacity(self, wave_micron, pressure):
        """
        Calculate the scattering opacity.

        Args:
            wave_micron (np.ndarray): Wavelengths in microns.
            pressure (np.ndarray): Pressure levels.

        Returns:
            np.ndarray: Scattering opacity.
        """
        # Total cloud opacity
        opacity = 1/(1-self.omega) * self.abs_opacity(wave_micron, pressure)
        return opacity * self.omega