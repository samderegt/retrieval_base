import numpy as np

def get_class(pressure, cloud_mode=None, **kwargs):

    if cloud_mode in [None, 'none']:
        return Cloud(pressure, **kwargs)
    if cloud_mode == 'gray':
        return Gray(pressure, **kwargs)
    if cloud_mode == 'EddySed':
        return EddySed(pressure, **kwargs)
    else:
        raise ValueError(f'Cloud mode {cloud_mode} not recognized.')
    
class Cloud:
    def __init__(self, pressure, **kwargs):
        self.pressure = pressure

    def __call__(self, ParamTable, **kwargs):
        return
    
class EddySed(Cloud):
    def __init__(self, pressure, cloud_species=['MgSiO3(c)'], **kwargs):
        # Give arguments to parent class
        super().__init__(pressure)

        self.cloud_species = []
        for species_i in cloud_species:
            # Remove crystalline-structure information
            species_i = species_i.replace('_am','')
            species_i = species_i.replace('_ad','')
            species_i = species_i.replace('_cm','')
            species_i = species_i.replace('_cd','')
            self.cloud_species.append(species_i)

    def get_cloud_base(self, Chem, PT, species):

        from petitRADTRANS.retrieval import cloud_cond

        # Intersection between condensation curve and PT profile
        P_base = cloud_cond.simple_cdf(
            name=species, 
            press=self.pressure, temp=PT.temperature, 
            FeH=Chem.FeH, CO=Chem.CO, 
            MMW=Chem.mass_fractions['MMW'], 
            )

        # Equilibrium-chemistry mass fraction
        mf_eq = cloud_cond.return_cloud_mass_fraction(
            name=species, FeH=Chem.FeH, CO=Chem.CO, 
            )

        return P_base, mf_eq

    def get_mass_fraction_profile(self, ParamTable, Chem, PT, species):

        P_base = ParamTable.get(f'P_base_{species}') # Base pressure
        mf_eq  = ParamTable.get(f'X_base_{species}')  # Equilibrium mass fraction

        if None in [P_base, mf_eq]:
            # Get cloud base and equilibrium mass fraction
            P_base, mf_eq = self.get_cloud_base(Chem, PT, species)

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

    def __call__(self, ParamTable, Chem, PT, **kwargs):

        # Mixing and particle size
        self.K_zz    = ParamTable.get('K_zz') * np.ones_like(self.pressure)
        self.sigma_g = ParamTable.get('sigma_g')
        self.f_sed = {}

        for species in self.cloud_species:
            self.get_mass_fraction_profile(ParamTable, Chem, PT, species)

class Gray(Cloud):
    def __init__(self, pressure, **kwargs):
        # Give arguments to parent class
        super().__init__(pressure)

        # Anchor point for non-gray power-law (1 um)
        self.wave_cloud_0 = kwargs.get('wave_cloud_0', 1.)

        self.n_clouds_max = kwargs.get('n_clouds_max', 10)

    def abs_opacity(self, wave_micron, pressure):

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

        # Total cloud opacity
        opacity = 1/(1-self.omega) * self.abs_opacity(wave_micron, pressure)
        return opacity * self.omega
            
    def __call__(self, ParamTable, **kwargs):
    
        self.P_base      = []
        self.opa_base    = []
        self.f_sed_gray  = []
        self.cloud_slope = []

        for i in [None, *range(self.n_clouds_max)]:

            suffix = f'_{i}' if i!=None else ''
            P_base_i     = ParamTable.get(f'P_base_gray{suffix}')   # Base pressure
            opa_base_i   = ParamTable.get(f'opa_base_gray{suffix}') # Opacity at the base
            f_sed_gray_i = ParamTable.get(f'f_sed_gray{suffix}')    # Power-law drop-off

            if (None in [P_base_i, opa_base_i, f_sed_gray_i]) and (i!=-1):
                break

            # Non-gray cloud slope
            cloud_slope_i = ParamTable.get(f'cloud_slope{suffix}', 0.)

            self.P_base.append(P_base_i)
            self.opa_base.append(opa_base_i)
            self.f_sed_gray.append(f_sed_gray_i)
            self.cloud_slope.append(cloud_slope_i)

        # Single-scattering albedo
        self.omega = ParamTable.get('omega', 0.)