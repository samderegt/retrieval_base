import numpy as np

from petitRADTRANS.retrieval import cloud_cond as fc

def get_Cloud_class(mode=None, **kwargs):

    if mode is None:
        return Cloud(**kwargs)
    if mode == 'gray':
        return Gray(**kwargs)
    if mode == 'EddySed':
        return EddySed(**kwargs)

class Cloud:
    def __init__(self, pressure, **kwargs):
        
        # If left at None, don't include a cloud
        self.f_sed   = None
        self.K_zz    = None
        self.sigma_g = None

        self.get_opacity = None

        self.pressure = pressure

    def __call__(self, params, mass_fractions, **kwargs):
        return mass_fractions
        
class EddySed(Cloud):

    def __init__(self, pressure, cloud_species=['MgSiO3(c)'], **kwargs):

        # Initialize parent class
        super().__init__(pressure)

        self.cloud_species = []
        for species_i in cloud_species:
            
            species_i = species_i.replace('_am','')
            species_i = species_i.replace('_ad','')
            species_i = species_i.replace('_cm','')
            species_i = species_i.replace('_cd','')
            
            self.cloud_species.append(species_i)

    def _get_cloud_base(self, temperature, CO, FeH, MMW, species_i='MgSiO3(c)'):

        # Intersection between the condensation curve and PT profile
        P_base = fc.simple_cdf(
            name=species_i, 
            press=self.pressure, 
            temp=temperature, 
            FeH=FeH, 
            CO=CO, 
            MMW=MMW
            )
        
        # Equilibrium-chemistry mass fraction
        X_eq = fc.return_cloud_mass_fraction(
            name=species_i, FeH=FeH, CO=CO
            )
        
        return P_base, X_eq

    def __call__(self, params, mass_fractions, temperature=None, CO=0.59, FeH=0.0, **kwargs):

        # Mixing + particle-size
        self.K_zz    = params.get('K_zz')
        self.sigma_g = params.get('sigma_g')
        self.f_sed = {}

        for species_i in self.cloud_species:

            # Power-law drop-off
            f_sed_i  = params.get(f'f_sed_{species_i}')

            if None in [X_i, f_sed_i, self.K_zz, self.sigma_g]:
                # Some parameter is un-defined
                return
            
            # Cloud-base pressure and abundance are given
            P_base_i = params.get(f'P_base_{species_i}')
            X_base_i = params.get(f'X_base_{species_i}')

            if (P_base_i is None) and (X_base_i is None):
                # Scaling of equilibrium abundance
                X_i = params.get(f'X_{species_i}')
                
                # Get the cloud-base pressure and eq.-chem. abundance
                P_base_i, X_eq_i = self._get_cloud_base(
                    temperature=temperature, CO=CO, FeH=FeH, 
                    MMW=np.nanmean(mass_fractions['MMW']), 
                    species_i=species_i
                    )
                X_base_i = X_i * X_eq_i # Mass fraction at base

            # Pressures above the cloud base
            mask_above_base = (self.pressure <= P_base_i)

            # Abundances at each layer
            mass_fractions[species_i] = np.zeros_like(self.pressure)
            mass_fractions[species_i][mask_above_base] = \
                X_base_i * (self.pressure[mask_above_base]/P_base_i)**f_sed_i

            # Specific drop-off for species_i
            self.f_sed[species_i] = f_sed_i

        # Extend over all layers
        self.K_zz *= np.ones_like(self.pressure)

        return mass_fractions

class Gray(Cloud):

    def __init__(self, pressure, **kwargs):
        
        # Initialize parent class
        super().__init__(pressure)

        # Overwrite attribute to the cloud-opacity function
        self.get_opacity = self.cloud_opacity

    def cloud_opacity(self, wave_micron, pressure):
        '''
        Function to be called by petitRADTRANS. 

        Input
        -----
        wave_micron: np.ndarrayv
            Wavelength in micron.
        pressure: np.ndarray
            Pressure in bar.

        Output
        ------
        opa_gray_cloud: np.ndarray
            Gray cloud opacity for each wavelength and pressure layer.
        '''
        
        # Create gray cloud opacity, i.e. independent of wavelength
        opacity = np.zeros((len(wave_micron), len(pressure)), dtype=np.float64)

        # Loop over multiple clouds
        iterables = zip(
            self.P_base, self.opa_base, self.f_sed_gray, self.cloud_slope
            )
        for P_base_i, opa_base_i, f_sed_gray_i, cloud_slope_i in iterables:

            if f_sed_gray_i == 0.:
                # No opacity change with altitude, assume deck below P_base_i
                mask_P = (pressure >= P_base_i)
                slope_pressure = 1.
            else:
                # Pressures above the cloud base
                mask_P = (pressure <= P_base_i)
                slope_pressure = (pressure[mask_P]/P_base_i)**f_sed_gray_i

            # Non-gray cloud model
            slope_wave = 1 / (1+(wave_micron/self.wave_cloud_0)**cloud_slope_i)

            # Opacity decreases with power-law above the base
            opacity[:,mask_P] += opa_base_i * slope_wave[:,None] * slope_pressure

        return opacity
    
    def __call__(self, params, mass_fractions,**kwargs):
        
        self.P_base     = [params.get('P_base_gray', np.nan), ]   # Base pressure
        self.opa_base   = [params.get('opa_base_gray', np.nan), ] # Opacity at the base
        self.f_sed_gray = [params.get('f_sed_gray', 0.), ]        # Power-law drop-off
        
        # Parameters for a non-gray cloud
        self.cloud_slope  = [params.get('cloud_slope', 0.), ]
        self.wave_cloud_0 = params.get('wave_cloud_0', 1.) # Anchor at 1 um

        idx = 0
        while True:
            # Add multiple clouds            
            P_base_i     = params.get(f'P_base_gray_{idx}')
            opa_base_i   = params.get(f'opa_base_gray_{idx}')
            if None in [P_base_i, opa_base_i]:
                break

            self.P_base.append(P_base_i)
            self.opa_base.append(opa_base_i)
            self.f_sed_gray.append(params.get(f'f_sed_gray_{idx}', 0.))
            self.cloud_slope.append(params.get(f'cloud_slope_{idx}', 0.))

            idx += 1

        mask_isnan = (
            np.isnan(self.P_base) | np.isnan(self.opa_base)
            )
        self.P_base      = np.array(self.P_base)[~mask_isnan]
        self.opa_base    = np.array(self.opa_base)[~mask_isnan]
        self.f_sed_gray  = np.array(self.f_sed_gray)[~mask_isnan]
        self.cloud_slope = np.array(self.cloud_slope)[~mask_isnan]
        
        return mass_fractions