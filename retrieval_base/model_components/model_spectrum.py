import numpy as np
from ..utils import sc

def get_class(**kwargs):
    return pRT(**kwargs)
    
class pRT:
    def __init__(self, ParamTable, d_spec, m_set, pressure, evaluation=False):

        self.m_set    = m_set
        self.pressure = pressure

        # Set some info from the data
        self.d_resolution        = d_spec.resolution
        self.d_wave_ranges_chips = d_spec.wave_ranges_chips

        # Set the wavelength ranges of the model
        self.set_wave_ranges(ParamTable, evaluation)

        # Create Radtrans objects
        self.set_Radtrans(ParamTable)

    def set_wave_ranges(self, ParamTable, evaluation):
        
        rv_max = 1001
        if not evaluation:
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

    def set_Radtrans(self, ParamTable):
        
        from petitRADTRANS import Radtrans

        self.atm = []
        for wave_range_i in self.wave_ranges:
            # Make a Radtrans object per chip
            atm_i = Radtrans(
                wlen_bords_micron=wave_range_i*1e-3, 
                **ParamTable.pRT_Radtrans_kwargs[self.m_set].copy()
                )
            
            # Set up the atmospheric layers
            atm_i.setup_opa_structure(self.pressure)
            
            self.atm.append(atm_i)

    def set_absorption_opacity(self, Cloud, LineOpacity=None):
        
        # Check if custom opacities are set
        cloud_abs_opacity = getattr(Cloud, 'abs_opacity')
        
        line_abs_opacity = None
        if LineOpacity is not None:
            line_abs_opacity = getattr(LineOpacity, 'abs_opacity')

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
    
    def set_scattering_opacity(self, Cloud):
        
        # Check if custom opacities are set
        cloud_scat_opacity = getattr(Cloud, 'scat_opacity')
        return cloud_scat_opacity
    
    def set_incidence_angles(self, Rotation):
        
        mu = getattr(Rotation, 'unique_mu_included')
        if not mu:
            # Do not update
            return
        
        for i, atm_i in enumerate(self.atm):
            # Update the incidence angles to compute
            atm_i.mu = mu
            atm_i.w_gauss_mu = np.ones_like(mu) / len(mu)

            self.atm[i] = atm_i

    def __call__(self, ParamTable, Chem, PT, Cloud, Rotation, LineOpacity=None, **kwargs):

        # Get the custom opacity functions
        abs_opacity  = self.set_absorption_opacity(Cloud, LineOpacity)
        scat_opacity = self.set_scattering_opacity(Cloud)

        # Update the incidence angles
        self.set_incidence_angles(Rotation)

        # Update the pRT call kwargs
        pRT_call_kwargs = {
            'temp': PT.temperature, 
            'abunds': Chem.mass_fractions, 
            'mmw': Chem.mass_fractions['MMW'], 

            'gravity': ParamTable.get('g'), 

            'give_absorption_opacity': abs_opacity, 
            'give_scattering_opacity': scat_opacity, 

            'Kzz': getattr(Cloud, 'K_zz', None),
            'fsed': getattr(Cloud, 'f_sed', None),
            'sigma_lnorm': getattr(Cloud, 'sigma_g', None),
        }
        if hasattr(Rotation, 'unique_mu_included'):
            pRT_call_kwargs['return_per_mu'] = True

        self.wave, self.flux = [], []
        # Loop over all chips
        for atm_i in self.atm:
            
            # Skip if no incidence angles are set
            if len(atm_i.mu) == 0:
                self.wave.append(np.zeros_like(atm_i.freq))
                self.flux.append(np.zeros_like(atm_i.freq))
                continue
            
            # Compute the emission spectrum
            atm_i.calc_flux(**pRT_call_kwargs)

            # Convert to the right units
            wave_i = sc.c*1e9 / atm_i.freq[None,:]  # [nm]

            # Get the flux per incidence angle or integrated
            flux_i = getattr(atm_i, 'flux_mu', atm_i.flux[None,:]) # [erg cm^{-2} s^{-1} Hz^{-1}]
            flux_i *= sc.c*1e9 / wave_i**2 # [erg cm^{-2} s^{-1} nm^{-1}]

            # Apply rotational broadening
            wave_i, flux_i = Rotation(wave_i, flux_i)

            # Apply radial-velocity shift
            wave_i *= (1 + ParamTable.get('rv')/(sc.c*1e-3))

            # Convert to observation by scaling with the radius
            flux_i *= (ParamTable.get('R_p',1.)*sc.r_jup_mean / ParamTable.get('distance')*sc.parsec)**2

            # Store the model spectrum
            self.wave.append(wave_i)
            self.flux.append(flux_i)