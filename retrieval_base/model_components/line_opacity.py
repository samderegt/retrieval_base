import numpy as np
import pandas as pd

from scipy.special import wofz as Faddeeva

from ..utils import sc

def get_class(m_spec, line_opacity_kwargs, **kwargs):
    """
    Returns a list of LineOpacity instances.

    Args:
        m_spec (object): Spectral model object.
        line_opacity_kwargs (dict): Dictionary of line opacity keyword arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        list: List of LineOpacity instances.
    """
    if len(line_opacity_kwargs) == 0:
        # No species to include as a custom opacity
        return None
    
    all_LineOpacity = []
    for line_species, line_opacity_kwargs_species in line_opacity_kwargs.items():
        LineOpacity_i = LineOpacity(
            m_spec=m_spec, line_species=line_species, **line_opacity_kwargs_species, **kwargs
            )
        all_LineOpacity.append(LineOpacity_i)

    return all_LineOpacity

class LineData:
    """
    Class for handling line data.
    """
    
    def __init__(self, **kwargs):
        """
        Initializes the LineData class.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self._load_NIST_states(kwargs['states_file'])
        
        if kwargs.get('states_file') is not None:
            # Use (some) transitions from the Kurucz database
            self._load_Kurucz_transitions(kwargs['transitions_file'])

        if kwargs.get('custom_transitions') is not None:
            # Add the custom transitions
            self._load_custom_transitions(kwargs['custom_transitions'])

        if not hasattr(self, 'nu_0'):
            raise ValueError('No transitions loaded')
        
        self.is_onthefly = getattr(
            self, 'is_onthefly', np.zeros_like(self.nu_0, dtype=bool)
            )
        self.has_impact_parameters = self.is_onthefly.copy()

        # Mask transitions that are not relevant
        self._mask_transitions(**kwargs)

        # Derive additional (PT-independent) parameters
        self.gf = 10**self.log_gf
        self.E_high = self.E_low + self.nu_0

        # Natural broadening
        mask = (self.log_gamma_N == 0.)
        self.gamma_N       = 10**self.log_gamma_N / (4*np.pi*(sc.c*1e2))      # [cm^-1]
        self.gamma_N[mask] = 0.22 * self.nu_0[mask]**2 / (4*np.pi*(sc.c*1e2)) # [cm^-1]

        print(f'\n--- {kwargs['line_species']} -------------')
        print(f'Loaded {len(self.nu_0)} transitions, {self.is_onthefly.sum()} will be calculated on-the-fly')
        if self.is_onthefly.any():
            print('On-the-fly transitions:')
            print('nu_0 (cm^-1):', (self.nu_0[self.is_onthefly]).round(3))
            print('wave_0 (nm): ', (1e7/self.nu_0[self.is_onthefly]).round(3))
        print()

    def get_partition_function(self, T):
        """
        Calculates the partition function.

        Args:
            T (array): Temperature array.

        Returns:
            array: Partition function values.
        """
        # Partition function per temperature
        Q = np.sum(
            self.g[None,:] * np.exp(-(sc.c2*1e2)*self.E[None,:]/T[:,None]), 
            axis=1, keepdims=True
            )
        return Q
    
    def _load_NIST_states(self, states_file):
        """
        Loads NIST states from a file.

        Args:
            states_file (str): Path to the states file.
        """
        # Load data from the NIST states
        d = np.loadtxt(states_file, dtype=str, skiprows=1, usecols=(0,2))

        g, E = [], []
        for g_i, E_i in d:

            if g_i == '""':
                continue
            g.append(float(g_i))
            E.append(float(E_i.replace('"', '')))
        
        self.g = np.array(g)
        self.E = np.array(E) # [cm^-1]

    def _load_Kurucz_transitions(self, trans_file):
        """
        Loads Kurucz transitions from a file.

        Args:
            trans_file (str): Path to the transitions file.
        """
        d = np.array(
            pd.read_fwf(trans_file, widths=(11,7,6,12,5,1,10,12,5,1,10,6,6,6), header=None)
            )
        self.log_gf = d[:,1].astype(np.float64)

        E_low  = d[:,3].astype(np.float64) # [cm^-1]
        E_high = d[:,7].astype(np.float64) # [cm^-1]

        self.nu_0 = np.abs(E_high - E_low) # [cm^-1]

        # Kurucz line lists are sorted by parity, not energy
        self.E_low = np.min(np.concatenate((E_low[None,:], E_high[None,:])), axis=0)

        self.log_gamma_N   = d[:,11].astype(np.float64)
        self.log_gamma_vdW = d[:,13].astype(np.float64)
    
    def _load_custom_transitions(self, custom_transitions):
        """
        Loads custom transitions.

        Args:
            custom_transitions (list): List of custom transitions.
        """
        def replace_or_append(dictionary, key, idx):

            # Read Kurucz array (if available)
            array = getattr(self, key, np.array([]))

            # Read custom data, otherwise use the default Kurucz value
            value = dictionary.get(key)
            if value is None:
                value = array[idx]

            if idx == len(array):
                # Append the new value
                array = np.append(array, value)
            else:
                # Replace the existing value
                array[idx] = value

            # Update the attribute
            setattr(self, key, array)
        
        self.nu_0 = getattr(self, 'nu_0', np.array([]))
        indices = []
        for trans in custom_transitions:

            nu_0 = trans.get('nu_0')
            if nu_0 is None:
                raise ValueError('Custom transition must have wavenumber nu_0')

            if self.nu_0.size == 0:
                # First transition, add as new value
                idx = 1
            elif not np.isclose(self.nu_0, nu_0).any():
                # Transition does not exist, add as new value
                idx = len(self.nu_0)
            else:
                # Check if the transition already exists
                idx = np.argmin(np.abs(self.nu_0 - nu_0))
            
            # Add the custom transition
            indices.append(idx)
            for key in ['nu_0', 'log_gf', 'E_low', 'log_gamma_N', 'log_gamma_vdW']:
                replace_or_append(trans, key=key, idx=idx)

        # Whether to calculate this transition on-the-fly
        self.is_onthefly = np.isin(np.arange(len(self.nu_0)), indices)

    def _mask_transitions(self, **kwargs):
        """
        Masks transitions based on given criteria.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        # Select transitions that affect modelled wavelengths
        self.line_cutoff = kwargs.get('line_cutoff', 60) # [cm^-1]
        mask = (self.nu_0 > self.nu_ranges.min()-self.line_cutoff) & \
            (self.nu_0 < self.nu_ranges.max()+self.line_cutoff)
        
        # Only consider moderate lines with a certain strength
        mask &= (self.log_gf >= kwargs.get('log_gf_cutoff', -np.inf))

        # Apply the mask
        self.nu_0   = self.nu_0[mask]
        self.log_gf = self.log_gf[mask]
        self.E_low  = self.E_low[mask]
        self.log_gamma_N   = self.log_gamma_N[mask]
        self.log_gamma_vdW = self.log_gamma_vdW[mask]
        self.is_onthefly   = self.is_onthefly[mask]
        self.has_impact_parameters = self.has_impact_parameters[mask]

        # Select strong transitions to calculate on-the-fly
        mask_exact = (self.log_gf >= kwargs.get('log_gf_cutoff_exact', np.inf))
        self.is_onthefly[mask_exact] = True

class LineOpacity(LineData):

    def __init__(self, m_spec, **kwargs):
        """
        Initializes the LineOpacity class.

        Args:
            m_spec (object): Spectral model object.
            **kwargs: Arbitrary keyword arguments.
        """
        # Define the wavenumber and PT grids
        self._set_grid(m_spec)

        # Give arguments to the parent class
        super().__init__(**kwargs)
        
        self.kwargs = kwargs
        self.mass   = kwargs['mass'] * (sc.amu*1e3) # Atomic mass [g]
        self.line_species = kwargs['line_species']  # pRT line species name

        # Precompute the opacity table for non-"on-the-fly" transitions
        self.status = 'pre-compute'
        self._get_precomputed_opacity_table()
        self.status = 'combine'

    def __call__(self, ParamTable, PT, Chem):
        """
        Calculates the line opacity.

        Args:
            ParamTable (dict): Parameter table.
            PT (object): PT profile object.
            Chem (object): Chemistry profile object.
        """
        self._update_parameters(
            T=PT.temperature, 
            VMRs=Chem.VMRs, 
            mass_fractions=Chem.mass_fractions, 
            ParamTable=ParamTable, 
            )
        
    def abs_opacity(self, wave_micron, pressure):
        """
        Calculates the absorption opacity.

        Args:
            wave_micron (array): Wavelength array in microns.
            pressure (array): Pressure array.

        Returns:
            array: Absorption opacity values.
        """
        # Determine which chip is being computed
        idx_chip = np.argmin(np.abs(
            np.mean(wave_micron) - np.mean(self.wave_ranges*1e-3, axis=1)
            ))
        nu = 1e4/wave_micron

        # Create line opacity array
        opacity = 1e-250 * np.ones((len(wave_micron), len(pressure)), dtype=np.float64)

        # Loop over atmospheric layers
        for i in range(opacity.shape[1]):

            # Loop over all transitions
            for j in range(len(self.nu_0)):

                if (self.status == 'pre-compute') and self.is_onthefly[j]:
                    # Ignore the on-the-fly transitions
                    continue
                elif (self.status == 'combine') and (not self.is_onthefly[j]):
                    # Ignore all non-"on-the-fly" transitions
                    continue

                nu_0_ij      = self.nu_0[j].copy()
                gamma_vdW_ij = self.gamma_vdW[i,j].copy()
                if self.has_impact_parameters[j]:
                    
                    # Find the impact width/shift index
                    idx_impact = np.sum(self.has_impact_parameters[:j])

                    # Shift the line core
                    nu_0_ij += self.d[i,idx_impact]
                    
                    if self.w[i,idx_impact] != 0.:
                        # Use the impact width
                        gamma_vdW_ij = self.w[i,idx_impact]
                    
                # Lorentz-wing cutoff
                mask_nu_j = np.abs(nu-nu_0_ij) < self.line_cutoff
                nu_j = nu[mask_nu_j]

                # Calculate the line profile
                opacity[mask_nu_j,i] += self._line_profile(
                    nu=nu_j, nu_0=nu_0_ij, gamma_N=self.gamma_N[j], gamma_vdW=gamma_vdW_ij, 
                    gamma_G=self.gamma_G[i,j], S=self.S[i,j]
                    )
                
            # Skip the interpolation step
            if self.status != 'combine':
                continue
            if not hasattr(self, 'interp_table'):
                continue

            # Combine the interpolated and on-the-fly opacities
            opacity[:,i] += 10**self.interp_table[idx_chip,i](self.temperature[i])

        # Scale the opacity by the abundance profile
        opacity *= self.mass_fraction[None,:]
        return opacity # [cm^2 g^-1]

    def _set_grid(self, m_spec):
        """
        Sets the wavenumber and PT grids.

        Args:
            m_spec (object): Spectral model object.
        """
        # Define the wavenumber grid
        self.wave_ranges = m_spec.wave_ranges   # [nm]
        self.nu_ranges   = 1e7/self.wave_ranges # [cm^-1]
        self.nu_grid     = [atm_i._frequencies/(sc.c*1e2) for atm_i in m_spec.atm] # [cm^-1]

        # Define the PT grid (use pRT default temperatures)
        self.P_grid = m_spec.pressure
        '''
        self.T_grid = np.array([
            81.14113604736988, 
            109.60677358237457, 
            148.05862230132453, 
            200., 
            270.163273706, 
            364.940972297, 
            492.968238926, 
            665.909566306, 
            899.521542126, 
            1215.08842295, 
            1641.36133093, 
            2217.17775249, 
            2995., 
            ])
        '''
        self.T_grid = np.array([
            100,200,300,400,500,600,700,800,900,1000,1200,1400,
            1600,1800,2000,2500,3000,3500,4000,4500,5000], dtype=np.float64
            )

    def _get_gamma_vdW(self):
        """
        Calculates the van der Waals broadening coefficient.

        Returns:
            array: Van der Waals broadening coefficients.
        """
        mask = (self.log_gamma_vdW == 0.)

        # Provided van-der-Waals broadening coefficients (Sharp & Burrows 2007)
        gamma_vdW = 10**self.log_gamma_vdW[None,:] / (4*np.pi*(sc.c*1e2)) * \
            self.n_density[:,None] * (self.temperature[:,None]/10000)**(3/10) # [cm^-1]
        gamma_vdW[:,mask] = np.nan

        if not self.kwargs.get('is_alkali', False):
            return gamma_vdW

        # For alkali atoms
        amu = sc.amu*1e3 # [g]
        red_mass_H_X  = (1.00784*amu)*self.mass / ((1.00784*amu)+self.mass)
        red_mass_H2_X = (2.01568*amu)*self.mass / ((2.01568*amu)+self.mass)
        red_mass_He_X = (4.002602*amu)*self.mass / ((4.002602*amu)+self.mass)

        alpha_H  = 0.666793 # 10^{-24} cm^3
        alpha_H2 = 0.806    # 10^{-24} cm^3
        alpha_He = 0.204956 # 10^{-24} cm^3

        # Eq. 23 from Sharp & Burrows (2007)
        gamma_vdW = 10**self.log_gamma_vdW[None,:] / (4*np.pi*(sc.c*1e2)) * \
            self.n_density[:,None] * (self.temperature[:,None]/10000)**(3/10) * (
                self.VMR_H2[:,None] * (red_mass_H_X/red_mass_H2_X)**(3/10) * (alpha_H2/alpha_H)**(2/5) + \
                self.VMR_He[:,None] * (red_mass_H_X/red_mass_He_X)**(3/10) * (alpha_He/alpha_H)**(2/5)
                )

        # Schweitzer et al. (1995) [cm^6 s^-1]
        E_H = self.kwargs.get('E_H', 13.6*8065.73) # [cm^-1]
        Z   = self.kwargs.get('Z', 1.)
        E_ion = self.kwargs['E_ion'] # [cm^-1]

        C_6_H2 = 1.01e-32 * alpha_H2/alpha_H * (Z+1)**2 * (
            (E_H/(E_ion-self.E_low))**2 - (E_H/(E_ion-self.E_high))**2
            )
        C_6_H2 = np.abs(C_6_H2)[None,:]

        C_6_He = 1.01e-32 * alpha_He/alpha_H * (Z+1)**2 * (
            (E_H/(E_ion-self.E_low))**2 - (E_H/(E_ion-self.E_high))**2
            )
        C_6_He = np.abs(C_6_He)[None,:]

        gamma_vdW[:,mask] = 1.664461/(2*(sc.c*1e2)) * self.n_density[:,None] * \
            ((sc.k*1e7)*self.temperature[:,None])**(3/10) * (
                self.VMR_H2[:,None] * (1/red_mass_H2_X)**(3/10) * C_6_H2[:,mask]**(2/5) + \
                self.VMR_He[:,None] * (1/red_mass_He_X)**(3/10) * C_6_He[:,mask]**(2/5)
                )
        
        return gamma_vdW

    def _get_gamma_G(self):
        """
        Calculates the thermal broadening coefficient.

        Returns:
            array: Thermal broadening coefficients.
        """
        # Thermal broadening coefficient
        gamma_G = self.nu_0[None,:]/(sc.c*1e2) * \
            np.sqrt((2*(sc.k*1e7)*self.temperature[:,None])/self.mass)
        return gamma_G
    
    def _get_oscillator_strength(self):
        """
        Calculates the oscillator strength.

        Returns:
            array: Oscillator strengths.
        """
        # Partition function
        Q = self.get_partition_function(T=self.temperature)

        e = 4.80320425e-10 # [cm^3/2 g^1/2 s^-1]

        term1 = (self.gf[None,:]*np.pi*e**2) / ((sc.m_e*1e3)*(sc.c*1e2)**2)
        term2 = np.exp(-(sc.c2*1e2)*self.E_low[None,:] / self.temperature[:,None]) / Q
        term3 = (1 - np.exp(-(sc.c2*1e2)*self.nu_0[None,:]/self.temperature[:,None]))
        
        S = term1 * term2 * term3 # [cm^1]
        return S
    
    def _get_impact_width_shift(self, ParamTable={}):
        """
        Calculates the impact width and shift.

        Args:
            ParamTable (dict, optional): Parameter table. Defaults to {}.

        Returns:
            tuple: Impact width and shift.
        """
        # Power-law for the impact width/shift
        w = np.zeros((len(self.P_grid), self.has_impact_parameters.sum()), dtype=np.float64)
        d = w.copy()

        # Reference density
        density_ref = self.kwargs.get('n_ref', 1e20) # [cm^-3]

        # Separate coefficients per perturber
        perturber_density = {
            '_H2': self.VMR_H2*self.n_density, '_He': self.VMR_He*self.n_density, '':self.n_density
            }
        for suffix, density in perturber_density.items():

            for i in range(w.shape[1]):
                # Read the parameters
                A_w_i = ParamTable.get(f'A_w_{i}{suffix}', None)
                b_w_i = ParamTable.get(f'b_w_{i}{suffix}', None)

                A_d_i = ParamTable.get(f'A_d_{i}{suffix}', None)
                b_d_i = ParamTable.get(f'b_d_{i}{suffix}', None)

                if None in [A_w_i, b_w_i, A_d_i, b_d_i]:
                    # Keep width/shift at 0
                    continue

                # Scale the impact width and shift with (P,T)
                w[:,i] += A_w_i * self.temperature**b_w_i * density/density_ref
                d[:,i] += A_d_i * self.temperature**b_d_i * density/density_ref

        return w, d

    def _update_parameters(self, T, VMRs={}, mass_fractions={}, ParamTable={}):
        """
        Updates the line opacity parameters.

        Args:
            T (float): Temperature.
            VMRs (dict, optional): Volume mixing ratios. Defaults to {}.
            mass_fractions (dict, optional): Mass fractions. Defaults to {}.
            ParamTable (dict, optional): Parameter table. Defaults to {}.
        """
        self.temperature = np.ones_like(self.P_grid) * T
        self.n_density   = self.P_grid*1e6 / ((sc.k*1e7)*self.temperature) # [cm^-3]

        self.VMR_H2 = VMRs.get('H2', 0.85*np.ones_like(self.P_grid))
        self.VMR_He = VMRs.get('He', 0.15*np.ones_like(self.P_grid))
        self.mass_fraction = mass_fractions.get(self.line_species, 1.*np.ones_like(self.P_grid))

        # Update the line opacity parameters
        self.gamma_vdW = self._get_gamma_vdW()
        self.gamma_G   = self._get_gamma_G()
        self.S = self._get_oscillator_strength()
        
        self.w, self.d = self._get_impact_width_shift(ParamTable=ParamTable)

    def _get_precomputed_opacity_table(self):
        """Precomputes the opacity table."""
        # Table of interpolation functions
        from scipy.interpolate import interp1d
        self.interp_table = np.empty((len(self.nu_grid),len(self.P_grid)), dtype=object)

        # Loop over chips
        for idx_order, nu in enumerate(self.nu_grid):
            
            opacity_of_order = np.ones(
                (len(self.T_grid), len(nu), len(self.P_grid)), dtype=np.float64
                )

            # Loop over temperature
            for idx_T, T in enumerate(self.T_grid):

                # Update the PT-dependent parameters
                self._update_parameters(T)

                # Calculate the opacity
                wave_micron = 1e4/nu
                opacity_of_order[idx_T,:,:] = self.abs_opacity(wave_micron, self.P_grid)

            # Set up interpolation functions
            for idx_P in range(len(self.P_grid)):

                self.interp_table[idx_order,idx_P] = interp1d(
                    self.T_grid, np.log10(opacity_of_order[:,:,idx_P]), axis=0, 
                    kind='linear', assume_sorted=True, bounds_error=False, 
                    fill_value=(
                        np.log10(opacity_of_order[0,idx_order,idx_P]), 
                        np.log10(opacity_of_order[-1,idx_order,idx_P])
                        )
                    )

    def _line_profile(self, nu, nu_0, gamma_N, gamma_vdW, gamma_G, S):
        """
        Calculates the line profile.

        Args:
            nu (array): Wavenumber array.
            nu_0 (float): Line center wavenumber.
            gamma_N (float): Natural broadening coefficient.
            gamma_vdW (float): Van der Waals broadening coefficient.
            gamma_G (float): Thermal broadening coefficient.
            S (float): Oscillator strength.

        Returns:
            array: Line profile values.
        """
        # Gandhi et al. (2020b)
        u = (nu - nu_0) / gamma_G
        a = (gamma_vdW + gamma_N) / gamma_G

        # Voigt profile
        f = Faddeeva(u + 1j*a).real / (gamma_G*np.sqrt(np.pi))  # [cm]
        return f*S/self.mass # [cm^2 g^-1]
        