import numpy as np
import petitRADTRANS.nat_cst as nc

from scipy.special import wofz as Faddeeva

class InterpolateOpacity:

    #T_grid = np.arange(0,5000+1e-6,100)
    #'''
    T_grid = np.array([
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
    #'''
    @classmethod
    def get_opa(cls, x):
        
        i, T_i, Line_i, P_grid_i, wave_i, T_grid_i = x
        if (i%10 == 0) or i == len(T_grid_i)-1:
            print(f'T-point: {i+1}/{len(T_grid_i)}')

        # Update the temperature and density arrays
        Line_i.temperature = np.ones_like(P_grid_i) * T_i
        Line_i.n_density   = P_grid_i*1e6 / (nc.kB*T_i)

        return Line_i.get_line_opacity(wave_i, P_grid_i)

    def __init__(
            self, LineOpacity, wave_micron, order_indices, 
            ):

        # Define the (P,T)-grid
        self.P_grid = LineOpacity.pressure

        self.T_grid[self.T_grid == 0.] = 1.
        
        self.log_P_grid = np.log10(self.P_grid)
        self.log_T_grid = np.log10(self.T_grid)

        self.wave_micron = wave_micron
        self.order_indices = order_indices
        
        # Set up the rectangular (P,T)-grid
        self.setup_grid(LineOpacity, order_indices)

    def setup_grid(self, LineOpacity, order_indices):

        # Set mass-fraction to 1, so un-scaled opacity is returned
        LineOpacity.mf = np.ones_like(self.P_grid)

        print('Setting up opacity table...')
        iterable = [
            (i, T_i, LineOpacity, self.P_grid, self.wave_micron, self.T_grid) \
            for i, T_i in enumerate(self.T_grid)
            ]
        opacity = np.array(list(map(self.get_opa, iterable))) # (T, wave, P)
        opacity = opacity.swapaxes(1,2) # (T, P, wave)
        opacity = opacity.swapaxes(0,1) # (P, T, wave)

        from scipy.interpolate import interp1d
        # Create an interp-function for each pressure + each order
        self.func = np.empty((len(self.P_grid),order_indices.max()+1), dtype=object)

        # Loop over pressure
        for i, opa_i in enumerate(opacity):
            # Loop over orders
            for j in range(self.func.shape[1]):
                # Select only pixels within order
                opa_ij = opa_i[:, (order_indices == j)]
                
                # Create an interpolation function
                self.func[i,j] = interp1d(
                    self.T_grid, np.log10(opa_ij), axis=0, kind='linear', 
                    assume_sorted=True, bounds_error=False, 
                    fill_value=(np.log10(opa_ij[0]), np.log10(opa_ij[-1]))
                    )

    def __call__(self, idx_P, idx_order, T):

        return 10**self.func[idx_P,idx_order](T)
        
class LineOpacity:

    c2 = 1.4387769 # cm K

    # H2, Schweitzer et al. (1996)
    alpha_H = 0.666793
    alpha_p = 0.806
    mass_p  = 2.016*nc.amu

    E_H = 13.6*8065.73, 
    VMR_H2 = 0.85
    VMR_He = 0.15

    def __init__(
            self, 
            pressure, 
            wave_range_micron, 
            NIST_states_file, 
            VALD_trans_file=None, 
            pRT_name='', 
            mass=1.0, 
            is_alkali=False, 
            E_ion=35009.8140, Z=1, # Potassium (K I)
            line_cutoff=4500, 
            n_density_ref=1e20, 
            log_gf_threshold=-np.inf, 
            log_gf_threshold_exact=-0.5, 
            nu_0=None, 
            log_gf=None, 
            E_low=None, 
            gamma_N=None, 
            gamma_vdW=None, 
            pre_compute=True, 
            wave_micron_broad=None, 
            order_indices=None, 
            **kwargs
            ):
        
        # Define atmospheric layers
        self.pressure = pressure
        
        self.wave_range_micron = wave_range_micron
        self.nu_range = (
            1e4/np.max(self.wave_range_micron)-line_cutoff, 
            1e4/np.min(self.wave_range_micron)+line_cutoff
            )

        self.mass = mass * nc.amu # [amu] to [g]
        self.pRT_name  = pRT_name  # Index correct abundance
        self.is_alkali = is_alkali
        self.E_ion = E_ion # [cm^-1]
        self.Z = Z

        # Lines with custom power-law broadening/shift
        self.nu_0      = np.reshape([nu_0], -1)
        self.log_gf    = np.reshape([log_gf], -1)
        self.E_low     = np.reshape([E_low], -1)
        self.gamma_N   = np.reshape([gamma_N], -1)
        self.gamma_vdW = np.reshape([gamma_vdW], -1)

        self.log_gf_threshold = log_gf_threshold
        self.log_gf_threshold_exact = log_gf_threshold_exact

        # Line cutoff [cm^-1]
        self.line_cutoff = line_cutoff

        # Reference number density [cm^-3]
        self.n_density_ref = n_density_ref

        self._load_states(NIST_states_file)
        self._load_transitions(VALD_trans_file)
        
        # Calculate opacities for all lines
        self.lines_to_skip = None

        if pre_compute:
            # Skip the custom lines, which are calculated exactly
            self.lines_to_skip = 'custom'
            
            # Pre-compute opacity table (per order) for interpolation
            self.Interp = InterpolateOpacity(
                LineOpacity=self, wave_micron=wave_micron_broad, 
                order_indices=order_indices
                )

            self.lines_to_skip = 'non-custom'

    def _load_transitions(self, file):

        self.idx_custom = np.arange(len(self.nu_0))

        if self.nu_0[0] is None:
            # No custom lines, clear arrays
            self.nu_0      = np.array([], dtype=np.float64)
            self.log_gf    = np.array([], dtype=np.float64)
            self.E_low     = np.array([], dtype=np.float64)
            self.gamma_N   = np.array([], dtype=np.float64)
            self.gamma_vdW = np.array([], dtype=np.float64)

            self.idx_custom = np.array([])
        
        if file is None:
            # Only use the custom lines
            return
        
        '''
        # Load data from the VALD transitions
        d = np.genfromtxt(
            file, delimiter=',', dtype=float, skip_header=2, 
            usecols=(1,2,3,4,5,6), invalid_raise=False
            )
        # Select transitions that affect modelled wavelengths
        mask_nu_range = (d[:,0] >= self.nu_range[0]) & (d[:,0] < self.nu_range[1])
        
        # Energies in transition
        self.nu_0   = np.concatenate((self.nu_0, d[mask_nu_range,0]))
        self.E_low  = np.concatenate((self.E_low, d[mask_nu_range,1]))
        
        # Oscillator strength
        self.log_gf = np.concatenate((self.log_gf, d[mask_nu_range,2]))

        # Natural broadening + vdW broadening
        self.gamma_N   = np.concatenate((self.gamma_N, d[mask_nu_range,3]))
        self.gamma_vdW = np.concatenate((self.gamma_vdW, d[mask_nu_range,5]))
        '''

        from pandas import read_fwf
        d = read_fwf(
            file, widths=(11,7,6,12,5,1,10,12,5,1,10,6,6,6), header=None, 
            )
        d = np.array(d)

        # Oscillator strength
        self.log_gf = np.concatenate((self.log_gf, d[:,1].astype(np.float64)))
        
        E_low  = d[:,3].astype(np.float64)
        E_high = d[:,7].astype(np.float64)

        # Kurucz line list are sorted by parity?
        self.nu_0 = np.concatenate((self.nu_0, np.abs(E_high-E_low)))
        
        # Energies in transition
        E_low = np.min(np.concatenate((E_low[None,:],E_high[None,:])), axis=0)
        self.E_low = np.concatenate((self.E_low, E_low))

        # Natural broadening + vdW broadening
        self.gamma_N   = np.concatenate((self.gamma_N, d[:,11].astype(np.float64)))
        self.gamma_vdW = np.concatenate((self.gamma_vdW, d[:,13].astype(np.float64)))

        # Select transitions that affect modelled wavelengths
        mask_nu_range = (self.nu_0 >= self.nu_range[0]) & \
            (self.nu_0 < self.nu_range[1])
        self.log_gf = self.log_gf[mask_nu_range]
        self.E_low  = self.E_low[mask_nu_range]
        self.nu_0   = self.nu_0[mask_nu_range]
        self.gamma_N   = self.gamma_N[mask_nu_range]
        self.gamma_vdW = self.gamma_vdW[mask_nu_range]

        for i in self.idx_custom:
            # Find matching lines and remove duplicates
            nu_0_i = self.nu_0[i]
            match_nu_0 = np.isclose(self.nu_0, nu_0_i)
            match_nu_0[i] = False # Keep first occurence

            print(self.nu_0[match_nu_0], nu_0_i)

            # Remove duplicates
            self.nu_0      = self.nu_0[~match_nu_0]
            self.E_low     = self.E_low[~match_nu_0]
            self.log_gf    = self.log_gf[~match_nu_0]
            self.gamma_N   = self.gamma_N[~match_nu_0]
            self.gamma_vdW = self.gamma_vdW[~match_nu_0]

        # Only consider the strong-ish lines
        mask_log_gf = (self.log_gf >= self.log_gf_threshold)

        self.nu_0      = self.nu_0[mask_log_gf]
        self.E_low     = self.E_low[mask_log_gf]
        self.log_gf    = self.log_gf[mask_log_gf]
        self.gamma_N   = self.gamma_N[mask_log_gf]
        self.gamma_vdW = self.gamma_vdW[mask_log_gf]


        self.gf = 10**self.log_gf
        self.E_high = self.E_low + self.nu_0

        # Replace missing natural broadening
        valid_gamma_N = (self.gamma_N != 0.0)
        self.gamma_N[~valid_gamma_N] = \
            0.22 * self.nu_0[~valid_gamma_N]**2/(4*np.pi*nc.c)
        # Use provided coefficient
        self.gamma_N[valid_gamma_N] = \
            10**self.gamma_N[valid_gamma_N]/(4*np.pi*nc.c)
        
        # Use exact treatment for strongest lines
        self.idx_strong = np.argwhere(
            self.log_gf >= self.log_gf_threshold_exact
            ).flatten()
        
        print('Strong lines:')
        print(1e7/self.nu_0[self.idx_strong])
        print(self.nu_0[self.idx_strong])

    def _load_states(self, file):

        # Load data from the NIST states
        d = np.loadtxt(file, dtype=str, skiprows=1, usecols=(0,2))

        g, E = [], []
        for g_i, E_i in d:

            if g_i == '""':
                continue
            g.append(float(g_i))
            E.append(float(E_i.replace('"', '')))
        
        self.g = np.array(g)
        self.E = np.array(E)

    def _partition_function(self):
        # Partition function per temperature
        Q = np.sum(
            self.g[None,:] * np.exp(
                -self.c2*self.E[None,:]/self.temperature[:,None]
                ), 
            axis=-1, keepdims=True
            )
        return Q

    def _cutoff_mask(self, nu_0):
        return np.abs(self.nu-nu_0) < self.line_cutoff
    
    def _oscillator_strength(self):

        # Partition function
        Q = self._partition_function()

        term1 = (self.gf[None,:]*np.pi*nc.e**2) / (nc.m_elec*nc.c**2)
        term2 = np.exp(-self.c2*self.E_low[None,:] / self.temperature[:,None]) / Q
        term3 = (1 - np.exp(-self.c2*self.nu_0[None,:]/self.temperature[:,None]))

        # Oscillator strengths (T,trans)
        S = term1 * term2 * term3 # [cm^1]
        return S
    
    def _get_gamma_G(self):

        # Thermal broadening coefficient (T,trans)
        gamma_G = self.nu_0[None,:]/nc.c * \
            np.sqrt((2*nc.kB*self.temperature[:,None])/self.mass)
        return gamma_G

    def _get_gamma_L(self):

        # Valid coefficients
        valid_gamma_vdW = (self.gamma_vdW != 0.0)
        
        # Provided vdW broadening coefficients (Sharp & Burrows 2007) (T/P,trans)
        gamma_vdW_PT = 10**self.gamma_vdW[None,:]/(4*np.pi*nc.c) * \
            (self.n_density[:,None]) * (self.temperature[:,None]/10000)**(3/10)
        gamma_vdW_PT[:,~valid_gamma_vdW] = np.nan

        if self.is_alkali:
            red_mass_H_X  = (1.00784*nc.amu)*self.mass / ((1.00784*nc.amu)+self.mass)
            red_mass_H2_X = (2.01568*nc.amu)*self.mass / ((2.01568*nc.amu)+self.mass)
            red_mass_He_X = (4.002602*nc.amu)*self.mass / ((4.002602*nc.amu)+self.mass)

            alpha_H  = 0.666793 # 10^{-24} cm^3
            alpha_H2 = 0.806    # 10^{-24} cm^3
            alpha_He = 0.204956 # 10^{-24} cm^3

            # Eq. 23 (Sharp & Burrows 2007)
            gamma_vdW_PT = 10**self.gamma_vdW[None,:]/(4*np.pi*nc.c) * \
                (self.n_density[:,None]) * (self.temperature[:,None]/10000)**(3/10) * (
                    self.VMR_H2 * (red_mass_H_X/red_mass_H2_X)**(3/10) * (alpha_H2/alpha_H)**(2/5) + \
                    self.VMR_He * (red_mass_H_X/red_mass_He_X)**(3/10) * (alpha_He/alpha_H)**(2/5)
                )

            # Schweitzer et al. (1995) [cm^6 s^-1]
            C_6_H2 = 1.01e-32 * alpha_H2/alpha_H * (self.Z+1)**2 * (
                (self.E_H / (self.E_ion-self.E_low))**2 - \
                (self.E_H / (self.E_ion-self.E_high))**2
                )
            C_6_H2 = np.abs(C_6_H2)[None,~valid_gamma_vdW]

            C_6_He = 1.01e-32 * alpha_He/alpha_H * (self.Z+1)**2 * (
                (self.E_H / (self.E_ion-self.E_low))**2 - \
                (self.E_H / (self.E_ion-self.E_high))**2
                )
            C_6_He = np.abs(C_6_He)[None,~valid_gamma_vdW]
            
            gamma_vdW_PT[:,~valid_gamma_vdW] = 1.664461/(2*nc.c) * self.n_density[:,None] * \
                (nc.kB*self.temperature[:,None])**(3/10) * (
                    self.VMR_H2 * (1/red_mass_H2_X)**(3/10) * C_6_H2**(2/5) + \
                    self.VMR_He * (1/red_mass_He_X)**(3/10) * C_6_He**(2/5)
                )
                        
            #C_6 = 1.01e-32 * self.alpha_p/self.alpha_H * (self.Z+1)**2 * \
            #    ((self.E_H / (self.E_ion-self.E_low))**2 - \
            #     (self.E_H / (self.E_ion-self.E_high))**2)
            #C_6 = np.abs(C_6)[None,~valid_gamma_vdW]

            # vdW broadening (T/P,trans)
            #gamma_vdW_PT[:,~valid_gamma_vdW] = 1.664461/(2*nc.c) * \
            #    (nc.kB*self.temperature[:,None] * (1/self.mass+1/self.mass_p))**(3/10) * \
            #    C_6**(2/5) * self.n_density[:,None]
            
            # Update mask, all transitions are valid
            #valid_gamma_vdW = np.ones_like(valid_gamma_vdW)

        if self.lines_to_skip != 'custom':
            for i in self.idx_custom:
                if (self.w[:,i] == 0.).all():
                    # Use equation above instead
                    continue
                # Replace with custom impact-width
                gamma_vdW_PT[:,i] = self.w[:,i]

        # Combine natural- and pressure-broadening
        gamma_L = self.gamma_N[None,:] + gamma_vdW_PT
        return gamma_L

    def _impact_width_shift(self):

        # Power law treatment of impact-width and -shift (T/P,trans)
        self.w = np.zeros((len(self.pressure),len(self.idx_custom)), dtype=np.float64)
        self.d = np.zeros((len(self.pressure),len(self.idx_custom)), dtype=np.float64)

        # Separate coefficients per perturber?
        iterables = zip(
            ['H2', 'He', ''], 
            [self.VMR_H2*self.n_density, self.VMR_He*self.n_density, self.n_density]
            )
        for perturber, perturber_density in iterables:

            # Access specific power-law coefficients
            suffix = f'_{perturber}'
            if perturber == '':
                suffix = ''

            for i in range(len(self.idx_custom)):
                
                # Read the parameters
                A_w_i = self.params.get(f'A_w_{i}{suffix}')
                b_w_i = self.params.get(f'b_w_{i}{suffix}')
                A_d_i = self.params.get(f'A_d_{i}{suffix}')
                b_d_i = self.params.get(f'b_d_{i}{suffix}')

                if None in [A_w_i, b_w_i, A_d_i, b_d_i]:
                    # Keep width and shift at 0
                    continue

                # Scale the impact width and shift with (P,T)
                self.w[:,i] += A_w_i * self.temperature**b_w_i * \
                    perturber_density/self.n_density_ref
                self.d[:,i] += A_d_i * self.temperature**b_d_i * \
                    perturber_density/self.n_density_ref

    def _line_profile(self, nu_0, gamma_L, gamma_G, mask_nu, S):
        
        # Gandhi et al. (2020b)
        u = (self.nu[mask_nu] - nu_0) / gamma_G
        a = gamma_L / gamma_G

        # Scale the line profile with the oscillator strength and mass
        f  = Faddeeva(u + 1j*a).real / (gamma_G*np.sqrt(np.pi))  # [cm]
        f *= S/self.mass # [cm^2 g^-1]
        return f

    def get_line_opacity(self, wave_micron, pressure):

        # Convert to centimeter, get wavenumber
        wave_cm = 1e-4*wave_micron
        self.nu = 1/wave_cm

        if hasattr(self, 'Interp'):
            
            idx_order = np.argwhere(
                (wave_micron.mean() > self.wave_range_micron[:,0]) & \
                (wave_micron.mean() < self.wave_range_micron[:,1])
            ).flatten()[0]
            mask_order = (self.Interp.order_indices==idx_order)
            mask_wave = (
                (self.Interp.wave_micron[mask_order] >= wave_micron.min()-1e-10) & \
                (self.Interp.wave_micron[mask_order] <= wave_micron.max()+1e-10)
            )

        if self.lines_to_skip != 'custom':
            # Update the impact widths and shifts
            self._impact_width_shift()

        # Update broadening coefficients (layer,trans)
        gamma_L = self._get_gamma_L()
        gamma_G = self._get_gamma_G()

        # Update line strengths (layer,trans)
        S = self._oscillator_strength()

        # Create line-opacity array
        opacity = 1e-250 * np.ones((len(wave_micron), len(pressure)), dtype=np.float64)

        # Loop over atmospheric layers
        for i in range(len(self.pressure)):

            # Loop over lines
            for j, nu_0_j in enumerate(self.nu_0):

                use_exact = (j in self.idx_custom) or (j in self.idx_strong)

                if not use_exact:
                    continue

                if (self.lines_to_skip == 'non-custom') and (not use_exact):
                    continue
                if (self.lines_to_skip == 'custom') and use_exact:
                    continue

                nu_0_ij = nu_0_j
                if j in self.idx_custom:
                    # Shift the line-centre
                    nu_0_ij = nu_0_j + self.d[i,j]

                # Lorentz-wing cutoff
                mask_nu_ij = self._cutoff_mask(nu_0_ij)

                # Get scaled line profile for this layer
                opacity[mask_nu_ij,i] += self._line_profile(
                    nu_0_ij, gamma_L=gamma_L[i,j], gamma_G=gamma_G[i,j], 
                    mask_nu=mask_nu_ij, S=S[i,j]
                    )
                
            if hasattr(self, 'Interp'):
                # Interpolate on the pre-computed grid
                opacity[:,i] += self.Interp(
                    idx_P=i, idx_order=idx_order, 
                    T=self.temperature[i]
                    )[mask_wave]
                
        # Scale the opacity by the abundance profile
        opacity *= self.mf[None,:]
        return opacity
    
    def __call__(self, params, temperature, mass_fractions):

        self.params = params

        # Update the PT profile
        self.temperature = temperature
        
        # Number density [cm^-3]
        self.n_density = self.pressure*1e6 / (nc.kB*self.temperature)

        self.VMR_H2 = np.mean(mass_fractions['H2']*mass_fractions['MMW']/2.01568)
        self.VMR_He = np.mean(mass_fractions['He']*mass_fractions['MMW']/4.002602)
        #if not hasattr(self, 'mf'):
        #    print(self.VMR_H2, self.VMR_He)

        # Mass fraction of species
        self.mf = mass_fractions[self.pRT_name].copy()