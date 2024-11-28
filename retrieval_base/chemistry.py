import numpy as np
import pandas as pd

import pathlib
directory_path = pathlib.Path(__file__).parent.resolve()

import petitRADTRANS.nat_cst as nc

def get_Chemistry_class(pressure, line_species, chem_mode='free', **kwargs):

    if chem_mode == 'free':
        return FreeChemistry(line_species, pressure, **kwargs)
    if chem_mode == 'pRT_table':
        return pRTChemistryTable(line_species, pressure, **kwargs)
    if chem_mode == 'fastchem':
        return FastChemistry(line_species, pressure, **kwargs)
    if chem_mode == 'fastchem_table':
        return FastChemistryTable(line_species, pressure, **kwargs)


class Chemistry:

    species_info = pd.read_csv(directory_path/'species_info.csv', index_col=0)
    neglect_species = {key_i: False for key_i in species_info.index}

    def __init__(self, line_species, pressure, CustomOpacity=None):

        self.line_species = [*line_species, 'H2', 'He']

        # Custom line-opacities
        if CustomOpacity is not None:
            self.line_species += [Opa_i.pRT_name for Opa_i in CustomOpacity]
        
        # Store the regular name and hill-notations too
        self.species, self.hill = [], []
        for species_i in self.species_info.index:
            
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            hill_i         = self.read_species_info(species_i, 'pyfc_name')

            if line_species_i not in self.line_species:
                continue

            self.species.append(species_i)
            self.hill.append(hill_i)

        self.pressure     = pressure
        self.n_atm_layers = len(self.pressure)

        # Set to None initially, changed during evaluation
        self.mass_fractions_envelopes = None
        self.mass_fractions_posterior = None
        self.unquenched_mass_fractions_posterior = None
        self.unquenched_mass_fractions_envelopes = None

    def remove_species(self):

        # Remove the contribution of the specified species
        for species_i, remove in self.neglect_species.items():
            
            if not remove:
                continue

            # Read the name of the pRT line species
            line_species_i = self.read_species_info(species_i, 'pRT_name')

            # Set abundance to 0 to evaluate species' contribution
            if line_species_i in self.line_species:
                self.mass_fractions[line_species_i] *= 0

    @classmethod
    def read_species_info(cls, species, info_key):
        
        if info_key == 'pRT_name':
            return cls.species_info.loc[species,info_key]
        if info_key == 'pyfc_name':
            return cls.species_info.loc[species,'Hill_notation']
        
        if info_key == 'mass':
            return cls.species_info.loc[species,info_key]
        
        if info_key == 'COH':
            return list(cls.species_info.loc[species,['C','O','H']])
        
        if info_key in ['C','O','H']:
            return cls.species_info.loc[species,info_key]

        if info_key == 'c' or info_key == 'color':
            return cls.species_info.loc[species,'color']
        if info_key == 'label':
            return cls.species_info.loc[species,'mathtext_name']
        
    def get_VMRs(self, *args):
        return
    
    def get_isotope_VMRs(self, params):

        # If true, eq-chem abundances should be split into isotopologues
        conserve_tot_VMR = isinstance(self, EquilibriumChemistry)

        # All isotope ratios per species
        all_CO_ratios = {
            '12CO': 1., 
            '13CO': 1./params.get('13CO_ratio', np.inf), 
            'C18O': 1./params.get('C18O_ratio', np.inf), 
            'C17O': 1./params.get('C17O_ratio', np.inf), 
        }
        all_H2O_ratios = {
            'H2O':     1., 
            'H2(18)O': 1./params.get('H2(18)O_ratio', np.inf), 
            'H2(17)O': 1./params.get('H2(17)O_ratio', np.inf), 
            'HDO':     1./params.get('HDO_ratio', np.inf), 
        }
        all_CH4_ratios = {
            'CH4':   1., 
            '13CH4': 1./params.get('13CH4_ratio', np.inf), 
        }
        all_NH3_ratios = {
            'NH3':   1., 
            '15NH3': 1./params.get('15NH3_ratio', np.inf), 
        }

        VMRs_copy = self.VMRs.copy()
        for species_i in self.species:

            if species_i not in [*all_CO_ratios, *all_H2O_ratios, *all_CH4_ratios, *all_NH3_ratios]:
                # Not a CO, H2O, CH4 or NH3 isotopologue
                continue
            
            iterables = zip(
                [all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios], 
                ['12CO', 'H2O', 'CH4', 'NH3']
            )
            for all_ratios, main_iso_i in iterables:

                # Minor-to-main ratio
                minor_main_ratio_i = all_ratios.get(species_i)

                # Read the VMR of the main isotopologue
                main_iso_VMR_i = VMRs_copy[main_iso_i]

                sum_of_ratios = 1.
                if conserve_tot_VMR:
                    # To conserve the total abundance
                    sum_of_ratios = sum(all_ratios.values())

                if minor_main_ratio_i is not None:
                    # Matching isotope ratio found
                    break
            
            # e.g. 13CO = CO_all * 13/12C / (12/12C+13/12C+18/16O+17/16O)
            self.VMRs[species_i] = main_iso_VMR_i * minor_main_ratio_i/sum_of_ratios
    
    def quench_VMRs(self, *args):
        return
    
    def convert_to_MFs(self):

        # Convert to mass-fractions using mass-ratio
        self.mass_fractions = {'MMW': self.MMW}
        for species_i, VMR_i in self.VMRs.items():
            
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')

            self.mass_fractions[line_species_i] = VMR_i * mass_i/self.MMW
    
    def get_diagnostics(self):

        C, O, H = 0, 0, 0
        for species_i, VMR_i in self.VMRs.items():
            # Record C, O, and H bearing species for C/O and metallicity
            COH_i = self.read_species_info(species_i, 'COH')
            C += COH_i[0]*VMR_i
            O += COH_i[1]*VMR_i
            H += COH_i[2]*VMR_i

        if self.CO is None:
            self.CO = np.mean(C/O)

        if self.FeH is None:
            log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
            if (C == 0).any():
                self.FeH = 0.
            else:
                self.FeH = np.mean(np.log10(C/H) - log_CH_solar)
        
    def __call__(self, params, temperature, param_VMRs=None):

        self.temperature = temperature
        
        self.mass_fractions = {}
        self.VMRs = {}
        self.MMW  = 0.
        
        self.CO  = None
        self.FeH = None

        # Get volume-mixing ratios
        self.get_VMRs(params, param_VMRs=param_VMRs)
        if self.VMRs == -np.inf:
            # Some issue was raised
            self.mass_fractions = -np.inf
            return -np.inf

        # Quench (eq-chem) abundances
        self.quench_VMRs(params)
        # Get isotope abundances
        self.get_isotope_VMRs(params)

        if hasattr(self, 'get_H2'):
            self.get_H2() # Compute H2 abundance last (free-chem)
        if hasattr(self, 'get_MMW'):
            self.get_MMW() # Compute MMW

        if self.VMRs == -np.inf:
            # Some issue was raised
            self.mass_fractions = -np.inf
            return -np.inf

        # Convert to a mass-fraction for pRT
        self.convert_to_MFs()

        # Get some diagnostics (i.e. C/O, Fe/H)
        self.get_diagnostics()

        # Remove certain species
        self.remove_species()

        return self.mass_fractions


class FreeChemistry(Chemistry):

    def __init__(self, line_species, pressure, CustomOpacity=None, **kwargs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure, CustomOpacity)

    def _power_law_drop_off(self, VMR, P0, alpha):

        if P0 is None:
            # No drop-off
            return VMR
        
        mask_TOA = (self.pressure < P0) # Top-of-atmosphere
        if alpha is None:
            VMR[mask_TOA] = 0. # Instant drop-off
            return VMR
        
        # Power-law drop-off
        VMR[mask_TOA] *= (self.pressure[mask_TOA]/P0)**alpha
        return VMR

    def get_VMRs(self, params, param_VMRs, **kwargs):

        # Constant He abundance
        self.VMRs = {'He':0.15*np.ones(self.n_atm_layers)}

        for species_i in self.species:

            if species_i in ['H2', 'He']:
                continue # Set by other VMRs

            # Read the fitted VMR (or set to 0)
            param_VMR_i = param_VMRs.get(species_i, 0.)
            # Expand to all layers
            param_VMR_i *= np.ones(self.n_atm_layers)

            # Parameterise a power-law drop-off
            self.VMRs[species_i] = self._power_law_drop_off(
                param_VMR_i, P0=params.get(f'{species_i}_P'), 
                alpha=params.get(f'{species_i}_alpha'), 
                )
        
    def get_H2(self):
        # H2 abundance is the remainder
        VMR_wo_H2 = np.sum([VMR_i for VMR_i in self.VMRs.values()], axis=0)
        self.VMRs['H2'] = 1 - VMR_wo_H2

        if (self.VMRs['H2'] < 0).any():
            # Other species are too abundant
            self.VMRs = -np.inf

    def get_MMW(self):
        # Get mean-molecular weight from free-chem VMRs
        MMW = 0.
        for species_i, VMR_i in self.VMRs.items():
            mass_i = self.read_species_info(species_i, 'mass')
            MMW += mass_i * VMR_i
        
    def convert_to_MFs(self):

        # Get mean-molecular weight from free-chem VMRs
        MMW = 0.
        for species_i, VMR_i in self.VMRs.items():
            mass_i = self.read_species_info(species_i, 'mass')
            MMW += mass_i * VMR_i

        # Convert to mass-fractions using mass-ratio
        self.mass_fractions = {'MMW': MMW * np.ones(self.n_atm_layers)}
        for species_i, VMR_i in self.VMRs.items():
            
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')

            self.mass_fractions[line_species_i] = VMR_i * mass_i/MMW
    
      
class EquilibriumChemistry(Chemistry):
    
    def __init__(self, line_species, pressure, CustomOpacity=None):
        
        # Give arguments to the parent class
        super().__init__(line_species, pressure, CustomOpacity)

        # Species to quench per system
        self.quench_settings = {
            'CO_CH4': [('12CO','CH4','H2O'), None], 
            'N2_NH3': [('N2','NH3'), None], 
        }

    def get_P_quench_from_Kzz(self, params, alpha=1):

        # Metallicity
        m = 10**self.FeH

        # Scale height at each layer
        H = nc.kB*self.temperature / (self.MMW*nc.amu * 10**params['log_g'])

        # Mixing length/time-scales
        L = alpha * H
        t_mix = L**2 / params['Kzz_chem']

        computed = {key_q: False for key_q in self.quench_settings}

        # Loop from bottom to top of atmosphere
        idx = np.argsort(self.pressure)[::-1]
        iterables = zip(t_mix[idx], self.pressure[idx], self.temperature[idx])
        for t_mix_i, P_i, T_i in iterables:

            if T_i < 500:
                # Avoid exponent overflow
                continue

            if all(computed.values()):
                break

            # Zahnle & Marley (2014)
            if not computed.get('CO_CH4', True):
                # Chemical timescale of CO-CH4
                inv_t_q1 = 1e6/1.5 * P_i * m**0.7 * np.exp(-42000/T_i)
                inv_t_q2 = 1/40 * P_i**2 * np.exp(-25000/T_i)
                t_CO_CH4 = 1/(inv_t_q1 + inv_t_q2)

                if t_mix_i < t_CO_CH4:
                    # Mixing is more efficient than chemical reactions
                    self.quench_settings['CO_CH4'][-1] = P_i
                    computed['CO_CH4'] = True

            if not computed.get('N2_NH3', True):
                # Chemical timescale of N2-NH3
                t_NH3 = 1.0e-7 * P_i**(-1) * np.exp(52000/T_i)

                if t_mix_i < t_NH3:
                    self.quench_settings['N2_NH3'][-1] = P_i
                    computed['N2_NH3'] = True

            if not computed.get('HCN', True):
                # Chemical timescale of HCN-NH3-N2
                t_HCN = 1.5e-4 * P_i**(-1) * m**(-0.7) * np.exp(36000/T_i)

                if t_mix_i < t_HCN:
                    self.quench_settings['HCN'][-1] = P_i
                    computed['HCN'] = True

            if not computed.get('CO2', True):
                # Chemical timescale of HCN-NH3-N2
                t_CO2 = 1.0e-10 * P_i**(-0.5) * np.exp(38000/T_i)

                if t_mix_i < t_CO2:
                    self.quench_settings['CO2'][-1] = P_i
                    computed['CO2'] = True
    
    def quench_VMRs(self, params):
        
        # Update the parameters
        for key_q in list(self.quench_settings):
            # Update the quench pressures from the free parameters
            self.quench_settings[key_q][-1] = params.get(f'P_quench_{key_q}')

        if params.get('Kzz_chem') is not None:
            # Update the quench pressures from diffusivity
            self.get_P_quench_from_Kzz(params)

        log_P = np.log10(self.pressure) # Take log for convenience
        for species_q, P_q in self.quench_settings.values():

            if P_q is None:
                # No quench-pressure given
                continue
            
            # Top of the atmosphere
            mask_TOA = (self.pressure < P_q)

            for species_i in species_q:
                if species_i not in self.species:
                    continue
                
                #if (self.VMRs[species_i]==0.).any():
                #    print(species_i, self.VMRs[species_i])

                # Quench the VMRs of specific species
                log_VMR_i = np.log10(self.VMRs[species_i])
                log_VMR_i[mask_TOA] = np.interp(
                    np.log10(P_q), xp=log_P, fp=log_VMR_i
                    )
                self.VMRs[species_i] = 10**log_VMR_i

class FastChemistryTable(EquilibriumChemistry):

    def __init__(self, line_species, pressure, CustomOpacity=None, path_fastchem_tables='./fastchem_tables', **kwargs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure, CustomOpacity)

        # Load the interpolation tables
        self._load_interp_tables(pathlib.Path(path_fastchem_tables))

        # Species to quench per system
        self.quench_settings = {
            'CO_CH4': [('12CO','CH4','H2O'), None], 
            'N2_NH3': [('N2','NH3'), None], 
        }

    def _load_interp_tables(self, path_tables):

        import h5py
        def load_hdf5(file, key):
            with h5py.File(f'{path_tables}/{file}', 'r') as f:
                return f[key][...]
        
        # Load the interpolation grid (ignore N/O)
        self.P_grid = load_hdf5('grid.hdf5', 'P')
        self.T_grid = load_hdf5('grid.hdf5', 'T')
        self.CO_grid  = load_hdf5('grid.hdf5', 'C/O')
        self.FeH_grid = load_hdf5('grid.hdf5', 'Fe/H')
        points = (self.P_grid, self.T_grid, self.CO_grid, self.FeH_grid)

        from scipy.interpolate import RegularGridInterpolator
        self.interp_tables = {}
        for species_i, hill_i in zip([*self.species, 'MMW'], [*self.hill, 'MMW']):
            key = 'MMW' if species_i=='MMW' else 'log_VMR'

            if not isinstance(hill_i, str):
                # Minor isotope
                continue

            # Load the eq-chem abundance tables
            arr = load_hdf5(f'{hill_i}.hdf5', key=key)

            # Generate interpolation functions
            self.interp_tables[species_i] = RegularGridInterpolator(
                values=arr[:,:,:,0,:], points=points, method='linear', 
                #bounds_error=False, fill_value=None
                )        

    def get_VMRs(self, params, **kwargs):
        
        def apply_bounds(val, grid):
            val[val > grid.max()] = grid.max()
            val[val < grid.min()] = grid.min()
            return val

        # Update the parameters
        self.CO  = params.get('C/O')
        self.FeH = params.get('Fe/H')

        # Apply the bounds of the grid
        P = apply_bounds(self.pressure.copy(), grid=self.P_grid)
        T = apply_bounds(self.temperature.copy(), grid=self.T_grid)
        CO  = apply_bounds(np.array([self.CO]).copy(), grid=self.CO_grid)[0]
        FeH = apply_bounds(np.array([self.FeH]).copy(), grid=self.FeH_grid)[0]
        
        # Interpolate abundances
        for species_i, interp_func_i in self.interp_tables.items():

            # Interpolate the equilibrium abundances
            arr_i = interp_func_i(xi=(P, T, CO, FeH))

            if species_i != 'MMW':
                self.VMRs[species_i] = 10**arr_i # log10(VMR)
            else:
                self.MMW = arr_i.copy() # Mean-molecular weight

class pRTChemistryTable(EquilibriumChemistry):

    def __init__(self, line_species, pressure, CustomOpacity=None, **kwargs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure, CustomOpacity)

        # Load the interpolation tables
        import petitRADTRANS.poor_mans_nonequ_chem as pm
        self.interp_tables = pm.interpol_abundances

        # Species to quench per system
        self.quench_settings = {
            'CO_CH4': [('12CO','CH4','H2O'), None], 
            'N2_NH3': [('N2','NH3'), None], 
        }

    def get_VMRs(self, params, **kwargs):

        # Update the parameters
        self.CO  = params.get('C/O')
        self.FeH = params.get('Fe/H')

        # Retrieve the mass fractions from the pRT eq-chem table
        pm_mass_fractions = self.interp_tables(
            self.CO*np.ones(self.n_atm_layers), 
            self.FeH*np.ones(self.n_atm_layers), 
            self.temperature, 
            self.pressure
            )
    
        # Fill in the VMR dictionary
        self.MMW = pm_mass_fractions['MMW']
        for species_i, hill_i in zip(self.species, self.hill):
            
            if not isinstance(hill_i, str):
                # Minor isotope
                continue

            # Get a free-chemistry VMR
            param_VMR_i = params.get(species_i)
            if param_VMR_i is not None:
                self.VMRs[species_i] = param_VMR_i * np.ones(self.n_atm_layers)
                continue

            # Search a different key for 12CO
            key_i = species_i
            if species_i == '12CO':
                key_i = 'CO'

            # Convert mass fraction to a VMR
            mass_i = self.read_species_info(species_i, 'mass')
            self.VMRs[species_i] = pm_mass_fractions[key_i] * self.MMW/mass_i

            if (self.VMRs[species_i]==0.).any():
                self.VMRs = -np.inf
                return
            

class FastChemistry(EquilibriumChemistry):

    def __init__(self, line_species, pressure, CustomOpacity=None, **kwargs):
        
        # Give arguments to the parent class
        super().__init__(line_species, pressure, CustomOpacity)

        # Create the FastChem object
        import pyfastchem as pyfc
        self.fastchem = pyfc.FastChem(
            kwargs.get('abundance_file'),
            kwargs.get('gas_data_file'), 
            kwargs.get('cond_data_file', 'none'), 
            1
            )
        self.gas_species_tot = self.fastchem.getGasSpeciesNumber()

        # Create in/out-put structures for FastChem
        self.input  = pyfc.FastChemInput()
        self.output = pyfc.FastChemOutput()

        # Use equilibrium (+rainout) condensation
        self.input.equilibrium_condensation = kwargs.get('use_eq_cond', True)
        self.input.rainout_condensation     = kwargs.get('use_rainout_cond', False)

        # Configure FastChem's internal parameters
        self.fastchem.setParameter('accuracyChem', 1e-4)

        # Get element and solar properties
        self._get_element_indices()
        self._get_solar()

    def _get_element_indices(self):

        # Get the indices of relevant species
        self.idx = {
            el: int(self.fastchem.getElementIndex(el)) \
            for el in ['H','He','C','N','O',]
        }

        # All but the H, He indices
        self.metal_idx = np.arange(self.fastchem.getElementNumber())
        self.metal_idx = np.delete(
            self.metal_idx, [self.idx['H'],self.idx['He']]
            )
        
    def _get_solar(self):
        
        # Make a copy of the solar abundances
        self.solar_abund = np.array(self.fastchem.getElementAbundances())
        
        # Solar abundance ratios
        self.solar_CO = self.solar_abund[self.idx['C']] / self.solar_abund[self.idx['O']]
        self.solar_NO = self.solar_abund[self.idx['N']] / self.solar_abund[self.idx['O']]
        
        self.solar_FeH = 0.0

    def _set_metallicity(self):
        self.el_abund[self.metal_idx] *= 10**self.FeH

    def _set_CO(self):
        # C = C/O * O_sol
        self.el_abund[self.idx['C']] = self.CO * self.el_abund[self.idx['O']]

        # Correct for the summed abundance of C+O
        tot_abund_ratio = (1+self.solar_CO) / (1+self.CO)
        self.el_abund[self.idx['C']] *= tot_abund_ratio
        self.el_abund[self.idx['O']] *= tot_abund_ratio

    def _set_NO(self):
        # N = N/O * O_sol
        self.el_abund[self.idx['N']] = self.NO * self.el_abund[self.idx['O']]

        # Correct for the summed abundance of N+O
        tot_abund_ratio = (1+self.solar_NO) / (1+self.NO)
        self.el_abund[self.idx['N']] *= tot_abund_ratio
        self.el_abund[self.idx['O']] *= tot_abund_ratio

    def get_VMRs(self, params, **kwargs):

        # Flip to order by increasing altitude
        self.input.pressure = self.pressure[::-1]

        # Fastchem doesn't converge for low temperatures
        temperature = self.temperature[::-1].copy()
        temperature[temperature<500.] = 500.
        self.input.temperature = temperature

        # Update the parameters
        self.CO  = params.get('C/O')
        self.NO  = params.get('N/O', self.solar_NO)
        self.FeH = params.get('Fe/H')

        # Modify the elemental abundances, initially solar
        self.el_abund = self.solar_abund.copy()
        self._set_CO()
        self._set_NO()
        self._set_metallicity()

        # Update the element abundances
        self.fastchem.setElementAbundances(self.el_abund)

        # Compute the number densities
        fastchem_flag = self.fastchem.calcDensities(self.input, self.output)

        if (fastchem_flag != 0) or (np.amin(self.output.element_conserved) != 1):
            # FastChem failed to converge or conserve elements
            self.VMRs = -np.inf
            return

        # Species-specific and total number densities
        n = np.array(self.output.number_densities)
        n_tot = n.sum(axis=1)

        # Fill in the VMRs dictionary
        self.MMW = np.array(self.output.mean_molecular_weight)[::-1] # Flip back
        for species_i, hill_i in zip(self.species, self.hill):
            
            if not isinstance(hill_i, str):
                # Minor isotope
                continue

            # Species' index in gas-density array
            idx = self.fastchem.getGasSpeciesIndex(hill_i)

            if idx >= self.gas_species_tot:
                # Species not in fastchem
                continue

            # Volume-mixing ratio, flip back to decreasing altitude
            self.VMRs[species_i] = n[:,idx][::-1] / n_tot

        print(self.VMRs)