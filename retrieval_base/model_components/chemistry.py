import numpy as np
import pandas as pd

import re
import pathlib
directory_path = pathlib.Path(__file__).parent.resolve()

from ..utils import sc

def get_class(pressure, line_species, chem_mode='free', **kwargs):
    """
    Factory function to get the appropriate Chemistry class based on the chem_mode.

    Args:
        pressure (np.ndarray): Pressure levels.
        line_species (list): List of line species.
        chem_mode (str, optional): Mode of the chemistry. Defaults to 'free'.
        **kwargs: Additional keyword arguments.

    Returns:
        Chemistry: An instance of a Chemistry subclass.
    """
    if chem_mode == 'free':
        return FreeChemistry(pressure, line_species, **kwargs)
    elif chem_mode == 'fastchem':
        return FastChemistry(pressure, line_species, **kwargs)
    elif chem_mode == 'pRT_table':
        return pRTChemistryTable(pressure, line_species, **kwargs)
    elif chem_mode == 'fastchem_table':
        return FastChemistryTable(pressure, line_species, **kwargs)
    elif chem_mode == 'fastchem_table_enhancement':
        return FastChemistryTableEnhancement(pressure, line_species, **kwargs)
    else:
        raise ValueError(f'Chemistry mode {chem_mode} not recognized.')

class Chemistry:
    """
    Base class for handling chemical species and their properties.
    """

    species_info = pd.read_csv(directory_path/'chemistry_info.csv', index_col=0)
    neglect_species = {key_i: False for key_i in species_info.index}

    def __init__(self, pressure, line_species, LineOpacity=None):
        """
        Initialize the Chemistry class.

        Args:
            pressure (np.ndarray): Pressure levels.
            line_species (list): List of line species.
            LineOpacity (list, optional): Custom opacity objects. Defaults to None.
        """
        self.line_species = [*line_species, 'H2', 'He']

        # Custom line-opacities
        if LineOpacity is not None:
            self.line_species += [LineOpacity_i.line_species for LineOpacity_i in LineOpacity]
        
        # Store the regular name and hill-notations too
        self.species, self.hill = [], []
        for species_i in self.species_info.index:
            
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            hill_i         = self.read_species_info(species_i, 'pyfc_name')

            if line_species_i not in self.line_species:
                continue

            if species_i == 'H2_lines':
                # Add H2 as a line_species separately
                self.add_H2_line_species = line_species_i
                self.line_species.remove(self.add_H2_line_species)
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
        """
        Remove the contribution of the specified species by setting their mass fractions to zero.
        """
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
        """
        Read species information from the species_info DataFrame.

        Args:
            species (str): Species name.
            info_key (str): Key to retrieve specific information.

        Returns:
            Various: Information based on the info_key.
        """
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

        if info_key in ['c','color']:
            return cls.species_info.loc[species,'color']
        if info_key == 'label':
            return cls.species_info.loc[species,'mathtext_name']
        
    def get_VMRs(self, *args):
        """
        Placeholder method to get volume mixing ratios (VMRs).
        Should be implemented by child classes.
        """
        raise NotImplementedError("Subclasses should implement this method")
    
    def get_isotope_VMRs(self, ParamTable):
        """
        Calculate the volume mixing ratios (VMRs) for isotopologues.

        Args:
            ParamTable (dict): Parameters including isotope ratios.
        """
        # If true, eq-chem abundances should be split into isotopologues
        conserve_tot_VMR = isinstance(self, EquilibriumChemistry)

        # Get all isotope ratios per species (possibly shared between molecules)
        all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios = \
            self._share_isotope_ratios(ParamTable)

        VMRs_copy = self.VMRs.copy()
        for species_i in self.species:

            VMR_i = VMRs_copy.get(species_i, np.zeros_like(self.pressure))
            if (VMR_i != 0.).any():
                # Already set
                continue

            if species_i not in [*all_CO_ratios, *all_H2O_ratios, *all_CH4_ratios, *all_NH3_ratios, *all_CO2_ratios]:
                # Not a CO, H2O, CH4, NH3, or CO2 isotopologue
                continue
            
            iterables = zip(
                [all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios], 
                ['12CO', 'H2O', 'CH4', 'NH3', 'CO2']
            )
            for all_ratios, main_iso_i in iterables:

                # Minor-to-main ratio
                minor_main_ratio_i = all_ratios.get(species_i)

                # Read the VMR of the main isotopologue
                main_iso_VMR_i = VMRs_copy.get(main_iso_i)

                sum_of_ratios = 1.
                if conserve_tot_VMR:
                    # To conserve the total abundance
                    sum_of_ratios = sum(all_ratios.values())

                if minor_main_ratio_i is not None:
                    # Matching isotope ratio found
                    break
            
            if main_iso_VMR_i is None:
                # Main isotopologue not set
                continue

            # e.g. 13CO = CO_all * 13/12C / (12/12C+13/12C+18/16O+17/16O)
            self.VMRs[species_i] = main_iso_VMR_i * minor_main_ratio_i/sum_of_ratios
    
    def convert_to_MFs(self):
        """
        Convert volume mixing ratios (VMRs) to mass fractions (MFs).
        """
        # Convert to mass-fractions using mass-ratio
        self.mass_fractions = {'MMW': self.MMW}
        for species_i, VMR_i in self.VMRs.items():
            
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')

            self.mass_fractions[line_species_i] = VMR_i * mass_i/self.MMW
    
    def get_diagnostics(self):
        """
        Calculate diagnostics such as C/O and Fe/H ratios.
        """
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
        
    def __call__(self, ParamTable, temperature):
        """
        Evaluate the chemistry model with given parameters and temperature.

        Args:
            ParamTable (dict): Parameters for the model.
            temperature (np.ndarray): Temperature profile.

        Returns:
            dict: Mass fractions.
        """
        self.temperature = temperature
        
        self.mass_fractions = {}
        self.VMRs = {}
        self.MMW  = 0.
        
        self.CO  = None
        self.FeH = None

        # Get volume-mixing ratios
        self.get_VMRs(ParamTable)
        if self.VMRs == -np.inf:
            # Some issue was raised
            self.mass_fractions = -np.inf
            return -np.inf

        if hasattr(self, 'quench_VMRs'):
            # Quench eq-chem abundances
            self.quench_VMRs(ParamTable)

        # Get isotope abundances
        self.get_isotope_VMRs(ParamTable)

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

        if hasattr(self, 'add_H2_line_species'):
            # Add H2 line opacity
            self.VMRs['H2_lines'] = self.VMRs['H2'].copy()
            self.mass_fractions[self.add_H2_line_species] = self.mass_fractions['H2'].copy()
        
        return self.mass_fractions

    def _share_isotope_ratios(self, ParamTable):
        """
        Share isotope ratios between molecules.

        Args:
            ParamTable (dict): Parameters for the model.
        """

        # First looks for isotopologue ratio, then for isotope ratio, otherwise sets abundance to 0
        all_CO_ratios = {
            '12CO': 1., 
            '13CO': 1./ParamTable.get('13CO_ratio', ParamTable.get('12/13C_ratio', np.inf)), 
            'C18O': 1./ParamTable.get('C18O_ratio', ParamTable.get('16/18O_ratio', np.inf)), 
            'C17O': 1./ParamTable.get('C17O_ratio', ParamTable.get('16/17O_ratio', np.inf)), 
        }
        all_H2O_ratios = {
            'H2O':     1., 
            'H2(18)O': 1./ParamTable.get('H2(18)O_ratio', ParamTable.get('16/18O_ratio', np.inf)), 
            'H2(17)O': 1./ParamTable.get('H2(17)O_ratio', ParamTable.get('16/17O_ratio', np.inf)), 
            'HDO':     1./ParamTable.get('HDO_ratio', ParamTable.get('H/D_ratio', np.inf)), 
        }
        all_CH4_ratios = {
            'CH4':   1., 
            '13CH4': 1./ParamTable.get('13CH4_ratio', ParamTable.get('12/13C_ratio', np.inf)), 
        }
        all_NH3_ratios = {
            'NH3':   1., 
            '15NH3': 1./ParamTable.get('15NH3_ratio', ParamTable.get('14/15N_ratio', np.inf)), 
        }
        all_CO2_ratios = {
            'CO2':     1., 
            '13CO2':   1./ParamTable.get('13CO2_ratio', ParamTable.get('12/13C_ratio', np.inf)), 
            'CO(18)O': 1./ParamTable.get('CO(18)O_ratio', ParamTable.get('16/18O_ratio', np.inf)), 
            'CO(17)O': 1./ParamTable.get('CO(17)O_ratio', ParamTable.get('16/17O_ratio', np.inf)), 
        }
        return all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios

    def _power_law_drop_off(self, VMR, P0, alpha):
        """
        Apply a power-law drop-off to the volume mixing ratio (VMR).

        Args:
            VMR (np.ndarray): Volume mixing ratio.
            P0 (float): Reference pressure.
            alpha (float): Power-law exponent.

        Returns:
            np.ndarray: Modified VMR.
        """
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
    
class FreeChemistry(Chemistry):
    """
    Class for handling free chemistry models.
    """

    def __init__(self, pressure, line_species, LineOpacity=None, **kwargs):
        """
        Initialize the FreeChemistry class.

        Args:
            pressure (np.ndarray): Pressure levels.
            line_species (list): List of line species.
            LineOpacity (list, optional): Custom opacity objects. Defaults to None.
            **kwargs: Additional arguments.
        """
        # Give arguments to the parent class
        super().__init__(pressure, line_species, LineOpacity)

    def get_VMRs(self, ParamTable):
        """
        Get volume mixing ratios (VMRs) for the free chemistry model.

        Args:
            ParamTable (dict): Parameters for the model.
        """
        # Constant He abundance
        self.VMRs = {'He':0.15*np.ones(self.n_atm_layers)}

        for species_i in self.species:

            if species_i in ['H2', 'He']:
                continue # Set by other VMRs

            # Read the fitted VMR (or set to 0)
            param_VMR_i = ParamTable.get(species_i, 0.)
            # Expand to all layers
            param_VMR_i *= np.ones(self.n_atm_layers)

            # Parameterise a power-law drop-off
            self.VMRs[species_i] = self._power_law_drop_off(
                param_VMR_i, P0=ParamTable.get(f'{species_i}_P'), 
                alpha=ParamTable.get(f'{species_i}_alpha'), 
                )

        # Overwrite C/O ratio and Fe/H with constants (if given)
        self.CO  = ParamTable.get('C/O', None)
        self.FeH = ParamTable.get('Fe/H', None)
        
    def get_H2(self):
        """
        Calculate the H2 abundance as the remainder of the total VMR.
        """
        # H2 abundance is the remainder
        VMR_wo_H2 = np.sum([VMR_i for VMR_i in self.VMRs.values()], axis=0)
        self.VMRs['H2'] = 1 - VMR_wo_H2

        if (self.VMRs['H2'] < 0).any():
            # Other species are too abundant
            self.VMRs = -np.inf

    def get_MMW(self):
        """
        Calculate the mean molecular weight (MMW) from the VMRs.
        """
        # Get mean-molecular weight from free-chem VMRs
        MMW = 0.
        for species_i, VMR_i in self.VMRs.items():
            mass_i = self.read_species_info(species_i, 'mass')
            MMW += mass_i * VMR_i
        
    def convert_to_MFs(self):
        """
        Convert volume mixing ratios (VMRs) to mass fractions (MFs).
        """
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
    """
    Class for handling equilibrium chemistry models.
    """
    
    def __init__(self, pressure, line_species, LineOpacity=None):
        """
        Initialize the EquilibriumChemistry class.

        Args:
            pressure (np.ndarray): Pressure levels.
            line_species (list): List of line species.
            LineOpacity (list, optional): Custom opacity objects. Defaults to None.
        """
        # Give arguments to the parent class
        super().__init__(pressure, line_species, LineOpacity)

        # Species to quench per system
        self.quench_settings = {
            'CO_CH4': [['12CO','CH4','H2O'], None], 
            'N2_NH3': [['N2','NH3'], None], 
            'HCN': [['HCN'], None],
            'CO2': [['CO2'], None],
        }

    def get_P_quench_from_Kzz(self, ParamTable, alpha=1):
        """
        Calculate the quench pressure from the eddy diffusion coefficient (Kzz).

        Args:
            ParamTable (dict): Parameters including Kzz.
            alpha (float, optional): Mixing length factor. Defaults to 1.
        """

        def interpolate_for_P_quench(log_t_mix, log_t_chem):
            """Interpolate the quench pressure based on mixing and chemical timescales."""
            idx_well_mixed = np.argwhere(log_t_mix < log_t_chem).flatten()
            if len(idx_well_mixed) == 0:
                # All layers are in chemical equilibrium
                return self.pressure.min()
            
            # Lowest layer that is well-mixed
            idx_lowest_mixed_layer = idx_well_mixed[-1]
            idx_highest_equilibrium_layer = idx_lowest_mixed_layer + 1

            if idx_highest_equilibrium_layer >= len(self.pressure):
                # All layers are well-mixed
                return self.pressure.max()
            
            # Quenching happens between these two layers
            indices = [idx_lowest_mixed_layer, idx_highest_equilibrium_layer]
            return 10**np.interp(
                0, xp=log_t_mix[indices]-log_t_chem[indices], fp=np.log10(self.pressure[indices])
                )

        # Metallicity
        m = 10**self.FeH

        # Scale height at each layer
        kB  = sc.k * 1e7
        amu = sc.amu * 1e3  # Convert to cgs (g)
        H = kB*self.temperature / (self.MMW*amu * ParamTable.get('g'))

        # Mixing length/time-scales
        L = alpha * H
        t_mix = L**2 / ParamTable.get('Kzz_chem')
        log_t_mix = np.log10(t_mix)

        # Ignore overflow warnings
        with np.errstate(over='ignore'):
            # Chemical timescales from Zahnle & Marley (2014)

            # CO-CH4
            inv_t_q1 = 1.0e6/1.5 * self.pressure * m**0.7 * np.exp(-42000/self.temperature)
            inv_t_q2 = 1/40 * self.pressure**2 * np.exp(-25000/self.temperature)
            log_t_CO_CH4 = -1 * np.log10(inv_t_q1 + inv_t_q2)
            self.quench_settings['CO_CH4'][-1] = interpolate_for_P_quench(log_t_mix, log_t_CO_CH4)
            
            # N2-NH3
            # t_NH3 = 1.0e-7 * self.pressure**(-1) * np.exp(52000/self.temperature)
            log_t_NH3 = (
                -7.0 - np.log10(self.pressure) + 52000/self.temperature*np.log10(np.e)
            )
            self.quench_settings['N2_NH3'][-1] = interpolate_for_P_quench(log_t_mix, log_t_NH3)

            # HCN-NH3-N2
            # t_HCN = 1.5e-4 * self.pressure**(-1) * m**(-0.7) * np.exp(36000/self.temperature)
            log_t_HCN = (
                np.log10(1.5e-4) - np.log10(self.pressure*m**0.7) + 36000/self.temperature*np.log10(np.e)
            )
            self.quench_settings['HCN'][-1] = interpolate_for_P_quench(log_t_mix, log_t_HCN)

            # CO2
            # t_CO2 = 1.0e-10 * self.pressure**(-0.5) * np.exp(38000/self.temperature)
            # log_t_CO2 = (
            #     -10.0 - 0.5*np.log10(self.pressure) + 38000/self.temperature*np.log10(np.e)
            # )
            # Approximate CO2 quenching as the same level as CO-CH4, since CO2 remains in equilibrium 
            # with the quenched (i.e. not decreasing) CO abundance
            log_t_CO2 = log_t_CO_CH4
            self.quench_settings['CO2'][-1] = interpolate_for_P_quench(log_t_mix, log_t_CO2)
            
    def quench_VMRs(self, ParamTable):
        """
        Quench the volume mixing ratios (VMRs) based on the quench settings.

        Args:
            ParamTable (dict): Parameters for the model.
        """
        # Update the parameters
        for key_q in list(self.quench_settings):
            # Update the quench pressures from the free parameters
            self.quench_settings[key_q][-1] = ParamTable.get(f'P_quench_{key_q}')

        if ParamTable.get('Kzz_chem') is not None:
            # Update the quench pressures from diffusivity
            self.get_P_quench_from_Kzz(ParamTable)

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
                
                # Quench the VMRs of specific species
                VMR_i = np.clip(self.VMRs[species_i], 1e-100, None)
                log_VMR_i = np.log10(VMR_i)
                
                log_VMR_i[mask_TOA] = np.interp(
                    np.log10(P_q), xp=log_P, fp=log_VMR_i
                    )
                self.VMRs[species_i] = 10**log_VMR_i

class FastChemistryTable(EquilibriumChemistry):
    """
    Class for handling fast chemistry models using interpolation tables.
    """

    def __init__(self, pressure, line_species, LineOpacity=None, path_fastchem_tables='./fastchem_tables', grid_ranges={}, **kwargs):
        """
        Initialize the FastChemistryTable class.

        Args:
            pressure (np.ndarray): Pressure levels.
            line_species (list): List of line species.
            LineOpacity (list, optional): Custom opacity objects. Defaults to None.
            path_fastchem_tables (str, optional): Path to the fast chemistry tables. Defaults to './fastchem_tables'.
            grid_ranges (dict, optional): Custom ranges for the interpolation grid. Defaults to {}.
            **kwargs: Additional arguments.
        """
        # Give arguments to the parent class
        super().__init__(pressure, line_species, LineOpacity)

        # Load the interpolation tables
        self._load_interp_tables(pathlib.Path(path_fastchem_tables), grid_ranges)

    def _load_interp_tables(self, path_tables, grid_ranges):
        """
        Load the interpolation tables for fast chemistry.

        Args:
            path_tables (Path): Path to the tables.
            grid_ranges (dict): Custom ranges for the interpolation grid.
        """
        
        import h5py
        def load_hdf5(file, key):
            with h5py.File(f'{path_tables}/{file}', 'r') as f:
                return f[key][...]
            
        def load_grid(key):
            grid = load_hdf5('MMW.hdf5', key)
            mask = (grid >= default_ranges[key][0]) & (grid <= default_ranges[key][1])
            return grid[mask], mask

        # Load only a certain range of the grid
        default_ranges = {
            'P_grid': [1e-6,1e3], 'T_grid': [150,6000], 'CO_grid': [0.1,1.6], 
            'NO_grid': [0.05,0.5], 'FeH_grid': [-1.0,1.0],
            }
        default_ranges.update(grid_ranges)

        # Load the interpolation grid
        self.P_grid, mask_P     = load_grid('P_grid')
        self.T_grid, mask_T     = load_grid('T_grid')
        self.CO_grid, mask_CO   = load_grid('CO_grid')
        self.NO_grid, mask_NO   = load_grid('NO_grid')
        self.FeH_grid, mask_FeH = load_grid('FeH_grid')

        points = (self.P_grid, self.T_grid, self.CO_grid, self.NO_grid, self.FeH_grid)
        
        from scipy.interpolate import RegularGridInterpolator
        self.interp_tables = {}
        for species_i, hill_i in zip([*self.species, 'MMW'], [*self.hill, 'MMW']):
            key = 'MMW' if species_i=='MMW' else 'log_VMR'

            if not isinstance(hill_i, str):
                # Minor isotope
                continue

            # Load the eq-chem abundance tables
            arr = load_hdf5(f'{hill_i}.hdf5', key=key)
            arr = arr[mask_P][:,mask_T][:,:,mask_CO][:,:,:,mask_NO][:,:,:,:,mask_FeH]

            # Generate interpolation functions
            self.interp_tables[species_i] = RegularGridInterpolator(
                values=arr, points=points, method='linear'
                )
            
    def get_VMRs(self, ParamTable):
        """
        Get volume mixing ratios (VMRs) using interpolation tables.

        Args:
            ParamTable (dict): Parameters for the model.
        """

        # Update the parameters
        self.CO  = ParamTable.get('C/O')
        self.NO  = ParamTable.get('N/O')
        self.FeH = ParamTable.get('Fe/H')

        # Apply the bounds of the grid
        P   = np.clip(self.pressure, self.P_grid.min(), self.P_grid.max())
        T   = np.clip(self.temperature, self.T_grid.min(), self.T_grid.max())
        CO  = np.clip(np.array([self.CO]), self.CO_grid.min(), self.CO_grid.max())[0]
        NO  = np.clip(np.array([self.NO]), self.NO_grid.min(), self.NO_grid.max())[0]
        FeH = np.clip(np.array([self.FeH]), self.FeH_grid.min(), self.FeH_grid.max())[0]
        
        # Interpolate abundances
        for species_i, interp_func_i in self.interp_tables.items():

            # Interpolate the equilibrium abundances
            arr_i = interp_func_i(xi=(P, T, CO, NO, FeH))

            if species_i != 'MMW':
                self.VMRs[species_i] = 10**arr_i # log10(VMR)
            else:
                self.MMW = arr_i.copy() # Mean-molecular weight

class FastChemistryTableEnhancement(FastChemistryTable):

    def __init__(self, pressure, line_species, **kwargs):
        """
        Initialize the FastChemistryTableEnhancement class. 
        (This class assumes that condensation is not 
        affected by individual elemental abundances.)

        Args:
            pressure (np.ndarray): Pressure levels.
            line_species (list): List of line species.
            **kwargs: Additional arguments.
        """
        # Give arguments to the parent class
        super().__init__(pressure, line_species, **kwargs)

    def get_VMRs(self, ParamTable):
        """
        Get volume mixing ratios (VMRs) using interpolation tables 
        and enhance the elemental abundances.

        Args:
            ParamTable (dict): Parameters for the model.
        """
        # Read VMRs in parent class
        super().get_VMRs(ParamTable)

        # Enhance the abundances of all elements
        for species_i, hill_i in zip(self.species, self.hill):
            
            if not isinstance(hill_i, str):
                # Minor isotope
                continue
            
            # Split the molecule into its elements
            elements_i = re.findall(r'[A-Z][a-z]?\d*', hill_i)
            for element_j in elements_i:

                # Number of this element in molecule (or 1 if atom)
                N_j = 1 if not element_j[-1].isdigit() else int(element_j[-1])
                element_j = element_j.rstrip('0123456789')

                # Read the alpha-enhancement factor
                alpha_j = ParamTable.get(f'alpha_{element_j}', None)
                if alpha_j is None:
                    continue
                
                # Enhance the abundance of this element
                self.VMRs[species_i] *= N_j * 10**alpha_j

class pRTChemistryTable(EquilibriumChemistry):
    """
    Class for handling pRT chemistry models using interpolation tables.
    """

    def __init__(self, pressure, line_species, LineOpacity=None, **kwargs):
        """
        Initialize the pRTChemistryTable class.

        Args:
            pressure (np.ndarray): Pressure levels.
            line_species (list): List of line species.
            LineOpacity (list, optional): Custom opacity objects. Defaults to None.
            **kwargs: Additional arguments.
        """
        # Give arguments to the parent class
        super().__init__(pressure, line_species, LineOpacity)

        # Load the interpolation tables
        import petitRADTRANS.poor_mans_nonequ_chem as pm
        self.interp_tables = pm.interpol_abundances

    def get_VMRs(self, ParamTable):
        """
        Get volume mixing ratios (VMRs) using pRT interpolation tables.

        Args:
            ParamTable (dict): Parameters for the model.
        """
        # Update the parameters
        self.CO  = ParamTable.get('C/O')
        self.FeH = ParamTable.get('Fe/H')

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
            param_VMR_i = ParamTable.get(species_i)
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
    """
    Class for handling fast chemistry models using the FastChem library.
    """

    def __init__(self, pressure, line_species, LineOpacity=None, **kwargs):
        """
        Initialize the FastChemistry class.

        Args:
            pressure (np.ndarray): Pressure levels.
            line_species (list): List of line species.
            LineOpacity (list, optional): Custom opacity objects. Defaults to None.
            **kwargs: Additional arguments.
        """
        # Give arguments to the parent class
        super().__init__(pressure, line_species, LineOpacity)

        # Complexity of reaction network
        self.abundance_file = kwargs.get('abundance_file')
        self.gas_data_file  = kwargs.get('gas_data_file')
        self.cond_data_file = kwargs.get('cond_data_file', 'none')

        # Use equilibrium (+rainout) condensation
        self.use_eq_cond      = kwargs.get('use_eq_cond', True)
        self.use_rainout_cond = kwargs.get('use_rainout_cond', False)

        # Create the FastChem object
        self._get_fastchem_object()
        self.gas_species_tot = self.fastchem.getGasSpeciesNumber()

        self.min_temperature = kwargs.get('min_temperature', 500.)

        # Get element and solar properties
        self._get_element_indices()
        self._get_solar()

    def _get_fastchem_object(self):
        """
        Get the FastChem object.
        """
        if hasattr(self, 'fastchem'):
            return
        
        verbose = 1
        if hasattr(self, 'solar_abund'):
            verbose = 0 # Suppress initialisation message during sampling

        import pyfastchem as pyfc
        self.fastchem = pyfc.FastChem(
            self.abundance_file, self.gas_data_file, self.cond_data_file, verbose
            )
        
        # Configure FastChem's internal parameters
        self.fastchem.setParameter('accuracyChem', 1e-4)
        self.fastchem.setVerboseLevel(1)

        # Create in/out-put structures for FastChem
        self.input  = pyfc.FastChemInput()
        self.output = pyfc.FastChemOutput()

        self.input.equilibrium_condensation = self.use_eq_cond
        self.input.rainout_condensation     = self.use_rainout_cond

    def _get_element_indices(self):
        """
        Get the indices of relevant elements in the FastChem library.
        """
        # Get the indices of relevant species
        self.idx = {
            self.fastchem.getElementSymbol(i): \
            i for i in range(self.fastchem.getElementNumber())
        }

        # All but the H, He indices
        self.metal_idx = np.arange(self.fastchem.getElementNumber())
        self.metal_idx = np.delete(
            self.metal_idx, [self.idx['H'],self.idx['He']]
            )
        
    def _get_solar(self):
        """
        Get the solar abundances and ratios.
        """
        # Make a copy of the solar abundances
        self.solar_abund = np.array(self.fastchem.getElementAbundances())
        
        # Solar abundance ratios
        self.solar_CO = self.solar_abund[self.idx['C']] / self.solar_abund[self.idx['O']]
        self.solar_NO = self.solar_abund[self.idx['N']] / self.solar_abund[self.idx['O']]
        
        self.solar_FeH = 0.0

    def _set_metallicity(self):
        """
        Set the metallicity for the element abundances.
        """
        self.el_abund[self.metal_idx] *= 10**self.FeH

    def _set_CO(self):
        """
        Set the C/O ratio for the element abundances.
        """
        # C = C/O * O_sol
        self.el_abund[self.idx['C']] = self.CO * self.el_abund[self.idx['O']]

        # Correct for the summed abundance of C+O
        tot_abund_ratio = (1+self.solar_CO) / (1+self.CO)
        self.el_abund[self.idx['C']] *= tot_abund_ratio
        self.el_abund[self.idx['O']] *= tot_abund_ratio

    def _set_NO(self):
        """
        Set the N/O ratio for the element abundances.
        """
        # N = N/O * O_sol
        self.el_abund[self.idx['N']] = self.NO * self.el_abund[self.idx['O']]

        # Correct for the summed abundance of N+O
        tot_abund_ratio = (1+self.solar_NO) / (1+self.NO)
        self.el_abund[self.idx['N']] *= tot_abund_ratio
        self.el_abund[self.idx['O']] *= tot_abund_ratio

    def _set_elemental_abundances(self, ParamTable):
        """
        Set the abundances of each element separately.
        """
        for el, i in self.idx.items():
            # Enhance the elemental abundance
            alpha_i = ParamTable.get(f'alpha_{el}', ParamTable.get(f'[M/H]', None))
            if alpha_i is None:
                continue

            if el in ['e-', 'H', 'He']:
                continue

            self.el_abund[i] = 10**alpha_i * self.el_abund[i]

    def get_VMRs(self, ParamTable):
        """
        Get volume mixing ratios (VMRs) using the FastChem library.

        Args:
            ParamTable (dict): Parameters for the model.
        """
        # Reinitialise the fastchem object if it doesn't exist
        self._get_fastchem_object()

        # Flip to order by increasing altitude
        self.input.pressure = self.pressure[::-1]

        # Fastchem doesn't converge for low temperatures
        temperature = self.temperature[::-1].copy()
        temperature[temperature<self.min_temperature] = self.min_temperature
        self.input.temperature = temperature

        # Update the parameters
        self.CO  = ParamTable.get('C/O', self.solar_CO)
        self.NO  = ParamTable.get('N/O', self.solar_NO)
        self.FeH = ParamTable.get('Fe/H', self.solar_FeH)

        # Modify the elemental abundances, initially solar
        self.el_abund = self.solar_abund.copy()
        self._set_CO()
        self._set_NO()
        self._set_metallicity()
        self._set_elemental_abundances(ParamTable)

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

            # Read retrieved VMR (if available)
            param_VMR_i = ParamTable.get(species_i)
            if param_VMR_i is not None:
                # Expand to all layers
                param_VMR_i *= np.ones(self.n_atm_layers)

                # Parameterise a power-law drop-off
                self.VMRs[species_i] = self._power_law_drop_off(
                    param_VMR_i, P0=ParamTable.get(f'{species_i}_P'), 
                    alpha=ParamTable.get(f'{species_i}_alpha'), 
                    )
                continue # Ignore FastChem abundance

            # Species' index in gas-density array
            idx = self.fastchem.getGasSpeciesIndex(hill_i)

            if idx >= self.gas_species_tot:
                # Species not in fastchem
                continue

            # Volume-mixing ratio, flip back to decreasing altitude
            self.VMRs[species_i] = (n[:,idx] / n_tot)[::-1]

        for species_i, VMR_i in self.VMRs.items():
            if (VMR_i==0.).any():
                print(f'FastChem failed for species {species_i}.')
                self.VMRs = -np.inf
                return