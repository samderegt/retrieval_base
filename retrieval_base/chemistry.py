import numpy as np
import pandas as pd

import pathlib
directory_path = pathlib.Path(__file__).parent.resolve()

from scipy.interpolate import make_interp_spline
import petitRADTRANS.nat_cst as nc

def get_Chemistry_class(line_species, pressure, mode, **kwargs):

    if mode == 'free':
        return FreeChemistry(line_species, pressure, **kwargs)
    if mode == 'eqchem':
        return EqChemistry(line_species, pressure, **kwargs)
    if mode == 'fastchem':
        return FastChemistry(line_species, pressure, **kwargs)
    if mode == 'SONORAchem':
        return SONORAChemistry(line_species, pressure, **kwargs)

class Chemistry:

    species_info = pd.read_csv(directory_path/'species_info.csv', index_col=0)
    neglect_species = {key_i: False for key_i in species_info.index}

    def __init__(self, line_species, pressure):

        self.line_species = line_species

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

class FreeChemistry(Chemistry):

    def __init__(self, line_species, pressure, spline_order=0, **kwargs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure)

        self.spline_order = spline_order

    def __call__(self, VMRs, params):

        self.VMRs = VMRs

        # Total VMR without H2, starting with He
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He

        # Create a dictionary for all used species
        self.mass_fractions = {}

        C, O, H = 0, 0, 0

        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue

            if line_species_i in self.line_species:
                
                if self.VMRs.get(species_i) is not None:
                    # Single value given: constant, vertical profile
                    VMR_i = self.VMRs[species_i] * np.ones(self.n_atm_layers)

                if self.VMRs.get(f'{species_i}_0') is not None:
                    # Multiple values given, use spline interpolation

                    # Define the spline knots in pressure-space
                    if params.get(f'log_P_{species_i}') is not None:
                        log_P_knots = np.array([
                            np.log10(self.pressure).min(), params[f'log_P_{species_i}'], 
                            np.log10(self.pressure).max()
                            ])
                    else:
                        log_P_knots = np.linspace(
                            np.log10(self.pressure).min(), 
                            np.log10(self.pressure).max(), num=3
                            )
                    
                    # Define the abundances at the knots
                    VMR_knots = np.array([self.VMRs[f'{species_i}_{j}'] for j in range(3)])[::-1]
                    
                    # Use a k-th order spline to vary the abundance profile
                    spl = make_interp_spline(log_P_knots, np.log10(VMR_knots), k=self.spline_order)
                    VMR_i = 10**spl(np.log10(self.pressure))

                self.VMRs[species_i] = VMR_i

                # Convert VMR to mass fraction using molecular mass number
                self.mass_fractions[line_species_i] = mass_i * VMR_i
                VMR_wo_H2 += VMR_i

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        self.mass_fractions['He'] = self.read_species_info('He', 'mass') * VMR_He
        self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * (1 - VMR_wo_H2)

        # Add to the H-bearing species
        H += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)

        if VMR_wo_H2.any() > 1:
            # Other species are too abundant
            self.mass_fractions = -np.inf
            return self.mass_fractions

        # Compute the mean molecular weight from all species
        MMW = 0
        for mass_i in self.mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)

        # Turn the molecular masses into mass fractions
        for line_species_i in self.mass_fractions.keys():
            self.mass_fractions[line_species_i] /= MMW

        # pRT requires MMW in mass fractions dictionary
        self.mass_fractions['MMW'] = MMW

        # Compute the C/O ratio and metallicity
        self.CO = C/O

        #log_CH_solar = 8.43 - 12 # Asplund et al. (2009)
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        self.FeH = np.log10(C/H) - log_CH_solar
        self.CH  = self.FeH

        self.CO = np.mean(self.CO)
        self.FeH = np.mean(self.FeH)
        self.CH = np.mean(self.CH)

        # Remove certain species
        self.remove_species()

        return self.mass_fractions

class EqChemistry(Chemistry):

    def __init__(self, line_species, pressure, quench_setup={}, **kwargs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure)

        # Retrieve the mass ratios of the isotopologues
        self.mass_ratio_13CO_12CO = self.read_species_info('13CO', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_C18O_12CO = self.read_species_info('C18O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_C17O_12CO = self.read_species_info('C17O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        
        # Load the interpolation function
        import petitRADTRANS.poor_mans_nonequ_chem as pm
        self.pm_interpol_abundances = pm.interpol_abundances

        # Species to quench per quench pressure
        self.quench_setup = quench_setup
        
    def get_pRT_mass_fractions(self, params):

        # Retrieve the mass fractions from the chem-eq table
        pm_mass_fractions = self.pm_interpol_abundances(
            self.CO*np.ones(self.n_atm_layers), 
            self.FeH*np.ones(self.n_atm_layers), 
            self.temperature, 
            self.pressure
            )
        
        # Fill in the dictionary with the right keys
        self.mass_fractions = {
            'MMW': pm_mass_fractions['MMW']
            }
        
        for line_species_i in self.line_species:

            if line_species_i in ['CO_main_iso', 'CO_high']:
                # 12CO mass fraction
                self.mass_fractions[line_species_i] = \
                    (1 - self.C13_12_ratio * self.mass_ratio_13CO_12CO - \
                     self.O18_16_ratio * self.mass_ratio_C18O_12CO - \
                     self.O17_16_ratio * self.mass_ratio_C17O_12CO
                    ) * pm_mass_fractions['CO']
                continue
                
            if line_species_i in ['CO_36', 'CO_36_high']:
                # 13CO mass fraction
                self.mass_fractions[line_species_i] = \
                    self.C13_12_ratio * self.mass_ratio_13CO_12CO * \
                    pm_mass_fractions['CO']
                continue
            
            if line_species_i in ['CO_28', 'CO_28_high']:
                # C18O mass fraction
                self.mass_fractions[line_species_i] = \
                    self.O18_16_ratio * self.mass_ratio_C18O_12CO * \
                    pm_mass_fractions['CO']
                continue
            
            if line_species_i in ['CO_27', 'CO_27_high']:
                # C17O mass fraction
                self.mass_fractions[line_species_i] = \
                    self.O17_16_ratio * self.mass_ratio_C17O_12CO * \
                    pm_mass_fractions['CO']
                continue

            # All other species    
            species_i = line_species_i.split('_')[0]
            self.mass_fractions[line_species_i] = pm_mass_fractions.get(species_i)
        
        # Add the H2 and He abundances
        self.mass_fractions['H2'] = pm_mass_fractions['H2']
        self.mass_fractions['He'] = pm_mass_fractions['He']

        # Convert the free-chemistry VMRs to mass fractions
        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')

            if line_species_i not in self.line_species:
                continue
            if self.mass_fractions.get(line_species_i) is not None:
                continue

            # Confirm that free-chemistry VMR is defined
            assert(params.get(f'log_{species_i}') is not None)

            VMR_i = 10**params.get(f'log_{species_i}')
            self.mass_fractions[line_species_i] = VMR_i * mass_i / self.mass_fractions['MMW']

    def get_P_quench(self, params, alpha=1):

        # Metallicity
        met = 10**self.FeH

        # Scale height at each layer
        MMW = self.mass_fractions['MMW']
        H = nc.kB*self.temperature / (MMW*nc.amu*10**params['log_g'])

        # Mixing length/time-scales
        L = alpha * H
        t_mix = L**2 / params['Kzz_chem']

        self.P_quench = {
            'P_quench_CO_CH4': None, 
            'P_quench_N2_NH3': None, 
            'P_quench_HCN': None, 
            'P_quench_CO2': None, 
            }
        self.quench_setup = {
            'P_quench_CO_CH4': [
                '12CO', '13CO', 'C18O', 'C17O', 
                'CH4', '13CH4', 
                'H2O', 'H2(18)O', 'H2(17)O', 'HDO',
                ], 
            'P_quench_N2_NH3': ['N2', 'NH3'], 
            'P_quench_HCN': ['HCN'], 
            'P_quench_CO2': ['CO2'], 
        }

        # Loop from bottom to top of atmosphere
        idx = np.argsort(self.pressure)[::-1]
        for t_mix_i, P_i, T_i in zip(t_mix[idx], self.pressure[idx], self.temperature[idx]):

            if T_i < 500:
                # Avoid exponent overflow
                continue

            if None not in self.P_quench.values():
                # All quench pressures assigned
                break

            # Zahnle & Marley (2014)
            if self.P_quench.get('P_quench_CO_CH4') is None:
                # Chemical timescale of CO-CH4
                t_CO_CH4_q1 = 1.5e-6 * P_i**(-1) * met**(-0.7) * np.exp(42000/T_i)
                t_CO_CH4_q2 = 40 * P_i**(-2) * np.exp(25000/T_i)
                t_CO_CH4 = (1/t_CO_CH4_q1 + 1/t_CO_CH4_q2)**(-1)

                if t_mix_i < t_CO_CH4:
                    # Mixing is more efficient than chemical reactions
                    self.P_quench['P_quench_CO_CH4'] = P_i
        
            if self.P_quench.get('P_quench_N2_NH3') is None:
                # Chemical timescale of NH3-N2
                t_NH3 = 1.0e-7 * P_i**(-1) * np.exp(52000/T_i)

                if t_mix_i < t_NH3:
                    self.P_quench['P_quench_N2_NH3'] = P_i

            if self.P_quench.get('P_quench_HCN') is None:
                # Chemical timescale of HCN-NH3-N2
                t_HCN = 1.5e-4 * P_i**(-1) * met**(-0.7) * np.exp(36000/T_i)

                if t_mix_i < t_HCN:
                    self.P_quench['P_quench_HCN'] = P_i

            if self.P_quench.get('P_quench_CO2') is None:
                # Chemical timescale of HCN-NH3-N2
                t_CO2 = 1.0e-10 * P_i**(-0.5) * np.exp(38000/T_i)

                if t_mix_i < t_CO2:
                    self.P_quench['P_quench_CO2'] = P_i

    def quench_chemistry(self, quench_key='P_quench'):

        # Layers to be replaced by a constant abundance
        mask_quenched = (self.pressure < self.P_quench[quench_key])

        for species_i in self.quench_setup[quench_key]:

            if species_i not in self.species_info.index:
                continue

            line_species_i = self.read_species_info(species_i, 'pRT_name')
            if not line_species_i in self.line_species:
                continue

            # Store the unquenched abundance profiles
            mass_fraction_i = self.mass_fractions[line_species_i]
            #self.unquenched_mass_fractions[line_species_i] = np.copy(mass_fraction_i)
            
            # Own implementation of quenching, using interpolation
            mass_fraction_i[mask_quenched] = np.interp(
                np.log10(self.P_quench[quench_key]), 
                xp=np.log10(self.pressure), fp=mass_fraction_i
                )
            self.mass_fractions[line_species_i] = mass_fraction_i

    def __call__(self, params, temperature):

        # Update the parameters
        self.CO  = params.get('C/O')
        self.FeH = params.get('Fe/H')

        self.C13_12_ratio = params.get('C13_12_ratio', 0)
        self.O18_16_ratio = params.get('O18_16_ratio', 0)
        self.O17_16_ratio = params.get('O17_16_ratio', 0)

        self.temperature = temperature

        # Retrieve the mass fractions
        self.get_pRT_mass_fractions(params)

        self.P_quench = {}
        if params.get('log_Kzz_chem') is not None:
            # Get quenching pressures from mixing
            self.get_P_quench(params)

        self.unquenched_mass_fractions = self.mass_fractions.copy()
        for quench_key, species_to_quench in self.quench_setup.items():

            if self.P_quench.get(quench_key) is None:
                # Quench pressure is set as free parameter

                if params.get(quench_key) is None:
                    continue
                # Add to all quenching points
                self.P_quench[quench_key] = params.get(quench_key)

            # Quench this chemical network
            self.quench_chemistry(quench_key)

        # Remove certain species
        self.remove_species()

        return self.mass_fractions

class FastChemistry(Chemistry):

    def __init__(
            self, 
            line_species, 
            pressure, 
            abundance_file, 
            gas_data_file, 
            cond_data_file='none', 
            verbose_level=1, 
            use_eq_cond=True, 
            use_rainout_cond=True, 
            quench_setup={}, 
            **kwargs
            ):
        
        # Give arguments to the parent class
        super().__init__(line_species, pressure)

        # Retrieve the mass ratios of the isotopologues
        self.mass_ratio_13CO_12CO = self.read_species_info('13CO', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_C18O_12CO = self.read_species_info('C18O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_C17O_12CO = self.read_species_info('C17O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')

        # Species to quench per quench pressure
        self.quench_setup = quench_setup

        import pyfastchem as pyfc

        # Create the FastChem object
        self.fastchem = pyfc.FastChem(
            abundance_file, 
            gas_data_file, 
            cond_data_file, 
            verbose_level
            )
        
        # Create in/out-put structures for FastChem
        self.input = pyfc.FastChemInput()
        self.input.pressure = self.pressure[::-1] # Flip to decrease

        self.output = pyfc.FastChemOutput()

        # Use equilibrium condensation
        self.input.equilibrium_condensation = use_eq_cond
        # Use rainout condensation approach
        self.input.rainout_condensation     = use_rainout_cond

        # Configure FastChem's internal parameters
        #self.fastchem.setParameter('accuracyChem', 1e-5)
        self.fastchem.setParameter('accuracyChem', 1e-4)

        #self.fastchem.setParameter('nbIterationsChemCond', 100)
        #self.fastchem.setParameter('nbIterationsChem', 3000)
        #self.fastchem.setParameter('nbIterationsCond', 700)
        
        #self.fastchem.setParameter('nbIterationsNewton', 1000)
        #self.fastchem.setParameter('nbIterationsBisection', 1000)
        #self.fastchem.setParameter('nbIterationsNelderMead', 1000)

        # Compute the solar abundances, C/O and Fe/H
        self.get_solar_abundances()

        # Obtain the indices for FastChem's table
        self.get_pyfc_indices(pyfc.FASTCHEM_UNKNOWN_SPECIES)

    def get_pyfc_indices(self, FASTCHEM_UNKNOWN_SPECIES):

        self.pyfc_indices = []
        for species_i in self.species_info.index:

            line_species_i = self.read_species_info(species_i, 'pRT_name')
            pyfc_species_i = self.read_species_info(species_i, 'pyfc_name')
            mass_i = self.read_species_info(species_i, 'mass')

            if (line_species_i in self.line_species) or (line_species_i in ['H2', 'He']):

                if pyfc_species_i is None:
                    continue

                index = self.fastchem.getGasSpeciesIndex(pyfc_species_i)

                if index == FASTCHEM_UNKNOWN_SPECIES:
                    print(f'Species {pyfc_species_i}, not found in FastChem')
                    continue
                
                self.pyfc_indices.append((line_species_i, index, mass_i))

    def get_solar_abundances(self):

        # Make a copy of the solar abundances from FastChem
        self.solar_abundances = np.array(self.fastchem.getElementAbundances())

        # Indices of carbon-bearing species
        self.index_C = np.array(self.fastchem.getElementIndex('C'))
        self.index_O = np.array(self.fastchem.getElementIndex('O'))

        # Compute the solar C/O ratio
        self.CO_solar = self.solar_abundances[self.index_C] / \
            self.solar_abundances[self.index_O]

        # Indices of H/He-bearing species
        self.index_H  = np.array(self.fastchem.getElementIndex('H'))
        self.index_He = np.array(self.fastchem.getElementIndex('He'))
        
        self.mask_metal = np.ones_like(self.solar_abundances, dtype=bool)
        self.mask_metal[self.index_H]  = False
        self.mask_metal[self.index_He] = False
        
        # Compute the solar metallicity Fe/H
        self.log_FeH_solar = np.log10(
            self.solar_abundances[self.fastchem.getElementIndex('Fe')]
            )
        
    def get_pRT_mass_fractions(self):

        # Compute the volume-mixing ratio of all species
        gas_number_density_tot = np.array(self.input.pressure)*1e6 / \
            (nc.kB * np.array(self.input.temperature))
        gas_number_density     = np.array(self.output.number_densities)

        self.VMR = gas_number_density / gas_number_density_tot[:,None]
        self.VMR = self.VMR[::-1] # Flip back

        # Store in the pRT mass fractions dictionary
        self.mass_fractions = {
            'MMW': np.array(self.output.mean_molecular_weight)[::-1], 
            }
        
        for line_species_i, index, mass_i in self.pyfc_indices:
            self.mass_fractions[line_species_i] = \
                self.VMR[:,index] * mass_i / self.mass_fractions['MMW']
    
    def quench_chemistry(self, quench_key='P_quench'):

        # Layers to be replaced by a constant abundance
        mask_quenched = (self.pressure < self.P_quench[quench_key])

        for species_i in self.quench_setup[quench_key]:

            if species_i not in self.species_info.index:
                continue

            line_species_i = self.read_species_info(species_i, 'pRT_name')
            if not line_species_i in self.line_species:
                continue

            # Store the unquenched abundance profiles
            mass_fraction_i = self.mass_fractions[line_species_i]
            #self.unquenched_mass_fractions[line_species_i] = np.copy(mass_fraction_i)
            
            # Own implementation of quenching, using interpolation
            mass_fraction_i[mask_quenched] = np.interp(
                np.log10(self.P_quench[quench_key]), 
                xp=np.log10(self.pressure), fp=mass_fraction_i
                )
            self.mass_fractions[line_species_i] = mass_fraction_i
    
    def get_isotope_mass_fractions(self):
        
        for line_species_i in self.line_species:

            if (line_species_i == 'CO_main_iso') or (line_species_i == 'CO_high'):
                # 12CO mass fraction
                self.mass_fractions[line_species_i] = \
                    (1 - self.C13_12_ratio * self.mass_ratio_13CO_12CO - \
                     self.O18_16_ratio * self.mass_ratio_C18O_12CO - \
                     self.O17_16_ratio * self.mass_ratio_C17O_12CO \
                    ) * self.mass_fractions['CO_main_iso']
            
            if (line_species_i == 'CO_36') or (line_species_i == 'CO_36_high'):
                # 13CO mass fraction
                self.mass_fractions[line_species_i] = \
                    self.C13_12_ratio * self.mass_ratio_13CO_12CO * \
                    self.mass_fractions['CO_main_iso']
            
            if line_species_i == 'CO_28':
                # C18O mass fraction
                self.mass_fractions[line_species_i] = \
                    self.O18_16_ratio * self.mass_ratio_C18O_12CO * \
                    self.mass_fractions['CO_main_iso']
                
            if line_species_i == 'CO_27':
                # C17O mass fraction
                self.mass_fractions[line_species_i] = \
                    self.O17_16_ratio * self.mass_ratio_C17O_12CO * \
                    self.mass_fractions['CO_main_iso']
        
    def __call__(self, params, temperature):

        # Make a copy to modify the elemental abundances
        self.element_abundances = np.copy(self.solar_abundances)

        # Update the parameters
        self.CO  = params.get('C/O')
        self.FeH = params.get('Fe/H')

        self.C13_12_ratio = params.get('C13_12_ratio')
        self.O18_16_ratio = params.get('O18_16_ratio')
        self.O17_16_ratio = params.get('O17_16_ratio')

        if self.C13_12_ratio is None:
            self.C13_12_ratio = 0
        if self.O18_16_ratio is None:
            self.O18_16_ratio = 0
        if self.O17_16_ratio is None:
            self.O17_16_ratio = 0

        self.temperature = temperature
        self.input.temperature = self.temperature[::-1] # Flip for FastChem usage

        # Apply C/O ratio and Fe/H to elemental abundances
        self.element_abundances[self.index_C] = \
            self.element_abundances[self.index_O] * self.CO/self.CO_solar
        
        self.metallicity_wrt_solar = 10**self.FeH
        self.element_abundances[self.mask_metal] *= self.metallicity_wrt_solar

        # Update the element abundances
        self.fastchem.setElementAbundances(self.element_abundances)

        # Compute the number densities
        fastchem_flag = self.fastchem.calcDensities(self.input, self.output)

        if fastchem_flag != 0:
            # FastChem failed to converge
            #print('Failed to converge')
            self.mass_fractions = -np.inf
            return self.mass_fractions
        
        if np.amin(self.output.element_conserved) != 1:
            # Failed element conservation
            #print('Failed element conservation')
            self.mass_fractions = -np.inf
            return self.mass_fractions
        
        # Store the pRT mass fractions in a dictionary
        self.get_pRT_mass_fractions()

        # Obtain the mass fractions for the isotopologues
        self.get_isotope_mass_fractions()

        self.unquenched_mass_fractions = self.mass_fractions.copy()
        self.P_quench = {}
        for quench_key, species_to_quench in self.quench_setup.items():

            if params.get(quench_key) is None:
                continue

            # Add to all quenching points
            self.P_quench[quench_key] = params.get(quench_key)

            # Quench this chemical network
            self.quench_chemistry(quench_key)

        # Remove certain species
        self.remove_species()

        return self.mass_fractions
    
class SONORAChemistry(Chemistry):

    def __init__(
            self, 
            line_species, 
            pressure, 
            path_SONORA_chem, 
            quench_setup={}, 
            **kwargs
            ):

        # Give arguments to the parent class
        super().__init__(line_species, pressure)

        # Retrieve the mass ratios of the isotopologues
        self.mass_ratio_13CO_12CO = self.read_species_info('13CO', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_C18O_12CO = self.read_species_info('C18O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_C17O_12CO = self.read_species_info('C17O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        
        # Species to quench per quench pressure
        self.quench_setup = quench_setup

        # Prepare the interpolation functions
        self.path_SONORA_chem = path_SONORA_chem
        self.get_interp_func()

    def get_VMR_table(self):

        CO_solar = 0.458

        import glob
        all_files = np.sort(
            glob.glob(f'{self.path_SONORA_chem}/*/sonora_*.txt')
            )
        all_FeH, all_CO, all_T, all_log_P, all_species = [], [], [], [], []
        for file_i in all_files:
            # Re-format the FeH string
            FeH_i = file_i.split('feh')[-1]
            FeH_i = FeH_i.split('_')[0]
            FeH_i = float(FeH_i[:2] + '.' + FeH_i[2:])
            all_FeH.append(FeH_i)

            # Re-format the C/O string
            CO_i = file_i.split('co_')[-1]
            CO_i = CO_i.split('.')[0]
            CO_i = float(CO_i[:1] + '.' + CO_i[1:])
            # Scale to the non-solar C/O ratio
            CO_i *= CO_solar
            all_CO.append(CO_i)

            # Read the temperatures and pressures
            T_i, log_P_i = np.loadtxt(file_i, skiprows=1, usecols=(0,1)).T
            all_T.append(T_i)
            all_log_P.append(log_P_i)

            species_i = np.loadtxt(file_i, max_rows=1, dtype=str)[4:]
            all_species.append(species_i)

        # Define the grids
        self.FeH_grid   = np.unique(all_FeH)
        self.CO_grid    = np.unique(all_CO)
        self.T_grid     = np.unique(all_T)
        self.log_P_grid = np.unique(all_log_P)
        self.species_grid = all_species[0]

        # Retrieve abundances at each grid point
        all_VMR = np.nan * np.ones((
            self.FeH_grid.size, self.CO_grid.size, 
            self.T_grid.size, self.log_P_grid.size, 
            self.species_grid.size
        ))
        for i, file_i in enumerate(all_files):
            # Parameter combination
            FeH_i   = all_FeH[i]
            CO_i    = all_CO[i]
            T_i     = all_T[i]
            log_P_i = all_log_P[i]

            # Obtain the indices for parameter combination
            idx = [
                np.argwhere(self.FeH_grid==FeH_i).flatten()[0], 
                np.argwhere(self.CO_grid==CO_i).flatten()[0], 
                None, None
                #np.argwhere(self.T_grid[:,None] == T_i[None,:])[:,0], 
                #np.argwhere(self.log_P_grid[:,None] == log_P_i[None,:])[:,0], 
            ]
            
            # Fill in the table
            VMR_i = np.loadtxt(
                file_i, skiprows=1, usecols=np.arange(2,2+len(self.species_grid),1)
                )
            for k, (T_k, log_P_k) in enumerate(zip(T_i, log_P_i)):
                idx[2] = np.argwhere(self.T_grid==T_k).flatten()[0]
                idx[3] = np.argwhere(self.log_P_grid==log_P_k).flatten()[0]

                all_VMR[idx[0],idx[1],idx[2],idx[3],:] = VMR_i[k,:]

        # Compute the mean molecular weight at each grid point
        from molmass import Formula
        masses = {}
        for species_i in self.species_grid:
            try:
                f = Formula(species_i)
                mass_i = f.isotope.massnumber
            except:
                mass_i = np.nan
            masses[species_i] = mass_i
        masses = np.array(list(masses.values()))
        all_MMW = np.nansum(all_VMR * masses[None,None,None,None,:], axis=-1)

        #print(self.FeH_grid[0], self.CO_grid[0])
        #print(all_VMR[0,0,:,:,1])
        #print(all_MMW[0,:,:,0])
        #exit()

        return all_VMR, all_MMW
    
    def get_interp_func(self):

        # Obtain the table
        all_VMR, all_MMW = self.get_VMR_table()

        # Function to generate interpolation function
        def func(values):
            from scipy.interpolate import RegularGridInterpolator
            points = (
                self.FeH_grid, self.CO_grid, self.T_grid, self.log_P_grid
                )
            interp_func = RegularGridInterpolator(
                points, values, method='linear', 
                bounds_error=False, 
                #bounds_error=True, 
                #fill_value=np.nan
                fill_value=None
            )
            return interp_func

        # Store the interpolation functions
        self.interp_func = {
            'MMW': func(all_MMW)
            }
        for species_i in self.species_info.index:

            line_species_i = self.read_species_info(species_i, 'pRT_name')
            if (line_species_i not in self.line_species) and \
                (line_species_i not in ['H2', 'He']):
                continue
            
            if species_i == '12CO':
                species_i = 'CO'

            self.interp_func[species_i] = None

            if species_i not in self.species_grid:
                continue

            idx = (self.species_grid == species_i)
            self.interp_func[species_i] = func(np.log10(all_VMR[:,:,:,:,idx]))

    def quench_chemistry(self, quench_key='P_quench'):

        # Layers to be replaced by a constant abundance
        mask_quenched = (self.pressure < self.P_quench[quench_key])

        for species_i in self.quench_setup[quench_key]:

            if species_i not in self.species_info.index:
                continue

            line_species_i = self.read_species_info(species_i, 'pRT_name')
            if not line_species_i in self.line_species:
                continue

            # Store the unquenched abundance profiles
            mass_fraction_i = self.mass_fractions[line_species_i]
            #self.unquenched_mass_fractions[line_species_i] = np.copy(mass_fraction_i)
            
            # Own implementation of quenching, using interpolation
            mass_fraction_i[mask_quenched] = np.interp(
                np.log10(self.P_quench[quench_key]), 
                xp=np.log10(self.pressure), fp=mass_fraction_i
                )
            self.mass_fractions[line_species_i] = mass_fraction_i

    def get_pRT_mass_fractions(self, params):

        # Point to interpolate onto
        point = (self.FeH, self.CO, self.temperature, np.log10(self.pressure))

        VMRs = {}
        for species_i, func_i in self.interp_func.items():

            if species_i == 'MMW':
                continue

            if func_i is None:
                if params.get(f'log_{species_i}') is None:
                    continue
                # Use free-chemistry if species not included in table
                res_i = params[f'log_{species_i}'] * np.ones_like(self.pressure)
            else:
                # Interpolate the abundances
                res_i = func_i(point).flatten()

            if np.isnan(res_i).any():
                return -np.inf
            
            # Abundance interpolation performed in log-space
            VMRs[species_i] = 10**res_i

        # Fill in the dictionary with the right keys
        self.mass_fractions = {
            'MMW': self.interp_func['MMW'](point).flatten()
            }
        
        for line_species_i in self.line_species:

            if line_species_i in ['CO_main_iso', 'CO_high']:
                # 12CO mass fraction
                self.mass_fractions[line_species_i] = \
                    (1 - self.C13_12_ratio - self.O18_16_ratio - self.O17_16_ratio) * VMRs['CO']
                
            elif line_species_i in ['CO_36', 'CO_36_high']:
                # 13CO mass fraction
                self.mass_fractions[line_species_i] = self.C13_12_ratio * VMRs['CO']
            
            elif line_species_i in ['CO_28', 'CO_28_high']:
                # C18O mass fraction
                self.mass_fractions[line_species_i] = self.O18_16_ratio * VMRs['CO']
            
            elif line_species_i in ['CO_27', 'CO_27_high']:
                # C17O mass fraction
                self.mass_fractions[line_species_i] = self.O17_16_ratio * VMRs['CO']
                
            else:
                # All other species
                self.mass_fractions[line_species_i] = VMRs[line_species_i.split('_')[0]]
        
        # Add the H2 and He abundances
        self.mass_fractions['H2'] = VMRs['H2']
        self.mass_fractions['He'] = VMRs['He']

        # Convert from VMRs to mass fractions
        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')

            if line_species_i not in self.line_species:
                continue

            #print(self.mass_fractions[line_species_i].shape)
            self.mass_fractions[line_species_i] *= mass_i / self.mass_fractions['MMW']
    
    def __call__(self, params, temperature):

        # Update the parameters
        self.CO  = params.get('C/O')
        self.FeH = params.get('Fe/H')

        self.C13_12_ratio = params.get('C13_12_ratio')
        self.O18_16_ratio = params.get('O18_16_ratio')
        self.O17_16_ratio = params.get('O17_16_ratio')

        if self.C13_12_ratio is None:
            self.C13_12_ratio = 0
        if self.O18_16_ratio is None:
            self.O18_16_ratio = 0
        if self.O17_16_ratio is None:
            self.O17_16_ratio = 0

        self.temperature = temperature

        # Retrieve the mass fractions
        res = self.get_pRT_mass_fractions(params)
        if res is not None:
            return -np.inf

        self.unquenched_mass_fractions = self.mass_fractions.copy()
        self.P_quench = {}
        for quench_key, species_to_quench in self.quench_setup.items():

            if params.get(quench_key) is None:
                continue

            # Add to all quenching points
            self.P_quench[quench_key] = params.get(quench_key)

            # Quench this chemical network
            self.quench_chemistry(quench_key)

        # Remove certain species
        self.remove_species()

        return self.mass_fractions