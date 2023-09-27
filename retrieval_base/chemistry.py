import numpy as np
from scipy.interpolate import make_interp_spline

import petitRADTRANS.poor_mans_nonequ_chem as pm

def get_Chemistry_class(line_species, pressure, mode, **kwargs):

    if mode == 'free':
        return FreeChemistry(line_species, pressure, **kwargs)
    if mode == 'eqchem':
        return EqChemistry(line_species, pressure, **kwargs)

class Chemistry:

    # Dictionary with info per molecular/atomic species
    # (line_species name, mass, number of (C,O,H) atoms
    species_info = {
        '12CO': ('CO_main_iso', 12.011 + 15.999, (1,1,0)), 
        #'12CO': ('CO_high', 12.011 + 15.999, (1,1,0)), 
        '13CO': ('CO_36', 13.003355 + 15.999, (1,1,0)), 
        #'13CO': ('CO_36_high', 13.003355 + 15.999, (1,1,0)), 
        'C18O': ('CO_28', 12.011 + 17.9991610, (1,1,0)), 
        'C17O': ('CO_27', 12.011 + 16.999131, (1,1,0)), 

        'H2O': ('H2O_pokazatel_main_iso', 2*1.00784 + 15.999, (0,1,2)), 
        'H2O_181': ('H2O_181', 2*1.00784 + 17.9991610, (0,1,2)), 
        'HDO': ('HDO_voronin', 1.00784 + 2.014 + 15.999, (0,1,2)), 

        'CH4': ('CH4_hargreaves_main_iso', 12.011 + 4*1.00784, (1,0,4)), 
        '13CH4': ('CH4_31111_hargreaves', 13.003355 + 4*1.00784, (1,0,4)), 

        'NH3': ('NH3_coles_main_iso', 14.0067 + 3*1.00784, (0,0,3)), 
        'HCN': ('HCN_main_iso', 1.00784 + 12.011 + 14.0067, (1,0,1)), 
        #'H2S': ('H2S_main_iso', 2*1.00784 + 32.065, (0,0,2)), 
        'H2S': ('H2S_ExoMol_main_iso', 2*1.00784 + 32.065, (0,0,2)), 
        'FeH': ('FeH_main_iso', 55.845 + 1.00784, (0,0,1)), 
        'CrH': ('CrH_main_iso', 51.9961 + 1.00784, (0,0,1)), 
        'NaH': ('NaH_main_iso', 22.989769 + 1.00784, (0,0,1)), 

        'TiO': ('TiO_48_Exomol_McKemmish', 47.867 + 15.999, (0,1,0)), 
        'VO': ('VO_ExoMol_McKemmish', 50.9415 + 15.999, (0,1,0)), 
        'AlO': ('AlO_main_iso', 26.981539 + 15.999, (0,1,0)), 
        'CO2': ('CO2_main_iso', 12.011 + 2*15.999, (1,2,0)),

        'HF': ('HF_main_iso', 1.00784 + 18.998403, (0,0,1)), 
        'HCl': ('HCl_main_iso', 1.00784 + 35.453, (0,0,1)), 
        
        'H2': ('H2', 2*1.00784, (0,0,2)), 
        'HD': ('H2_12', 1.00784 + 2.014, (0,0,2)), 

        'K': ('K', 39.0983, (0,0,0)), 
        'Na': ('Na_allard', 22.989769, (0,0,0)), 
        'Ti': ('Ti', 47.867, (0,0,0)), 
        'Fe': ('Fe', 55.845, (0,0,0)), 
        'Ca': ('Ca', 40.078, (0,0,0)), 
        'Al': ('Al', 26.981539, (0,0,0)), 
        'Mg': ('Mg', 24.305, (0,0,0)), 
        'He': ('He', 4.002602, (0,0,0)), 
        }

    species_plot_info = {
        '12CO': ('C2', r'$^{12}$CO'), 
        '13CO': ('chocolate', r'$^{13}$CO'), 
        'C18O': ('C6', r'C$^{18}$O'), 
        'C17O': ('C7', r'C$^{17}$O'), 

        'H2O': ('C3', r'H$_2$O'), 
        'H2O_181': ('C7', r'H$_2^{18}$O'), 
        'HDO': ('b', r'HDO'), 

        'CH4': ('C4', r'CH$_4$'), 
        '13CH4': ('purple', r'$^{13}$CH$_4$'), 
        
        'NH3': ('C8', r'NH$_3$'), 
        'HCN': ('C10', r'HCN'), 
        'H2S': ('C11', r'H$_2$S'), 
        'FeH': ('C12', r'FeH'), 
        'CrH': ('C15', r'CrH'), 
        'NaH': ('C16', r'NaH'), 

        'TiO': ('C13', r'TiO'), 
        'VO': ('C15', r'VO'), 
        'AlO': ('C14', r'AlO'), 
        'CO2': ('C9', r'CO$_2$'),

        'HF': ('C14', r'HF'), 
        'HCl': ('C15', r'HCl'), 
        
        #'H2': ('C16', r'H$_2$'), 
        'HD': ('C17', r'HD'), 

        'K': ('C18', r'K'), 
        'Na': ('C19', r'Na'), 
        'Ti': ('C20', r'Ti'), 
        'Fe': ('C21', r'Fe'), 
        'Ca': ('C22', r'Ca'), 
        'Al': ('C23', r'Al'), 
        'Mg': ('C24', r'Mg'), 
        #'He': ('C22', r'He'), 
        }

    # Neglect certain species to find respective contribution
    neglect_species = {
        '12CO': False, 
        '13CO': False, 
        'C18O': False, 
        'C17O': False, 
        
        'H2O': False, 
        'H2O_181': False, 
        'HDO': False, 

        'CH4': False, 
        '13CH4': False, 
        
        'NH3': False, 
        'HCN': False, 
        'H2S': False, 
        'FeH': False, 
        'CrH': False, 
        'NaH': False, 

        'TiO': False, 
        'VO': False, 
        'AlO': False, 
        'CO2': False, 

        'HF': False, 
        'HCl': False, 

        #'H2': False, 
        'HD': False, 

        'K': False, 
        'Na': False, 
        'Ti': False, 
        'Fe': False, 
        'Ca': False, 
        'Al': False, 
        'Mg': False, 
        #'He': False, 
        }

    def __init__(self, line_species, pressure):

        self.line_species = line_species

        self.pressure     = pressure
        self.n_atm_layers = len(self.pressure)

        # Set to None initially, changed during evaluation
        self.mass_fractions_envelopes = None
        self.mass_fractions_posterior = None
        self.unquenched_mass_fractions_posterior = None
        self.unquenched_mass_fractions_envelopes = None

    def remove_species(self, species):

        # Remove the contribution of the specified species
        for species_i in species:
            
            # Read the name of the pRT line species
            line_species_i = self.read_species_info(species_i, 'pRT_name')

            # Set mass fraction to negligible values
            # TODO: does 0 work?
            if line_species_i in self.line_species:
                self.mass_fractions[line_species_i] = 0

    @classmethod
    def read_species_info(cls, species, info_key):
        
        if info_key == 'pRT_name':
            return cls.species_info[species][0]
        elif info_key == 'mass':
            return cls.species_info[species][1]
        elif info_key == 'C':
            return cls.species_info[species][2][0]
        elif info_key == 'O':
            return cls.species_info[species][2][1]
        elif info_key == 'H':
            return cls.species_info[species][2][2]

        elif info_key == 'c' or info_key == 'color':
            return cls.species_plot_info[species][0]
        elif info_key == 'label':
            return cls.species_plot_info[species][1]

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
        for species_i, (line_species_i, mass_i, COH_i) in self.species_info.items():

            if species_i in ['H2', 'He']:
                continue

            if line_species_i in self.line_species:

                if self.VMRs.get(species_i) is not None:
                    # Single value given: constant, vertical profile
                    VMR_i = self.VMRs[species_i] * np.ones(self.n_atm_layers)
                else:
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

        log_CH_solar = 8.43 - 12 # Asplund et al. (2009)
        self.FeH = np.log10(C/H) - log_CH_solar
        self.CH  = self.FeH

        self.CO = np.mean(self.CO)
        self.FeH = np.mean(self.FeH)
        self.CH = np.mean(self.CH)

        for species_i in self.neglect_species:
            if self.neglect_species[species_i]:
                line_species_i = self.read_species_info(species_i, 'pRT_name')

                # Set abundance to 0 to evaluate species' contribution
                self.mass_fractions[line_species_i] *= 0

        return self.mass_fractions

class EqChemistry(Chemistry):

    def __init__(self, line_species, pressure, **kwargs):

        # Give arguments to the parent class
        super().__init__(line_species, pressure)

        # Retrieve the mass ratios of the isotopologues
        self.mass_ratio_13CO_12CO = self.read_species_info('13CO', 'mass') / \
                                    self.read_species_info('12CO', 'mass')

        self.mass_ratio_C18O_12CO = self.read_species_info('C18O', 'mass') / \
                                    self.read_species_info('12CO', 'mass')
        self.mass_ratio_H2_18O_H2O = self.read_species_info('H2O_181', 'mass') / \
                                     self.read_species_info('H2O', 'mass')

    def quench_carbon_chemistry(self, pm_mass_fractions):

        # Layers to be replaced by a constant abundance
        mask_quenched = (self.pressure < self.P_quench)

        for i, species_i in enumerate(['CO', 'CH4', 'H2O']):
            
            # Store the unquenched abundance profiles
            line_species_i = self.read_species_info(['12CO', 'CH4', 'H2O'][i], 'pRT_name')
            self.unquenched_mass_fractions[line_species_i] = np.copy(pm_mass_fractions[species_i])

            # Own implementation of quenching, using interpolation
            mass_fraction_i = pm_mass_fractions[species_i]
            mass_fraction_i[mask_quenched] = np.interp(self.P_quench, 
                                                       xp=self.pressure, 
                                                       fp=mass_fraction_i
                                                       )
            pm_mass_fractions[species_i] = mass_fraction_i

        return pm_mass_fractions

    def __call__(self, params, temperature):

        # Update the parameters
        self.CO = params['C/O']
        self.FeH = params['Fe/H']
        self.P_quench = params['P_quench']

        self.C_ratio = params['C_ratio']
        self.O_ratio = params['O_ratio']

        self.temperature = temperature

        # Retrieve the mass fractions from the chem-eq table
        pm_mass_fractions = pm.interpol_abundances(self.CO*np.ones(self.n_atm_layers), 
                                                   self.FeH*np.ones(self.n_atm_layers), 
                                                   self.temperature, 
                                                   self.pressure
                                                   )
        
        self.mass_fractions = {'MMW': pm_mass_fractions['MMW']}

        if self.P_quench is not None:
            self.unquenched_mass_fractions = {}
            pm_mass_fractions = self.quench_carbon_chemistry(pm_mass_fractions)

        for line_species_i in self.line_species:
            if (line_species_i == 'CO_main_iso') or (line_species_i == 'CO_high'):
                # 12CO mass fraction
                self.mass_fractions[line_species_i] = \
                    (1 - self.C_ratio * self.mass_ratio_13CO_12CO - \
                     self.O_ratio * self.mass_ratio_C18O_12CO) * pm_mass_fractions['CO']

            elif (line_species_i == 'CO_36') or (line_species_i == 'CO_36_high'):
                # 13CO mass fraction
                self.mass_fractions[line_species_i] = \
                    self.C_ratio * self.mass_ratio_13CO_12CO * pm_mass_fractions['CO']

            elif line_species_i == 'CO_28':
                # C18O mass fraction
                self.mass_fractions[line_species_i] = \
                    self.O_ratio * self.mass_ratio_C18O_12CO * pm_mass_fractions['CO']

            else:
                self.mass_fractions[line_species_i] = pm_mass_fractions[line_species_i.split('_')[0]]

        # Add the H2 and He abundances
        self.mass_fractions['H2'] = pm_mass_fractions['H2']
        self.mass_fractions['He'] = pm_mass_fractions['He']

        for species_i in self.neglect_species:
            if self.neglect_species[species_i]:
                line_species_i = self.read_species_info(species_i, 'pRT_name')

                # Set abundance to 0 to evaluate species' contribution
                self.mass_fractions[line_species_i] *= 0

        return self.mass_fractions
