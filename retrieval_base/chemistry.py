import numpy as np

import petitRADTRANS.poor_mans_nonequ_chem as pm

class Chemistry:

    # Dictionary with info per molecular/atomic species
    # (line_species name, mass, number of (C,O,H) atoms
    species_info = {
        '12CO': ('CO_main_iso', 12.011 + 15.999, (1,1,0)), 
        'H2O': ('H2O_main_iso', 2*1.00784 + 15.999, (0,1,2)), 
        'CH4': ('CH4_hargreaves_main_iso', 12.011 + 4*1.00784, (1,0,4)), 
        '13CO': ('CO_36', 13.003355 + 15.999, (1,1,0)), 
        'C18O': ('CO_28', 12.011 + 17.9991610, (1,1,0)), 
        'H2O_181': ('H2O_181', 2*1.00784 + 17.9991610, (0,1,2)), 
        'NH3': ('NH3_main_iso', 14.0067 + 3*1.00784, (0,0,3)), 
        'CO2': ('CO2_main_iso', 12.011 + 2*15.999, (1,2,0)),
        'HCN': ('HCN_main_iso', 1.00784 + 12.011 + 14.0067, (1,0,1)), 
        'He': ('He', 4.002602, (0,0,0)), 
        'H2': ('H2', 2*1.00784, (0,0,2)), 
        }

    species_plot_info = {
        '12CO': ('C2', r'$^{12}$CO'), 
        'H2O': ('C3', r'H$_2$O'), 
        'CH4': ('C4', r'CH$_4$'), 
        '13CO': ('C5', r'$^{13}$CO'), 
        'C18O': ('C6', r'C$^{18}$O'), 
        'H2O_181': ('C7', r'H$_2^{18}$O'), 
        'NH3': ('C8', r'NH$_3$'), 
        'CO2': ('C9', r'CO$_2$'),
        'HCN': ('C10', r'HCN'), 
        }

    # Neglect certain species to find respective contribution
    neglect_species = {
        '12CO': False, 
        'H2O': False, 
        'CH4': False, 
        '13CO': False, 
        'C18O': False, 
        'H2O_181': False, 
        'NH3': False, 
        'CO2': False,
        'HCN': False, 
        #'He': False, 
        #'H2': False, 
        }

    def __init__(self, line_species, pressure):

        self.line_species = line_species

        self.pressure     = pressure
        self.n_atm_layers = len(self.pressure)

    def remove_species(self, species):

        # Remove the contribution of the specified species
        for species_i in species:
            
            # Read the name of the pRT line species
            line_species_i = self.read_species_info(species_i, 'pRT_name')

            # Set mass fraction to negligible values
            # TODO: does 0 work?
            if line_species_i in line_species:
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

    def __init__(self, line_species, pressure):

        # Give arguments to the parent class
        super().__init__(line_species, pressure)

    def __call__(self, VMRs):

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
                # Convert VMR to mass fraction using molecular mass number
                self.mass_fractions[line_species_i] = mass_i * self.VMRs[species_i]
                VMR_wo_H2 += self.VMRs[species_i]

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * self.VMRs[species_i]
                O += COH_i[1] * self.VMRs[species_i]
                H += COH_i[2] * self.VMRs[species_i]

        # Add the H2 and He abundances
        self.mass_fractions['He'] = self.read_species_info('He', 'mass') * VMR_He
        self.mass_fractions['H2'] = self.read_species_info('H2', 'mass') * (1 - VMR_wo_H2)

        # Add to the H-bearing species
        H += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)

        if VMR_wo_H2 > 1:
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

        for species_i in self.neglect_species:
            if self.neglect_species[species_i]:
                line_species_i = self.read_species_info(species_i, 'pRT_name')

                # Set abundance to 0 to evaluate species' contribution
                self.mass_fractions[line_species_i] *= 0

        return self.mass_fractions

class EqChemistry(Chemistry):

    def __init__(self, line_species, pressure):

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

        # Own implementation of quenching, using interpolation

        # Layers to be replaced by a constant abundance
        mask_quenched = (self.pressure < self.P_quench)

        for species_i in ['CO', 'CH4', 'CO2', 'HCN']:
            mass_fraction_i = pm_mass_fractions[species_i]
            mass_fraction_i[mask_quenched] = np.interp(self.P_quench, 
                                                       xp=self.pressure, 
                                                       fp=mass_fraction_i
                                                       )
            pm_mass_fractions[species_i] = mass_fraction_i

        return pm_mass_fractions

    def __call__(self, params):

        # Update the parameters
        self.CO = params['C/O']
        self.FeH = params['Fe/H']
        self.P_quench = params['P_quench']

        self.C_ratio = params['C_ratio']
        self.O_ratio = params['O_ratio']

        # Retrieve the mass fractions from the chem-eq table
        pm_mass_fractions = pm.interpol_abundances(self.CO*np.ones(self.n_atm_layers), 
                                                   self.FeH*np.ones(self.n_atm_layers), 
                                                   self.temperature, 
                                                   self.pressure
                                                   )
        
        self.mass_fractions = {'MMW': pm_mass_fractions['MMW']}

        if self.P_quench is not None:
            pm_mass_fractions = self.quench_carbon_chemistry(pm_mass_fractions)

        for line_species_i in line_species:
            if species_i == 'CO_main_iso':
                # 12CO mass fraction
                self.mass_fractions[species_i] = (1 - self.C_ratio * self.mass_ratio_13CO_12CO - \
                                                  self.O_ratio * self.mass_ratio_C18O_12CO
                                                  ) * pm_mass_fractions['CO']
            elif species_i == 'CO_36':
                # 13CO mass fraction
                self.mass_fractions['CO_36'] = self.C_ratio * self.mass_ratio_13CO_12CO * pm_mass_fractions['CO']
            elif species_i == 'CO_28':
                # C18O mass fraction
                self.mass_fractions['CO_28'] = self.O_ratio * self.mass_ratio_C18O_12CO * pm_mass_fractions['CO']
            else:
                self.mass_fractions[species_i] = pm_mass_fractions[species_i.split('_')[0]]

        # Add the H2 and He abundances
        self.mass_fractions['H2'] = pm_mass_fractions['H2']
        self.mass_fractions['He'] = pm_mass_fractions['He']

        for species_i in self.neglect_species:
            if self.neglect_species[species_i]:
                line_species_i = self.read_species_info(species_i, 'pRT_name')

                # Set abundance to 0 to evaluate species' contribution
                self.mass_fractions[line_species_i] *= 0

        return self.mass_fractions