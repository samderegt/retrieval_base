import numpy as np

import petitRADTRANS.poor_mans_nonequ_chem as pm
import petitRADTRANS.nat_cst as nc

class Chemistry:

    # Dictionary with info per molecular/atomic species
    # (line_species name, mass, number of (C,O,H) atoms
    species_info = {
        '12CO': ('CO_main_iso', 12.011 + 15.999, (1,1,0)), 
        #'12CO': ('CO_high', 12.011 + 15.999, (1,1,0)), 
        'H2O': ('H2O_pokazatel_main_iso', 2*1.00784 + 15.999, (0,1,2)), 
        'CH4': ('CH4_hargreaves_main_iso', 12.011 + 4*1.00784, (1,0,4)), 
        '13CO': ('CO_36', 13.003355 + 15.999, (1,1,0)), 
        #'13CO': ('CO_36_high', 13.003355 + 15.999, (1,1,0)), 
        'C18O': ('CO_28', 12.011 + 17.9991610, (1,1,0)), 
        'H2O_181': ('H2O_181', 2*1.00784 + 17.9991610, (0,1,2)), 
        'NH3': ('NH3_coles_main_iso', 14.0067 + 3*1.00784, (0,0,3)), 
        'CO2': ('CO2_main_iso', 12.011 + 2*15.999, (1,2,0)),
        'HCN': ('HCN_main_iso', 1.00784 + 12.011 + 14.0067, (1,0,1)), 
        'He': ('He', 4.002602, (0,0,0)), 
        'H2': ('H2', 2*1.00784, (0,0,2)), 
        }

    species_plot_info = {
        '12CO': ('C2', r'$^{12}$CO'), 
        'H2O': ('C3', r'H$_2$O'), 
        'CH4': ('C4', r'CH$_4$'), 
        #'13CO': ('C5', r'$^{13}$CO'), 
        '13CO': ('chocolate', r'$^{13}$CO'), 
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

    def quench_chemistry(self, pm_mass_fractions):

        for P_quench, species_to_quench in [
            (self.P_quench_CO_CH4, ['CO', 'CH4', 'H2O']), 
            (self.P_quench_NH3, ['NH3']), 
            (self.P_quench_HCN, ['HCN']), 
            (self.P_quench_CO2, ['CO2']), 
            ]:

            if P_quench is None:
                continue

            # Layers to be replaced by a constant abundance
            mask_quenched = (self.pressure < P_quench)

            for species_i in species_to_quench:
                # Own implementation of quenching, using interpolation
                mass_fraction_i = pm_mass_fractions[species_i]
                mass_fraction_i[mask_quenched] = 10**np.interp(
                    np.log10(P_quench), xp=np.log10(self.pressure), 
                    fp=np.log10(mass_fraction_i)
                    )
                #mass_fraction_i[mask_quenched] = np.interp(P_quench, 
                #                                        xp=self.pressure, 
                #                                        fp=mass_fraction_i
                #                                        )
                pm_mass_fractions[species_i] = mass_fraction_i

        return pm_mass_fractions
    
    def get_quench_pressure(self, pm_mass_fractions, alpha=1):

        def interp_P_quench(t_mix, t_chem, P):

            chem_mix_ratio = t_chem / t_mix
            ratio_increase = np.diff(chem_mix_ratio, append=True) > 0
            if not ratio_increase.any():
                return None

            indices = np.arange(len(t_chem))[ratio_increase]

            for i, idx_i in enumerate(indices):
                if i == 0:
                    idx_low = idx_i
                elif idx_i != idx_high+1:
                    break
                idx_high = idx_i

            if idx_low == idx_high:
                return P[idx_low]
            
            xp = np.log10(chem_mix_ratio[idx_low:idx_high+1])
            fp = np.log10(P[idx_low:idx_high+1])

            P_quench = 10**np.interp(
                0, xp=xp, fp=fp, left=np.nan, right=np.nan
                )
            
            if np.isfinite(P_quench):
                return P_quench
            return None

        # Metallicity
        met = 10**self.FeH

        # Scale height at each layer
        MMW = pm_mass_fractions['MMW']
        H = nc.kB*self.temperature / (MMW*nc.amu*10**self.log_g)

        # Mixing length/time-scales
        L = alpha * H
        t_mix = L**2 / (10**self.log_Kzz)

        # --- Zahnle & Marley (2014) ------------------------
        mask_T = (self.temperature > 500) & (self.temperature < 5000)
        P = self.pressure[mask_T][::-1]
        T = self.temperature[mask_T][::-1]
        t_mix = t_mix[mask_T][::-1]
        
        # Chemical timescale of CO-CH4
        t_CO_CH4_q1 = 1.5e-6 * P**(-1) * met**(-0.7) * np.exp(42000/T)
        t_CO_CH4_q2 = 40 * P**(-2) * np.exp(25000/T)
        t_CO_CH4 = (1/t_CO_CH4_q1 + 1/t_CO_CH4_q2)**(-1)

        self.P_quench_CO_CH4 = interp_P_quench(t_mix, t_CO_CH4, P)

        # Chemical timescale of NH3-N2
        t_NH3 = 1.0e-7 * P**(-1) * np.exp(52000/T)
        
        self.P_quench_NH3 = interp_P_quench(t_mix, t_NH3, P)

        # Chemical timescale of HCN-NH3-N2
        t_HCN = 1.5e-4 * P**(-1) * met**(-0.7) * np.exp(36000/T)

        self.P_quench_HCN = interp_P_quench(t_mix, t_HCN, P)

        # Chemical timescale of HCN-NH3-N2
        t_CO2 = 1.0e-10 * P**(-0.5) * np.exp(38000/T)

        self.P_quench_CO2 = interp_P_quench(t_mix, t_CO2, P)
        
        '''
        self.P_quench_CO_CH4 = None
        self.P_quench_NH3 = None
        self.P_quench_HCN = None
        self.P_quench_CO2 = None
        for t_mix_i, P_i, T_i in zip(t_mix[::-1], self.pressure[::-1], self.temperature[::-1]):

            if T_i < 500:
                continue
            
            if (self.P_quench_CO_CH4 is not None) and (self.P_quench_NH3 is not None) \
                and (self.P_quench_HCN is not None) and (self.P_quench_CO2 is not None):
                break

            # Zahnle & Marley (2014)
            if self.P_quench_CO_CH4 is None:
                # Chemical timescale of CO-CH4
                t_CO_CH4_q1 = 1.5e-6 * P_i**(-1) * met**(-0.7) * np.exp(42000/T_i)
                t_CO_CH4_q2 = 40 * P_i**(-2) * np.exp(25000/T_i)
                t_CO_CH4 = (1/t_CO_CH4_q1 + 1/t_CO_CH4_q2)**(-1)

                if t_mix_i < t_CO_CH4:
                    # Mixing is more efficient than chemical reactions
                    self.P_quench_CO_CH4 = P_i

            if self.P_quench_NH3 is None:
                # Chemical timescale of NH3-N2
                t_NH3 = 1.0e-7 * P_i**(-1) * np.exp(52000/T_i)

                if t_mix_i < t_NH3:
                    self.P_quench_NH3 = P_i

            if self.P_quench_HCN is None:
                # Chemical timescale of HCN-NH3-N2
                t_HCN = 1.5e-4 * P_i**(-1) * met**(-0.7) * np.exp(36000/T_i)

                if t_mix_i < t_HCN:
                    self.P_quench_HCN = P_i

            if self.P_quench_CO2 is None:
                # Chemical timescale of HCN-NH3-N2
                t_CO2 = 1.0e-10 * P_i**(-0.5) * np.exp(38000/T_i)

                if t_mix_i < t_CO2:
                    self.P_quench_CO2 = P_i
        '''

    def __call__(self, params, temperature):

        # Update the parameters
        self.CO = params['C/O']
        self.FeH = params['Fe/H']
        
        self.P_quench_CO_CH4 = params.get('P_quench_CO_CH4')
        self.P_quench_NH3 = params.get('P_quench_NH3')
        self.P_quench_HCN = params.get('P_quench_HCN')
        self.P_quench_CO2 = params.get('P_quench_CO2')

        self.log_Kzz = params.get('log_Kzz_chem')
        self.log_g = params.get('log_g')

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

        if (self.log_Kzz is not None):
            self.get_quench_pressure(pm_mass_fractions)

        if (self.P_quench_CO_CH4 is not None):
            pm_mass_fractions = self.quench_chemistry(pm_mass_fractions)

        for line_species_i in self.line_species:
            if (line_species_i == 'CO_main_iso') or (line_species_i == 'CO_high'):
                # 12CO mass fraction
                self.mass_fractions[line_species_i] = \
                    (1 - self.C_ratio * self.mass_ratio_13CO_12CO - \
                     self.O_ratio * self.mass_ratio_C18O_12CO) * pm_mass_fractions['CO']
            elif (line_species_i == 'CO_36') or (line_species_i == 'CO_36_high'):
                # 13CO mass fraction
                self.mass_fractions[line_species_i] = self.C_ratio * self.mass_ratio_13CO_12CO * pm_mass_fractions['CO']
            elif line_species_i == 'CO_28':
                # C18O mass fraction
                self.mass_fractions[line_species_i] = self.O_ratio * self.mass_ratio_C18O_12CO * pm_mass_fractions['CO']
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
