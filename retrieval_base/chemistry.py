import numpy as np
from scipy.interpolate import make_interp_spline
import petitRADTRANS.nat_cst as nc

def get_Chemistry_class(line_species, pressure, mode, **kwargs):

    if mode == 'free':
        return FreeChemistry(line_species, pressure, **kwargs)
    if mode == 'eqchem':
        return EqChemistry(line_species, pressure, **kwargs)
    if mode == 'fastchem':
        return FastChemistry(line_species, pressure, **kwargs)

class Chemistry:

    # Dictionary with info per molecular/atomic species
    # (pRT_name, pyfc_name, mass, number of (C,O,H) atoms
    species_info = {
        '12CO':    ('CO_main_iso',             'C1O1',     12.011 + 15.999,            (1,1,0)), 
       #'12CO':    ('CO_high',                 'C1O1',     12.011 + 15.999,            (1,1,0)), 
        '13CO':    ('CO_36',                   None,       13.003355 + 15.999,         (1,1,0)), 
       #'13CO':    ('CO_36_high',              None,       13.003355 + 15.999,         (1,1,0)), 
        'C18O':    ('CO_28',                   None,       12.011 + 17.9991610,        (1,1,0)), 
        'C17O':    ('CO_27',                   None,       12.011 + 16.999131,         (1,1,0)), 
  
        'H2O':     ('H2O_pokazatel_main_iso',  'H2O1',     2*1.00784 + 15.999,         (0,1,2)), 
       #'H2O_181': ('H2O_181',                 None,       2*1.00784 + 17.9991610,     (0,1,2)), 
       #'HDO':     ('HDO_voronin',             None,       1.00784 + 2.014 + 15.999,   (0,1,2)), 
  
        'CH4':     ('CH4_hargreaves_main_iso', 'C1H4',     12.011 + 4*1.00784,         (1,0,4)), 
       #'13CH4':   ('CH4_31111_hargreaves',    None,       13.003355 + 4*1.00784,      (1,0,4)), 
  
        'NH3':     ('NH3_coles_main_iso',      'H3N1',     14.0067 + 3*1.00784,        (0,0,3)), 
        'HCN':     ('HCN_main_iso',            'C1H1N1_1', 1.00784 + 12.011 + 14.0067, (1,0,1)), 
       #'H2S':     ('H2S_main_iso',            'H2S1',     2*1.00784 + 32.065,         (0,0,2)), 
        'H2S':     ('H2S_ExoMol_main_iso',     'H2S1',     2*1.00784 + 32.065,         (0,0,2)), 
        'FeH':     ('FeH_main_iso',            'Fe1H1',    55.845 + 1.00784,           (0,0,1)), 
        'CrH':     ('CrH_main_iso',            'Cr1H1',    51.9961 + 1.00784,          (0,0,1)), 
        'NaH':     ('NaH_main_iso',            'H1Na1',    22.989769 + 1.00784,        (0,0,1)), 

        'TiO':     ('TiO_48_Exomol_McKemmish', 'O1Ti1',    47.867 + 15.999,            (0,1,0)), 
        'VO':      ('VO_ExoMol_McKemmish',     'O1V1',     50.9415 + 15.999,           (0,1,0)), 
        'AlO':     ('AlO_main_iso',            'Al1O1',    26.981539 + 15.999,         (0,1,0)), 
        'CO2':     ('CO2_main_iso',            'C1O2',     12.011 + 2*15.999,          (1,2,0)),
    
        'HF':      ('HF_main_iso',             'F1H1',     1.00784 + 18.998403,        (0,0,1)), 
        'HCl':     ('HCl_main_iso',            'Cl1H1',    1.00784 + 35.453,           (0,0,1)), 
        
        'H2':      ('H2',                      'H2',       2*1.00784,                  (0,0,2)), 
       #'HD':      ('H2_12',                   None,       1.00784 + 2.014,            (0,0,2)), 

        'K':       ('K',                       'K',        39.0983,                    (0,0,0)), 
        'Na':      ('Na_allard',               'Na',       22.989769,                  (0,0,0)), 
        'Ti':      ('Ti',                      'Ti',       47.867,                     (0,0,0)), 
        'Fe':      ('Fe',                      'Fe',       55.845,                     (0,0,0)), 
        'Ca':      ('Ca',                      'Ca',       40.078,                     (0,0,0)), 
        'Al':      ('Al',                      'Al',       26.981539,                  (0,0,0)), 
        'Mg':      ('Mg',                      'Mg',       24.305,                     (0,0,0)), 
        'He':      ('He',                      'He',       4.002602,                   (0,0,0)), 
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
            return cls.species_info[species][0]
        if info_key == 'pyfc_name':
            return cls.species_info[species][1]
        
        if info_key == 'mass':
            return cls.species_info[species][2]
        
        if info_key == 'COH':
            return cls.species_info[species][3]
        if info_key == 'C':
            return cls.species_info[species][3][0]
        if info_key == 'O':
            return cls.species_info[species][3][1]
        if info_key == 'H':
            return cls.species_info[species][3][2]

        if info_key == 'c' or info_key == 'color':
            return cls.species_plot_info[species][0]
        if info_key == 'label':
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

        for species_i in self.species_info.keys():
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
        
    def get_pRT_mass_fractions(self):

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
                
            elif line_species_i in ['CO_36', 'CO_36_high']:
                # 13CO mass fraction
                self.mass_fractions[line_species_i] = \
                    self.C13_12_ratio * self.mass_ratio_13CO_12CO * \
                    pm_mass_fractions['CO']
            
            elif line_species_i in ['CO_28', 'CO_28_high']:
                # C18O mass fraction
                self.mass_fractions[line_species_i] = \
                    self.O18_16_ratio * self.mass_ratio_C18O_12CO * \
                    pm_mass_fractions['CO']
            
            elif line_species_i in ['CO_27', 'CO_27_high']:
                # C17O mass fraction
                self.mass_fractions[line_species_i] = \
                    self.O17_16_ratio * self.mass_ratio_C17O_12CO * \
                    pm_mass_fractions['CO']
                
            else:
                # All other species
                self.mass_fractions[line_species_i] = \
                    pm_mass_fractions[line_species_i.split('_')[0]]
        
        # Add the H2 and He abundances
        self.mass_fractions['H2'] = pm_mass_fractions['H2']
        self.mass_fractions['He'] = pm_mass_fractions['He']

    def quench_chemistry(self, quench_key='P_quench'):

        # Layers to be replaced by a constant abundance
        mask_quenched = (self.pressure < self.P_quench[quench_key])

        for species_i in self.quench_setup[quench_key]:

            if self.species_info.get(species_i) is None:
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
            #mass_fraction_i[mask_quenched] = np.interp(self.P_quench, 
            #                                           xp=self.pressure, 
            #                                           fp=mass_fraction_i
            #                                           )

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
        self.get_pRT_mass_fractions()

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
        for species_i in self.species_info.keys():

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

            if self.species_info.get(species_i) is None:
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
            #mass_fraction_i[mask_quenched] = np.interp(self.P_quench, 
            #                                           xp=self.pressure, 
            #                                           fp=mass_fraction_i
            #                                           )
    
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