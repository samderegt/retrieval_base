import numpy as np
import os
import h5py

from itertools import product
from tqdm import tqdm

import pyfastchem

os.environ['OMP_NUM_THREADS'] = '4'

species = [
    'Al',
    #'Al1+',
    'Al1Cl1',
    'Al1H1',
    'Al1H1O1_2',
    'C',
    #'C1+',
    'C1H1',
    'C1H1N1_1',
    'C1H1O1',
    'C1H3',
    'C1H4',
    'C1O1',
    'C1O1S1',
    'C1O2',
    'C2H2',
    'C2H4',
    'Ca',
    #'Ca1+',
    'Ca1H1',
    'Ca1H1O1',
    'Cl',
    'Cl1H1',
    'Cl1K1',
    'Cl1Na1',
    'Cr',
    #'Cr1+',
    'Cr1H1',
    'Cs',
    'Cu',
    'Cu1H1',
    'F1H1',
    'Fe',
    #'Fe1+',
    'Fe1H1',
    'Fe1S1',
    'H',
    'H1+',
    'H1-',
    'H1K1O1',
    'H1Mg1',
    'H1Mg1O1',
    'H1Mn1',
    'H1N1',
    'H1Na1',
    'H1Na1O1',
    'H1Ni1',
    'H1O1',
    'H1P1',
    'H1S1',
    'H1Si1',
    'H2',
    'H2N1',
    'H2O1',
    'H2P1',
    'H2S1',
    'H2Si1',
    'H3N1',
    'H3P1',
    'H3Si1',
    'H4Si1',
    'He',
    'Hg',
    'K',
    #'K1+',
    'Li',
    'Mg',
    #'Mg1+',
    'Mn',
    #'Mn1+',
    'N',
    'N2',
    'Na',
    #'Na1+',
    'Ni',
    #'Ni1+',
    'Ni1S1',
    'O',
    'O1P1',
    'O1S1',
    'O1Si1',
    'O1Ti1',
    'O1V1',
    'O2Ti1',
    'O2V1',
    'P',
    'P1S1',
    'P2',
    'S',
    #'S1+',
    'S1Si1',
    'S2',
    'Si',
    #'Si1+',
    'Sr',
    'Y',
    'Zn',
    'Zr',
    'e-'
]

class FastChemistry():

    def __init__(self, use_eq_cond=True, use_rainout=False, verbose_level=1):
        
        #path = os.path.dirname(pyfastchem.__file__)
        path = '/net/lem/data1/regt/fastchem'

        self.fastchem = pyfastchem.FastChem(
            f'{path}/input/element_abundances/asplund_2020_extended.dat', 
            f'{path}/input/logK/logK_extended.dat',
            f'{path}/input/logK/logK_condensates.dat', 
            verbose_level
            )
        
        self.gas_species_tot = self.fastchem.getGasSpeciesNumber()
        
        # Configure FastChem's internal parameters
        self.fastchem.setParameter('accuracyChem', 1e-5)
        #self.fastchem.setParameter('accuracyElementConservation', 1e-5)
        #self.fastchem.setParameter('accuracyNewton', 1e-5)
        
        self.fastchem.setParameter('nbIterationsChem', 100000)
        #self.fastchem.setParameter('nbIterationsNewton', 10000)
        #self.fastchem.setParameter('nbIterationsNelderMead', 10000)

        # Create in/out-put structures for FastChem
        self.input  = pyfastchem.FastChemInput()
        self.output = pyfastchem.FastChemOutput()

        # Whether to use condensation and/or rainoutprint(len(species))
        self.input.equilibrium_condensation = use_eq_cond
        self.input.rainout_condensation     = use_rainout

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
        self.solar_abund = np.array(
            self.fastchem.getElementAbundances()
            )
        
        # Solar abundance ratios
        self.solar_CO = self.solar_abund[self.idx['C']] \
            / self.solar_abund[self.idx['O']]
        self.solar_NO = self.solar_abund[self.idx['N']] \
            / self.solar_abund[self.idx['O']]
        
        self.solar_FeH = 0.0
                
    def _set_metallicity(self, FeH):

        #tot_abund = np.sum(self.el_abund)
        self.el_abund[self.metal_idx] *= 10**FeH
        #self.el_abund *= tot_abund/np.sum(self.el_abund)
    
    def _set_CO(self, CO):
        
        # C = C/O * O_sol
        self.el_abund[self.idx['C']] = CO * self.el_abund[self.idx['O']]

        # Correct for the summed abundance of C+O
        tot_abund_ratio = (1+self.solar_CO) / (1+CO)
        self.el_abund[self.idx['C']] *= tot_abund_ratio
        self.el_abund[self.idx['O']] *= tot_abund_ratio
    
    def _set_NO(self, NO):

        # N = N/O * O_sol
        self.el_abund[self.idx['N']] = NO * self.el_abund[self.idx['O']]

        # Correct for the summed abundance of N+O
        tot_abund_ratio = (1+self.solar_NO) / (1+NO)
        self.el_abund[self.idx['N']] *= tot_abund_ratio
        self.el_abund[self.idx['O']] *= tot_abund_ratio

    def _save_to_hdf5(self, el, arr, key='log_VMR', save_path='./data/'):

        file = f'{save_path}/{el}.hdf5'
        try:
            # Load previous save
            with h5py.File(file, 'r') as f:
                s_arr = f[key][...]

            # Update the non-zero elements
            mask = (arr != 0.)
            s_arr[mask] = arr[mask]

        except FileNotFoundError:
            # No previous save
            s_arr = arr

        with h5py.File(file, 'w') as f:
            # Save array
            _ = f.create_dataset(
                key, compression='gzip', data=s_arr
            )
    
    def get_VMRs(self, P, T, **kwargs):

        # Modify the elemental abundances, initially solar
        self.el_abund = self.solar_abund.copy()

        CO = kwargs.get('CO', self.solar_CO)
        self._set_CO(CO)

        NO = kwargs.get('NO', self.solar_NO)
        self._set_NO(NO)

        FeH = kwargs.get('FeH', self.solar_FeH)
        self._set_metallicity(FeH)
        
        # Update the element abundances
        self.fastchem.setElementAbundances(self.el_abund)

        # Set the temperature and pressure
        self.input.temperature = np.array([T]).flatten()
        self.input.pressure    = np.array([P]).flatten()

        # Compute the number densities
        fastchem_flag = self.fastchem.calcDensities(
            self.input, self.output
            )
        
        if fastchem_flag != 0:
            # FastChem failed to converge
            print('Failed to converge')
        if np.amin(self.output.element_conserved) != 1:
            # Failed element conservation
            print('Failed element conservation')

        # Species-specific and total number densities
        n = np.array(self.output.number_densities)
        n_tot = n.sum(axis=1)

        # Volume-mixing ratio
        VMR = n / n_tot[:,None]
        idx = [self.fastchem.getGasSpeciesIndex(el) for el in self.species]

        # Mean-molecular weight
        MMW = self.output.mean_molecular_weight

        return VMR[:,idx], MMW

    def run_grid(self, species, P_grid, T_grid, CO_grid, NO_grid, FeH_grid, save_steps=2000, save_path='./data/'):

        self.species = species

        self.P_grid, self.T_grid, \
        self.CO_grid, self.NO_grid, self.FeH_grid \
            = P_grid, T_grid, CO_grid, NO_grid, FeH_grid
        
        shape = (
            len(self.species), len(P_grid), len(T_grid), 
            len(CO_grid), len(NO_grid), len(FeH_grid)
            ) 
        self.log_VMR_grid = np.ones(shape, dtype=np.float64)
        self.MMW_grid     = np.zeros_like(self.log_VMR_grid[0])
        
        iterables = list(product(T_grid, CO_grid, NO_grid, FeH_grid))

        # Make a nice progress bar
        pbar_kwargs = dict(
            total=len(iterables), 
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}', 
            )
        with tqdm(**pbar_kwargs) as pbar:

            for i, (T_i, CO_i, NO_i, FeH_i) in enumerate(iterables):
                
                pbar.set_postfix(
                    T='{:.0f} K'.format(T_i), 
                    CO='{:.2f}'.format(CO_i), 
                    NO='{:.2f}'.format(NO_i), 
                    FeH='{:.2f}'.format(FeH_i), 
                    refresh=True
                    )
                
                # Obtain the VMRs for these species and atmospheric conditions
                VMR_i, MMW_i = self.get_VMRs(
                    P=P_grid, T=T_i*np.ones_like(P_grid), 
                    CO=CO_i, NO=NO_i, FeH=FeH_i, 
                    )
                VMR_i[VMR_i==0.] += 1e-250
                log_VMR_i = np.around(np.log10(VMR_i), decimals=3)
                
                # Indices where to insert current VMRs
                idx_T   = np.searchsorted(self.T_grid, T_i)
                idx_CO  = np.searchsorted(self.CO_grid, CO_i)
                idx_NO  = np.searchsorted(self.NO_grid, NO_i)
                idx_FeH = np.searchsorted(self.FeH_grid, FeH_i)

                self.log_VMR_grid[:,:,idx_T,idx_CO,idx_NO,idx_FeH] = log_VMR_i.T
                self.MMW_grid[:,idx_T,idx_CO,idx_NO,idx_FeH] = MMW_i
                
                if (i % save_steps == 0) or (i+1 == len(iterables)):
                    # Save VMRs and MMW to an hdf5 file
                    self._save_to_hdf5(
                        'MMW', self.MMW_grid, key='MMW', save_path=save_path
                        )

                    for el_j, log_VMR_j in zip(self.species, self.log_VMR_grid):
                        self._save_to_hdf5(
                            el_j, log_VMR_j, key='log_VMR', save_path=save_path
                            )
                    
                pbar.update(1)

        self.species = np.unique(self.species)

FC = FastChemistry()

species = [
    FC.fastchem.getGasSpeciesSymbol(idx) \
    for idx in range(FC.fastchem.getGasSpeciesNumber())
    ]

P_grid   = 10**np.arange(-6,3+1e-6,0.1)
T_grid   = np.arange(150,6000+1e-6,50)
CO_grid  = np.arange(0.1,1.6+1e-6,0.1)
NO_grid  = [FC.solar_NO] #np.arange(0.0,0.5+1e-6,0.1)
FeH_grid = np.arange(-10,10+1e-6,1)/10
'''
# Instantiate the parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--FeH', type=float, default=None)
args = parser.parse_args()

FeH_grid = np.array([args.FeH])
'''
FC.run_grid(
    species=species, 
    P_grid=P_grid, 
    T_grid=T_grid, 
    CO_grid=CO_grid, 
    NO_grid=NO_grid, 
    FeH_grid=FeH_grid, 
    #save_path='/net/lem/data2/regt/fastchem_tables/FeH_{:.1f}/'.format(FeH_grid[0])
    save_path='/net/lem/data2/regt/fastchem_tables/'
)