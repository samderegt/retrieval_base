import numpy as np

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'g395h_nrs12_eqchem_ret_2'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    nirspec_g395h_1 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_395h_1.dat', 'grating':'G395H', 'min_SNR':3}),  
    nirspec_g395h_2 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_395h_2.dat', 'grating':'G395H', 'min_SNR':3}),  
)

####################################################################################
# Model parameters
####################################################################################
from retrieval_base.utils import sc
M_p, sigma_M_p = 19.0, 5.0
R_p, sigma_R_p = 1.1, 0.2
g     = (sc.G*1e3) * (M_p*sc.m_jup*1e3) / (R_p*sc.r_jup_mean*1e2)**2
log_g = np.log10(g)
sigma_log_g = np.sqrt((sigma_M_p/M_p)**2 + (2*sigma_R_p/R_p)**2) / np.log(10)

# Define the priors of the parameters
free_params = {
    # Covariance parameters
    'b': ['U', (-1.0,2.0), r'$b_{395}$'], 
    'a':     ['U', (0.5,4.0), r'$a$'], 
    'log_l': ['U', (1.0,3.0), r'$\log\ l$'],

    # General properties
    'log_g': ['G', (log_g,sigma_log_g), r'$\log\ g$'],
    'R_p':   ['G', (1.1,0.2), r'$\mathrm{R_p}$'], 
    'rv':    ['U', (-7.0,0.0), r'$v_\mathrm{rad}$'], 

    # Broadening
    'vsini': ['U', (0.0,10.0), r'$v\ \sin\ i$'], 

    # Cloud properties
    'log_sigma_cl':                       ['U', (-2.0,0.0), r'$\Delta\log\sigma$'], 
    'log_P_base_Fe(s)_amorphous__Mie':    ['U', (-2.0,3.0), r'$\log P_\mathrm{Fe}$'], 
    'log_X_base_Fe(s)_amorphous__Mie':    ['U', (-12.0,-1.0), r'$\log X_\mathrm{Fe}$'], 
    'f_sed_Fe(s)_amorphous__Mie':         ['U', (0.0,10.0), r'$f_\mathrm{sed,Fe}$'], 
    'log_radius_cl_Fe(s)_amorphous__Mie': ['U', (-7.0,1.0), r'$\log r_\mathrm{Fe}$'], 

    'log_P_base_Mg2SiO4(s)_amorphous__Mie':    ['U', (-2.0,1.0), r'$\log P_\mathrm{Mg2SiO4}$'], 
    'log_X_base_Mg2SiO4(s)_amorphous__Mie':    ['U', (-12.0,-1.0), r'$\log X_\mathrm{Mg2SiO4}$'], 
    'f_sed_Mg2SiO4(s)_amorphous__Mie':         ['U', (0.0,10.0), r'$f_\mathrm{sed,Mg2SiO4}$'], 
    'log_radius_cl_Mg2SiO4(s)_amorphous__Mie': ['U', (-7.0,1.0), r'$\log r_\mathrm{Mg2SiO4}$'], 
    
    # Chemistry
    'alpha_C':  ['U', (-1.0,2.0), '[C/H]'], 
    'alpha_O':  ['U', (-1.0,2.0), '[O/H]'], 
    'alpha_N':  ['U', (-1.0,2.0), '[N/H]'], 
    'alpha_S':  ['U', (-1.0,2.0), '[S/H]'],
    '[M/H]':    ['U', (-1.0,2.0), '[M/H]'],
    'log_Kzz_chem': ['U', (3.0,13.0), r'$\log\ K_\mathrm{zz}$'],

    'log_13CO_ratio':    ['U', (1.0,4.0), r'$\log\ \mathrm{^{12}/^{13}CO}$'], 
    'log_C18O_ratio':    ['U', (1.0,4.0), r'$\log\ \mathrm{C^{16}/^{18}O}$'], 
    'log_C17O_ratio':    ['U', (1.0,4.0), r'$\log\ \mathrm{C^{16}/^{17}O}$'], 
    'log_H2(18)O_ratio': ['U', (1.0,4.0), r'$\log\ \mathrm{H_2^{16}/^{18}O}$'], 
    'log_H2(17)O_ratio': ['U', (1.0,4.0), r'$\log\ \mathrm{H_2^{16}/^{17}O}$'], 
    'log_13CH4_ratio':   ['U', (1.0,4.0), r'$\log\ \mathrm{^{12}/^{13}CH_4}$'], 

    # PT profile 
    'dlnT_dlnP_0': ['G', (0.25,0.07), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['G', (0.25,0.07), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['G', (0.17,0.07), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['G', (0.12,0.08), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['G', (0.07,0.09), r'$\nabla_4$'], 
    'dlnT_dlnP_5': ['G', (0.0,0.1), r'$\nabla_5$'], 
    'dlnT_dlnP_6': ['G', (0.0,0.1), r'$\nabla_6$'], 
    'dlnT_dlnP_7': ['G', (0.0,0.1), r'$\nabla_7$'], 
    'T_phot':      ['U', (1000.,2000.), r'$T_\mathrm{phot}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 47.2733,  # +/- 37 mas

    'epsilon_limb': 0.6, # Limb-darkening

    'log_P_phot': 0., 
    'd_log_P_phot+4': 4.,
    'd_log_P_phot+3': 3.,
    'd_log_P_phot+2': 2.,
    'd_log_P_phot+1': 1., 
    'd_log_P_phot-1': 1.,
}

####################################################################################
# Physical model keyword-arguments
####################################################################################

PT_kwargs = dict(
    shared_between_m_set = True,
    PT_mode = 'free_gradient', 
    n_knots = 7, 
    interp_mode = 'linear', 

    log_P_range = (-5.,2.), 
    n_atm_layers = 70,
)

line_species_g395h_1 = [
    '1H2-16O__POKAZATEL', '1H2-18O__HotWat78', '1H2-17O__HotWat78', 
    '12C-1H4__MM', '13C-1H4__HITRAN', 
    '12C-16O2__HITEMP', 

    '14N-1H3__CoYuTe', 
    '1H-12C-14N__Harris',
    '1H2-32S__AYT2', 
]
line_species_g395h_2 = [
    '1H2-16O__POKAZATEL', '1H2-18O__HotWat78', '1H2-17O__HotWat78', 
    '12C-1H4__MM', 
    '12C-16O__HITEMP', '13C-16O__HITEMP', '12C-18O__HITEMP', '12C-17O__HITEMP', 
    '12C-16O2__HITEMP', 

    '14N-1H3__CoYuTe', 
    '1H2-32S__AYT2', 
]
line_species = list(np.unique(line_species_g395h_1+line_species_g395h_2))

chem_kwargs = dict(
    shared_between_m_set = True,
    chem_mode = 'fastchem', 
    line_species = line_species, 

    abundance_file='/net/lem/data1/regt/fastchem/input/element_abundances/asplund_2020_simplified.dat', 
    gas_data_file='/net/lem/data1/regt/fastchem/input/logK/logK_simplified.dat', 
    cond_data_file='/net/lem/data1/regt/fastchem/input/logK/logK_condensates_simplified.dat', 
    use_rainout_cond=True, 
    min_temperature=200.,
)

cloud_kwargs = dict(
    shared_between_m_set = True,
    cloud_mode = 'EddySed', 
    cloud_species = ['Fe(s)_amorphous__Mie', 'Mg2SiO4(s)_amorphous__Mie'], 
)

rotation_kwargs = dict(
    shared_between_m_set = True,
    rotation_mode = 'convolve', 
)

pRT_Radtrans_kwargs = dict(
    nirspec_g395h_1 = dict(line_species=line_species_g395h_1),
    nirspec_g395h_2 = dict(line_species=line_species_g395h_2),

    rayleigh_species           = ['H2','He'],
    gas_continuum_contributors = ['H2-H2','H2-He'],
    cloud_species              = cloud_kwargs.get('cloud_species'), 

    line_opacity_mode             = 'lbl',
    line_by_line_opacity_sampling = 10, # Faster radiative transfer by down-sampling
    scattering_in_emission        = True, 

    # pRT_input_data_path = '/projects/0/prjs1096/pRT3/input_data', 
    pRT_input_data_path = '/net/lem/data2/pRT3_formatted/input_data', 
)

####################################################################################
# Log-likelihood, covariance keyword-arguments
####################################################################################

loglike_kwargs = dict(
    scale_flux = False, 
    scale_err = False, 
    sum_model_settings = False, 
)

cov_kwargs = dict(
    trunc_dist = 5, # For Matern, 4.16 would be equivalent to 3*sigma
    scale_amp  = True,  
    max_separation = 5 * 10**free_params.get('log_l', [None,[None,np.inf]])[1][1], 
    kernel_mode = 'matern', separation_mode = 'velocity', 
)

all_model_kwargs = dict(
    PT_kwargs=PT_kwargs, 
    chem_kwargs=chem_kwargs, 
    cloud_kwargs=cloud_kwargs, 
    rotation_kwargs=rotation_kwargs,
    pRT_Radtrans_kwargs=pRT_Radtrans_kwargs, 

    cov_kwargs=cov_kwargs, 
    loglike_kwargs=loglike_kwargs,
)

####################################################################################
# Multinest parameters
####################################################################################

pymultinest_kwargs = dict(    
    verbose = True, 

    const_efficiency_mode = True, 
    sampling_efficiency   = 0.05, 
    evidence_tolerance    = 0.5, 
    n_live_points         = 150, 
    n_iter_before_update  = 200, 
)