import numpy as np

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'g140h_nrs12_eqchem_ret_1'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    nirspec_g140h_1 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_140h_1.dat', 'grating':'G140H', 'min_SNR':3}),  
    nirspec_g140h_2 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_140h_2.dat', 'grating':'G140H', 'min_SNR':3}),  
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
    'a':     ['U', (0.,2.), r'$\log\ a$'], 
    'log_l': ['U', (1.0,2.6), r'$\log\ l$'],
    'b':     ['U', (0.0,3.0), r'$b$'],

    # General properties
    'log_g': ['G', (log_g,sigma_log_g), r'$\log\ g$'],
    'R_p': ['G', (1.1,0.2), r'$\mathrm{R_p}$'], 
    'rv':  ['U', (-10.,7.), r'$v_\mathrm{rad}$'], 

    # Broadening
    'vsini': ['U', (0.,40.), r'$v\ \sin\ i$'], 

    # Cloud properties
    'log_sigma_cl':                       ['U', (-2,0), r'$\Delta\log\sigma$'], 
    'log_P_base_Fe(s)_amorphous__Mie':    ['U', (-5,3), r'$\log P_\mathrm{Fe}$'], 
    'log_X_base_Fe(s)_amorphous__Mie':    ['U', (-10,-2), r'$\log X_\mathrm{Fe}$'], 
    'f_sed_Fe(s)_amorphous__Mie':         ['U', (0,10), r'$f_\mathrm{sed,Fe}$'], 
    'log_radius_cl_Fe(s)_amorphous__Mie': ['U', (-6,0), r'$\log r_\mathrm{Fe}$'], 

    'log_P_base_Mg2SiO4(s)_amorphous__Mie':    ['U', (-5,3), r'$\log P_\mathrm{Mg2SiO4}$'], 
    'log_X_base_Mg2SiO4(s)_amorphous__Mie':    ['U', (-10,-2), r'$\log X_\mathrm{Mg2SiO4}$'], 
    'f_sed_Mg2SiO4(s)_amorphous__Mie':         ['U', (0,10), r'$f_\mathrm{sed,Mg2SiO4}$'], 
    'log_radius_cl_Mg2SiO4(s)_amorphous__Mie': ['U', (-6,0), r'$\log r_\mathrm{Mg2SiO4}$'], 
    
    # Chemistry
    'alpha_C':  ['U', (-1.0,1.0), '[C/H]'], 
    'alpha_O':  ['U', (-1.0,1.0), '[O/H]'], 
    'alpha_N':  ['U', (-1.0,1.0), '[N/H]'], 
    'alpha_S':  ['U', (-1.0,1.0), '[S/H]'],
    'alpha_K':  ['U', (-1.0,1.0), '[K/H]'],
    'alpha_Na': ['U', (-1.0,1.0), '[Na/H]'],
    '[M/H]':    ['U', (-1.0,1.0), '[M/H]'],

    'log_Kzz_chem': ['U', (3,15), r'$\log\ K_\mathrm{zz}$'],

    # PT profile
    'dlnT_dlnP_0': ['U', (0.0,0.34), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['U', (0.0,0.34), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['U', (0.0,0.34), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['U', (0.0,0.34), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['U', (0.0,0.34), r'$\nabla_4$'], 
    'dlnT_dlnP_5': ['U', (0.0,0.34), r'$\nabla_5$'], 
    'dlnT_dlnP_6': ['U', (0.0,0.34), r'$\nabla_6$'], 
    'dlnT_dlnP_7': ['U', (0.0,0.34), r'$\nabla_7$'], 
    'T_phot':      ['U', (1000.,2600.), r'$T_\mathrm{phot}$'], 
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

line_species_g140h_1 = [
    '1H2-16O__POKAZATEL', 
    '12C-1H4__MM', 
    '1H2-32S__AYT2', 
    '1H-19F__Coxon-Hajig',
    '39K__Kurucz', 
    '23Na__Kurucz', 
    '56Fe__Kurucz',
    '56Fe-1H__MoLLIST', 
    '52Cr-1H__MoLLIST',
    '51V-16O__HyVO', 
    '48Ti-16O__Toto', 
    '40Ca__Kurucz', 
    '27Al__Kurucz', 
    '1H-12C-14N__Harris', 
    '14N-1H3__CoYuTe', 
]
line_species_g140h_2 = [
    '1H2-16O__POKAZATEL', 
    '12C-1H4__MM', 
    '1H2-32S__AYT2', 
    '1H-19F__Coxon-Hajig',
    '23Na__Kurucz', 
    '56Fe-1H__MoLLIST', 
    '48Ti__Kurucz', 
    '12C-16O__HITEMP', 
    '1H-12C-14N__Harris', 
    '14N-1H3__CoYuTe', 
]
chem_kwargs = dict(
    shared_between_m_set = True,
    chem_mode = 'fastchem', 
    line_species = list(np.unique(line_species_g140h_1+line_species_g140h_2)), 

    abundance_file='/home/sdregt/FastChem/input/element_abundances/bergemann_2025_protosolar_simplified.dat', 
    gas_data_file='/home/sdregt/FastChem/input/logK/logK_simplified.dat', 
    cond_data_file='/home/sdregt/FastChem/input/logK/logK_condensates_simplified.dat', 
    use_rainout_cond=True, 
    min_temperature=200.,
)

cloud_kwargs = dict(
    shared_between_m_set = True,
    cloud_mode = 'EddySed', 
    cloud_species = [
        'Fe(s)_amorphous__Mie', 'Mg2SiO4(s)_amorphous__Mie'
    ], 
)

rotation_kwargs = dict(
    shared_between_m_set = True,
    rotation_mode = 'convolve', 
)

pRT_Radtrans_kwargs = dict(
    # line_species = chem_kwargs.get('line_species', []),
    nirspec_g140h_1 = dict(line_species=line_species_g140h_1),
    nirspec_g140h_2 = dict(line_species=line_species_g140h_2),

    rayleigh_species           = ['H2','He'],
    gas_continuum_contributors = ['H2-H2','H2-He'],
    cloud_species              = cloud_kwargs.get('cloud_species'), 
    # nirspec_g140h_1 = dict(cloud_species=cloud_kwargs['nirspec_g140h_1']['cloud_species']), 
    # nirspec_g140h_2 = dict(cloud_species=cloud_kwargs['nirspec_g140h_2']['cloud_species']), 

    line_opacity_mode             = 'lbl',
    line_by_line_opacity_sampling = 10, # Faster radiative transfer by down-sampling
    scattering_in_emission        = True, 

    pRT_input_data_path = '/projects/0/prjs1096/pRT3/input_data', 
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
    n_live_points         = 200, 
    n_iter_before_update  = 100, 
)