import numpy as np

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'g140h_nrs1_freechem_ret_1'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    nirspec_g140h_1 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_140h_1.dat', 'grating':'G140H', 'min_SNR':3}),  
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
    'b': ['U', (0.0,2.0), r'$b_{140}$'], 
    'a':     ['U', (0.5,3.0), r'$a$'], 
    'log_l': ['U', (1.0,2.6), r'$\log\ l$'],

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
    'log_H2O': ['U', (-6.0,-2.0), r'$\log\ \mathrm{H_2O}$'],
    'log_CH4': ['U', (-8.0,-2.0), r'$\log\ \mathrm{CH_4}$'], 
    
    'log_NH3': ['U', (-8.0,-4.0), r'$\log\ \mathrm{NH_3}$'], 
    'log_H2S': ['U', (-6.0,-2.0), r'$\log\ \mathrm{H_2S}$'],
    'log_HF':  ['U', (-12.0,-6.0), r'$\log\ \mathrm{HF}$'],

    'log_FeH':   ['U', (-14.0,-4.0), r'$\log\ \mathrm{FeH}$'], 
    'log_FeH_P': ['U', (-1.0,1.0), r'$\log\ \mathrm{FeH_P}$'],
    'FeH_alpha': ['U', (0.0,10.0), r'$\alpha_\mathrm{FeH}$'],
    'log_CrH':   ['U', (-14.0,-4.0), r'$\log\ \mathrm{CrH}$'], 
    'log_CrH_P': ['U', (-1.0,1.0), r'$\log\ \mathrm{CrH_P}$'],
    'CrH_alpha': ['U', (0.0,10.0), r'$\alpha_\mathrm{CrH}$'],
    
    'log_VO':  ['U', (-14.0,-6.0), r'$\log\ \mathrm{VO}$'],
    'log_TiO': ['U', (-14.0,-6.0), r'$\log\ \mathrm{TiO}$'],

    'log_K':   ['U', (-8.0,-4.0), r'$\log\ \mathrm{K}$'],
    'log_Na':  ['U', (-8.0,-4.0), r'$\log\ \mathrm{Na}$'],
    'log_Fe':  ['U', (-14.0,-4.0), r'$\log\ \mathrm{Fe}$'],

    'log_H2(18)O_ratio': ['U', (1.0,4.0), r'$\log\ \mathrm{H_2^{16}/^{18}O}$'], 

    # PT profile 
    'dlnT_dlnP_0': ['G', (0.25,0.07), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['G', (0.26,0.07), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['G', (0.2,0.07), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['G', (0.12,0.07), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['G', (0.07,0.07), r'$\nabla_4$'], 
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

line_species_g140h_1 = [
    '1H2-16O__POKAZATEL', '1H2-18O__HotWat78', 
    '12C-1H4__MM', 

    '14N-1H3__CoYuTe', 
    # '1H-12C-14N__Harris', 
    '1H2-32S__AYT2', 
    '1H-19F__Coxon-Hajig',
    
    '56Fe-1H__MoLLIST', 
    '52Cr-1H__MoLLIST',
    '51V-16O__HyVO', 
    '48Ti-16O__Toto', 

    '39K__Kurucz', 
    '23Na__Kurucz', 
    '56Fe__Kurucz',
]
line_species = list(np.unique(line_species_g140h_1))
chem_kwargs = dict(
    shared_between_m_set = True,
    chem_mode = 'free', 
    line_species = line_species, 
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
    nirspec_g140h_1 = dict(line_species=line_species_g140h_1),

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
    n_live_points         = 200, 
    n_iter_before_update  = 200, 
)