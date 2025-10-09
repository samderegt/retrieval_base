import numpy as np

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'g235h_nrs12_freechem_ret_9'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    # nirspec_g235h_1 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_235h_1.dat', 'grating':'G235H', 'min_SNR':3}),  
    # nirspec_g235h_2 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_235h_2.dat', 'grating':'G235H', 'min_SNR':3}),  
    nirspec_g235h_12_A = dict(instrument='JWST', kwargs={'file':'./data/nirspec_235h_12.dat', 'grating':'G235H', 'min_SNR':3}),  
    # nirspec_g235h_12_B = dict(instrument='JWST', kwargs={'file':'./data/nirspec_235h_12.dat', 'grating':'G235H', 'min_SNR':3}),  
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
    'b': ['U', (0.0,2.0), r'$b_{235}$'], 
    # 'a':     ['U', (0.5,4.0), r'$a$'], 
    # 'log_l': ['U', (1.0,3.0), r'$\log\ l$'],

    # General properties
    'log_g': ['G', (log_g,sigma_log_g), r'$\log\ g$'],
    'R_p':   ['G', (1.1,0.2), r'$\mathrm{R_p}$'], 
    'rv':    ['U', (-7.0,0.0), r'$v_\mathrm{rad}$'], 

    # Broadening
    'vsini': ['U', (0.0,10.0), r'$v\ \sin\ i$'], 

    # Cloud properties
    # 'cloud_fraction': ['U', (0.0,1.0), 'cf'], 
    # 'log_sigma_cl':                       ['U', (-2.0,0.0), r'$\Delta\log\sigma$'], 
    # 'log_P_base_Fe(s)_amorphous__Mie':    ['U', (-2.0,3.0), r'$\log P_\mathrm{Fe}$'], 
    # 'log_X_base_Fe(s)_amorphous__Mie':    ['U', (-12.0,-1.0), r'$\log X_\mathrm{Fe}$'], 
    # 'f_sed_Fe(s)_amorphous__Mie':         ['U', (0.0,10.0), r'$f_\mathrm{sed,Fe}$'], 
    # 'log_radius_cl_Fe(s)_amorphous__Mie': ['U', (-7.0,1.0), r'$\log r_\mathrm{Fe}$'], 

    # 'log_P_base_Mg2SiO4(s)_amorphous__Mie':    ['U', (-2.0,1.0), r'$\log P_\mathrm{Mg2SiO4}$'], 
    # 'log_X_base_Mg2SiO4(s)_amorphous__Mie':    ['U', (-12.0,-1.0), r'$\log X_\mathrm{Mg2SiO4}$'], 
    # 'f_sed_Mg2SiO4(s)_amorphous__Mie':         ['U', (0.0,10.0), r'$f_\mathrm{sed,Mg2SiO4}$'], 
    # 'log_radius_cl_Mg2SiO4(s)_amorphous__Mie': ['U', (-7.0,1.0), r'$\log r_\mathrm{Mg2SiO4}$'], 
    
    # Chemistry
    'log_H2O': ['U', (-4.0,-2.0), r'$\log\ \mathrm{H_2O}$'],
    'log_CH4': ['U', (-7.0,-3.0), r'$\log\ \mathrm{CH_4}$'], 
    'log_12CO':  ['U', (-4.0,-2.0), r'$\log\ \mathrm{CO}$'], 
    'log_CO2': ['U', (-14.0,-4.0), r'$\log\ \mathrm{CO_2}$'], 
    
    'log_NH3': ['U', (-8.0,-4.0), r'$\log\ \mathrm{NH_3}$'], 
    'log_HCN': ['U', (-14.0,-4.0), r'$\log\ \mathrm{HCN}$'], 
    'log_H2S': ['U', (-6.0,-3.0), r'$\log\ \mathrm{H_2S}$'],
    'log_HF':  ['U', (-12.0,-6.0), r'$\log\ \mathrm{HF}$'],
    
    # 'log_FeH':   ['U', (-14.0,-4.0), r'$\log\ \mathrm{FeH}$'], 
    # 'log_FeH_P': ['U', (-1.0,1.0), r'$\log\ \mathrm{FeH_P}$'],
    # 'FeH_alpha': ['U', (0.0,10.0), r'$\alpha_\mathrm{FeH}$'],

    'log_13CO_ratio':    ['U', (1.0,4.0), r'$\log\ \mathrm{^{12}/^{13}CO}$'], 
    # 'log_C18O_ratio':    ['U', (1.0,4.0), r'$\log\ \mathrm{C^{16}/^{18}O}$'], 
    # 'log_H2(18)O_ratio': ['U', (1.0,4.0), r'$\log\ \mathrm{H_2^{16}/^{18}O}$'], 
    # 'log_H2(17)O_ratio': ['U', (1.0,4.0), r'$\log\ \mathrm{H_2^{16}/^{17}O}$'], 

    # PT profile 
    'dlnT_dlnP_0': ['TG', (0.15,0.01,-2,+2), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['TG', (0.18,0.04,-2,+2), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['TG', (0.21,0.05,-2,+2), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['TG', (0.16,0.06,-2,+2), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['TG', (0.08,0.025,-2,+2), r'$\nabla_4$'], 
    'dlnT_dlnP_5': ['TG', (0.06,0.02,-2,+2), r'$\nabla_5$'], 
    'dlnT_dlnP_6': ['G', (0.0,0.02), r'$\nabla_6$'], 
    'dlnT_dlnP_7': ['G', (0.0,0.02), r'$\nabla_7$'], 
    'T_phot':      ['U', (750.,2500.), r'$T_\mathrm{phot}$'], 

    'coverage_fraction': ['U', (0.0,1.0), 'cov'], 
    'sigma_g': ['U', (1.02,3), r'$\sigma_g$'], 
    'log_K_zz': ['U', (5,13), r'$\log\ K_\mathrm{zz}$'], 
    'log_X_base_Fe(s)_amorphous__Mie':    ['U', (-10.0,0.0), r'$\log X_\mathrm{Fe}$'], 
    'f_sed_Fe(s)_amorphous__Mie':         ['U', (0.0,10.0), r'$f_\mathrm{sed,Fe}$'], 
    'log_X_base_Mg2SiO4(s)_amorphous__Mie':    ['U', (-10.0,0.0), r'$\log X_\mathrm{Mg2SiO4}$'], 
    'f_sed_Mg2SiO4(s)_amorphous__Mie':         ['U', (0.0,10.0), r'$f_\mathrm{sed,Mg2SiO4}$'], 

    # 'nirspec_g235h_12_A': {
    #     'sigma_g': ['U', (1.02,3), r'$\sigma_g$'], 
    #     'log_K_zz': ['U', (5,13), r'$\log\ K_\mathrm{zz}$'], 

    #     # 'log_P_base_Fe(s)_amorphous__Mie':    ['U', (-2.0,3.0), r'$\log P_\mathrm{Fe}$'], 
    #     'log_X_base_Fe(s)_amorphous__Mie':    ['U', (-10.0,0.0), r'$\log X_\mathrm{Fe}$'], 
    #     'f_sed_Fe(s)_amorphous__Mie':         ['U', (0.0,10.0), r'$f_\mathrm{sed,Fe}$'], 

    #     # 'log_P_base_Mg2SiO4(s)_amorphous__Mie':    ['U', (-2.0,3.0), r'$\log P_\mathrm{Mg2SiO4}$'], 
    #     'log_X_base_Mg2SiO4(s)_amorphous__Mie':    ['U', (-10.0,0.0), r'$\log X_\mathrm{Mg2SiO4}$'], 
    #     'f_sed_Mg2SiO4(s)_amorphous__Mie':         ['U', (0.0,10.0), r'$f_\mathrm{sed,Mg2SiO4}$'], 
    # }, 
    # 'nirspec_g235h_12_B': {
    #     'sigma_g': ['U', (1.02,3), r'$\sigma_g$'], 
    #     'log_K_zz': ['U', (5,13), r'$\log\ K_\mathrm{zz}$'], 

    #     # 'log_P_base_Fe(s)_amorphous__Mie':    ['U', (-2.0,3.0), r'$\log P_\mathrm{Fe}$'], 
    #     'log_X_base_Fe(s)_amorphous__Mie':    ['U', (-10.0,0.0), r'$\log X_\mathrm{Fe}$'], 
    #     'f_sed_Fe(s)_amorphous__Mie':         ['U', (0.0,10.0), r'$f_\mathrm{sed,Fe}$'], 
    # }, 
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
    # shared_between_m_set = False,
    PT_mode = 'free_gradient', 
    n_knots = 8, 
    interp_mode = 'linear', 

    log_P_range = (-5.,2.), 
    n_atm_layers = 70,
)

line_species_g235h_1 = [
    '1H2-16O__POKAZATEL', #'1H2-18O__HotWat78', '1H2-17O__HotWat78', 
    '12C-1H4__MM', 
    '12C-16O__HITEMP', '13C-16O__HITEMP', #'12C-18O__HITEMP', 
    '12C-16O2__HITEMP', 

    '14N-1H3__CoYuTe', 
    '1H-12C-14N__Harris', 
    '1H2-32S__AYT2', 
    '1H-19F__Coxon-Hajig',
    
    # '56Fe-1H__MoLLIST', 
]
line_species_g235h_2 = [
    '1H2-16O__POKAZATEL', #'1H2-18O__HotWat78', '1H2-17O__HotWat78', 
    '12C-1H4__MM', 
    '12C-16O__HITEMP', '13C-16O__HITEMP', #'12C-18O__HITEMP', 
    '12C-16O2__HITEMP', 

    '14N-1H3__CoYuTe', 
    '1H-12C-14N__Harris', 
    '1H2-32S__AYT2', 
    '1H-19F__Coxon-Hajig',
]
line_species = list(np.unique(line_species_g235h_1+line_species_g235h_2))
chem_kwargs = dict(
    shared_between_m_set = True,
    chem_mode = 'free', 
    line_species = line_species, 
)

cloud_kwargs = dict(
    shared_between_m_set = False,
    cloud_mode = 'EddySed', 
    nirspec_g235h_12_A = dict(
        cloud_species = ['Fe(s)_amorphous__Mie', 'Mg2SiO4(s)_amorphous__Mie'], 
    ), 
    # nirspec_g235h_12_B = dict(
    #     cloud_species = ['Fe(s)_amorphous__Mie'], 
    # ), 
)

rotation_kwargs = dict(
    shared_between_m_set = True,
    rotation_mode = 'convolve', 
)

pRT_Radtrans_kwargs = dict(
    nirspec_g235h_12_A = dict(line_species=line_species, cloud_species=cloud_kwargs['nirspec_g235h_12_A']['cloud_species']), 
    # nirspec_g235h_12_B = dict(line_species=line_species, cloud_species=cloud_kwargs['nirspec_g235h_12_B']['cloud_species']), 

    rayleigh_species           = ['H2','He'],
    gas_continuum_contributors = ['H2-H2','H2-He'],
    # cloud_species              = cloud_kwargs.get('cloud_species'), 

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
    sum_model_settings = True, 
)

cov_kwargs = dict(
    trunc_dist = 5, # For Matern, 4.16 would be equivalent to 3*sigma
    scale_amp  = True,  
    max_separation = 5 * 10**free_params.get('log_l', [None,[None,np.inf]])[1][1], 
    # kernel_mode = 'matern', separation_mode = 'velocity', 
    kernel_mode = None, separation_mode = 'velocity', 
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
    n_live_points         = 100, 
    n_iter_before_update  = 100, 
)