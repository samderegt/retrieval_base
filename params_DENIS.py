import numpy as np

file_params = 'params_DENIS.py'

plot_counter             = 0
return_PT                = False
return_spec              = False
return_contr             = False
return_mass_fractions    = False
fit_photometry           = False
show_PT_params_in_corner = True

run_multinest = True
run_optimize  = False

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'DENIS_J0255_retrieval_outputs_142'
prefix = f'./{prefix}/test_'

file_target = '/home/sam/Documents/PhD/CRIRES_data_reduction/DENIS_J0255/data/DENIS_J0255.dat'
file_std    = '/home/sam/Documents/PhD/CRIRES_data_reduction/DENIS_J0255/data/DENIS_J0255_std.dat'
file_wave   = '/home/sam/Documents/PhD/CRIRES_data_reduction/DENIS_J0255/data/DENIS_J0255.dat'

magnitudes = {
    '2MASS/2MASS.J': (13.25, 0.03), # Dupuy et al. (2012)
    '2MASS/2MASS.H': (12.20, 0.02), 
    '2MASS/2MASS.Ks': (11.56, 0.02), 
    #'MKO/MKO.K': (11.551, 0.024), 
    'WISE/WISE.W1': (10.77, 0.02), # Gagne et al. (2015)
    'WISE/WISE.W2': (10.21, 0.02), 
}
filter_2MASS = '2MASS/2MASS.Ks'

parallax = 205.4251 # +/- 0.1857 mas

pwv = 1.5

ra, dec = 43.775978, -47.01850
mjd = 59886.14506179

ra_std, dec_std = 36.746755, -47.70470
mjd_std = 59886.13828925
T_std = 14700

slit = 'w_0.2'
lbl_opacity_sampling = 3

wave_range = (1900, 2500)

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
param_priors = {
    # Uncertainty scaling
    #'log_a': (-4,-1), 
    #'log_a_5': (-18,-14), 
    'log_a_6': (-18,-14), 
    #'log_a_7': (-18,-14), 
    #'log_tau': (-2,0),  
    #'log_tau_5': (-3,0),  
    'log_tau_6': (-3,0),  
    #'log_tau_7': (-3,0),  
    #'beta': (0.1,1.5), 
    #'beta_1': (0.1,1.5), 
    #'beta_2': (0.1,1.5), 
    #'beta_3': (0.1,1.5), 
    #'beta_4': (0.1,1.5), 
    #'beta_5': (0.1,1.5), 
    'beta_6': (0.1,1.5), 
    #'beta_7': (0.1,1.5), 

    'x_tol': (0,0.5), 

    # Spectral resolution
    #'R': (1000,50000),

    # General properties
    'R_p': (0.6,1.5), 
    'log_g': (4.5,6), 
    'epsilon_limb': (0.2,1),

    # Cloud properties
    #'log_X_cloud_base_MgSiO3': (-12,0),
    #'log_P_base_MgSiO3': (-2,2), 
    #'log_X_MgSiO3': (-2.3,1),
    #'f_sed': (0,10),
    #'log_K_zz': (5,13), 
    #'sigma_g': (1.05,3), 

    'log_opa_base_gray': (-10,3), 
    'log_P_base_gray': (-2,2), 
    'f_sed_gray': (0,20), 
    
    # Chemistry
    #'C/O': (0.1,1.2),
    #'Fe/H': (-1.5,1.5),
    #'log_P_quench': (-6,2),

    'log_12CO': (-12,0), 
    'log_H2O': (-12,0), 
    'log_CH4': (-12,0), 
    'log_NH3': (-12,0), 
    #'log_HCN': (-12,0), 
    #'log_CO2': (-12,0), 
    'log_C_ratio': (-12,0),
    #'log_O_ratio': (-12,0),

    # Velocities
    'vsini': (35,50),
    'rv': (18,22),
    
    # PT profile
    'log_gamma': (-3,4), 

    'T_bottom': (0,7000),

    #'log_P_phot': (-2,2),
    #'alpha': (0.5,4),
    #'T_int': (700,2000),
}

# Number of knots to define PT profile
n_T_knots = 7

#T_knots_prior = [(0, 1)] * n_T_knots
T_knots_prior = [(0, 4000)] * n_T_knots
ln_L_penalty_order = 3

# Mathtext labels used in figure
labels_to_plot = [
    #r'$\log\ a$', 
    #r'$\log\ a_5$', 
    r'$\log\ a_6$', 
    #r'$\log\ a_7$', 
    #r'$\log\ \tau$', 
    #r'$\log\ \tau_5$', 
    r'$\log\ \tau_6$', 
    #r'$\log\ \tau_7$', 
    #r'$\beta$', 
    #r'$\beta_1$', 
    #r'$\beta_2$', 
    #r'$\beta_3$', 
    #r'$\beta_4$', 
    #r'$\beta_5$', 
    r'$\beta_6$', 
    #r'$\beta_7$',
    r'$x_{tol}$',  
    #r'$\mathrm{R}$', 
    r'$R_\mathrm{p}$', 
    r'$\log\ g$', 
    r'$\epsilon_\mathrm{limb}$', 
    #r'$\log\ X_\mathrm{base}^\mathrm{MgSiO_3}$', 
    #r'$\log\ P_\mathrm{base}^\mathrm{MgSiO_3}$', 
    #r'$\log\ \tilde{X}_0^\mathrm{MgSiO_3}$', 
    #r'$f_\mathrm{sed}$', 
    #r'$\log K_\mathrm{zz}$',
    #r'$\sigma_\mathrm{g}$',
    r'$\log\ \kappa_\mathrm{base}^\mathrm{gray}$', 
    r'$\log\ P_\mathrm{base}^\mathrm{gray}$', 
    r'$f_\mathrm{sed}^\mathrm{gray}$', 
    #'C/O', 
    #'Fe/H', 
    #r'$\log\ P_\mathrm{quench}$', 
    r'$\log\ \mathrm{^{12}CO}$', 
    r'$\log\ \mathrm{H_2O}$', 
    r'$\log\ \mathrm{CH_4}$', 
    r'$\log\ \mathrm{NH_3}$', 
    #r'$\log\ \mathrm{CO_2}$', 
    #r'$\log\ \mathrm{HCN}$', 
    r'$\log\ \mathrm{^{13}C/^{12}C}$', 
    #r'$\log\ \mathrm{^{18}O/^{16}O}$', 
    r'$v\ \sin\ i$', 
    r'$v_\mathrm{rad}$', 
    r'$\log\ \gamma$', 
    r'$T_\mathrm{bottom}$', 
    #r'$\log\ P_\mathrm{phot}$', 
    #r'$\alpha$', 
    #r'$T_\mathrm{int}$', 
    ]

line_species = ['H2O_main_iso', 
                'CO_main_iso', 
                'CO_36', 
                #'CO_28', 
                #'H2O_181', 
                'CH4_hargreaves_main_iso', 
                'NH3_main_iso', 
                #'CO2_main_iso', 
                #'HCN_main_iso', 
                ]
"""
line_species_low_res  = ['H2O_HITEMP_R_120',
                         'CH4_R_120', 
                         'NH3_R_120', 
                         'CO_all_iso_HITEMP_R_120', 
                         'CO2_R_120', 
                         'HCN_R_120', 
                         'H2S_R_120', 
                         'PH3_R_120', 
                         ], 
"""
cloud_species = None

# Constants to use if prior is not given
param_constant = {
    # Uncertainty scaling
    'log_a': -np.inf*np.ones(21), 
    'log_tau': np.zeros(21), 
    'beta': np.ones(21),  
    'beta_telluric': None, 

    # Spectral resolution
    'R': 100000, 

    # General properties
    'R_p': 1.0, 
    'log_g': 5.4, 
    'epsilon_limb': 0.6, 

    # Cloud properties
    'log_X_cloud_base_MgSiO3': None, 
    'log_P_base_MgSiO3': None, 

    'log_X_MgSiO3': None, 
    'f_sed': None, 
    'log_K_zz': None, 
    'sigma_g': None, 

    'log_opa_base_gray': None, 
    'log_P_base_gray': None, 
    'f_sed_gray': None, 
    
    # Chemistry
    'log_P_quench': -8,
    'log_C_ratio': -30,
    'log_O_ratio': -30,

    # Velocities
    'vsini': 40,
    'rv': 20,

    # PT profile
    'log_gamma': None,
    'P_knots': np.array([1e-6, 1e-4, 1e-2, 1e-1, 1e0, 10**(0.5), 1e1, 1e2]), 
    #'P_knots': None, 
}

scale_flux_per_detector = True
apply_high_pass_filter = False
f_dets_global = None

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 200
n_iter_before_update = 200
