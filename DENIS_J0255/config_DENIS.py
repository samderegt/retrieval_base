import numpy as np

file_params = 'config_DENIS.py'

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
prefix = f'./retrieval_outputs/{prefix}/test_'

file_target = './data/DENIS_J0255.dat'
file_std    = './data/DENIS_J0255_std.dat'
file_wave   = './data/DENIS_J0255.dat'
file_skycalc_transm = './data/skycalc_transm.dat'

magnitudes = {
    '2MASS/2MASS.J': (13.25, 0.03), # Dupuy et al. (2012)
    '2MASS/2MASS.H': (12.20, 0.02), 
    '2MASS/2MASS.Ks': (11.56, 0.02), 
    #'MKO/MKO.K': (11.551, 0.024), 
    'WISE/WISE.W1': (10.77, 0.02), # Gagne et al. (2015)
    'WISE/WISE.W2': (10.21, 0.02), 
}
filter_2MASS = '2MASS/2MASS.Ks'

pwv = 1.5

ra, dec = 43.775978, -47.01850
mjd = 59886.14506179

ra_std, dec_std = 36.746755, -47.70470
mjd_std = 59886.13828925
T_std = 14700

slit = 'w_0.2'
lbl_opacity_sampling = 3

#wave_range = (2120, 2400)
wave_range = (2300, 2400)

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Uncertainty scaling
    #'log_a': [(-18,-14), r'$\log\ a$'], 
    #'log_a': [(-4,-1), r'$\log\ a$'], 
    #'log_l': [(-3,0), r'$\log\ l$'], 

    # General properties
    'R_p': [(0.6,1.5), r'$R_\mathrm{p}$'], 
    'log_g': [(4.5,6), r'$\log\ g$'], 
    'epsilon_limb': [(0.2,1), r'$\epsilon_\mathrm{limb}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10,3), r'$\log\ \kappa_\mathrm{base}^\mathrm{gray}$'], 
    'log_P_base_gray': [(-2,2), r'$\log\ P_\mathrm{base}^\mathrm{gray}$'], 
    'f_sed_gray': [(0,20), r'$f_\mathrm{sed}^\mathrm{gray}$'], 
    
    # Chemistry
    'log_12CO': [(-12,0), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_H2O': [(-12,0), r'$\log\ \mathrm{H_{2}O}$'], 
    'log_CH4': [(-12,0), r'$\log\ \mathrm{CH_{4}}$'], 
    'log_C_ratio': [(-12,0), r'$\log\ \mathrm{^{13}C/^{12}C}$'], 

    # Velocities
    'vsini': [(35,50), r'$v\ \sin\ i$'], 
    'rv': [(21,28), r'$v_\mathrm{rad}$'], 
    
    # PT profile
    'log_gamma': [(-3,4), r'$\log\ \gamma$'], 

    'T_0': [(0,7000), r'$T_0$'], 
    'T_1': [(0,4000), r'$T_1$'], 
    'T_2': [(0,4000), r'$T_2$'], 
    'T_3': [(0,4000), r'$T_3$'], 
    'T_4': [(0,4000), r'$T_4$'], 
    'T_5': [(0,4000), r'$T_5$'], 
    'T_6': [(0,4000), r'$T_6$'], 
    'T_7': [(0,4000), r'$T_7$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 205.4251,  # +/- 0.1857 mas

    # PT profile
    'log_P_knots': [-6, -4, -2, -1, 0, 0.5, 1, 2], 
}

# Number of knots to define PT profile
ln_L_penalty_order = 3

line_species = ['H2O_main_iso', 
                'CO_main_iso', 
                'CO_36', 
                #'CO_28', 
                #'H2O_181', 
                'CH4_hargreaves_main_iso', 
                #'NH3_main_iso', 
                #'CO2_main_iso', 
                #'HCN_main_iso', 
                ]
cloud_species = None

scale_flux = True
scale_err  = True
scale_GP_amp = True

apply_high_pass_filter = False

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 200
n_iter_before_update = 200
