import numpy as np

file_params = 'config_DENIS.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'DENIS_J0255_retrieval_outputs_synthetic_1'
prefix = f'./retrieval_outputs/{prefix}/test_'

file_target = './data/DENIS_J0255.dat'
file_std    = './data/DENIS_J0255_std.dat'
#file_wave   = './data/DENIS_J0255.dat'
file_wave   = './data/DENIS_J0255_std.dat'
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
#T_std = 14700
T_std = 12000
log_g_std = 3.5
rv_std, vsini_std = 25.5, 5

slit = 'w_0.2'
lbl_opacity_sampling = 3

tell_threshold = 0.6

wave_range = (1900, 2500)

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # General properties
    'R_p': [(0.4,1.2), r'$R_\mathrm{p}$'], 
    'log_g': [(4.5,6), r'$\log\ g$'], 
    'epsilon_limb': [(0.2,1), r'$\epsilon_\mathrm{limb}$'], 

    # Chemistry
    'C/O': [(0.1,1.0), r'C/O'], 
    'Fe/H': [(-1.5,1.5), r'Fe/H'], 
    'log_P_quench': [(-6,2), r'$\log\ P_\mathrm{quench}$'], 
    'log_C_ratio': [(-8,0), r'$\log\ \mathrm{^{13}C/^{12}C}$'], 

    # Velocities
    'vsini': [(35,50), r'$v\ \sin\ i$'], 
    'rv': [(20,25), r'$v_\mathrm{rad}$'], 
    
    # PT profile
    'T_eff': [(1000,2000), r'$T_\mathrm{eff}$'], 
    'log_gamma': [(-4,3), r'$\log\ \gamma$'], 

    'T_0': [(0,7000), r'$T_0$'], 
    'T_1': [(0,3000), r'$T_1$'], 
    'T_2': [(0,3000), r'$T_2$'], 
    'T_3': [(0,3000), r'$T_3$'], 
    'T_4': [(0,3000), r'$T_4$'], 

    'd_log_P_01': [(0.5,1.5), r'$\Delta\log\ P_{01}$'], 
    'd_log_P_12': [(0.5,1.5), r'$\Delta\log\ P_{12}$'], 
    'd_log_P_23': [(0.5,1.5), r'$\Delta\log\ P_{23}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 205.4251,  # +/- 0.1857 mas

    # PT profile
    'log_P_knots': [-6, -1.5, 0, 0.5, 1, 2], 
}

# Number of knots to define PT profile
ln_L_penalty_order = 3

line_species = [
    #'H2O_main_iso', 
    'H2O_pokazatel_main_iso', 
    'CO_main_iso', 
    #'CO_high', 
    'CO_36', 
    #'CO_36_high', 
    #'CO_28', 
    #'H2O_181', 
    'CH4_hargreaves_main_iso', 
    'NH3_main_iso', 
    #'CO2_main_iso', 
    #'HCN_main_iso', 
    ]
cloud_species = None

scale_flux = True
scale_err  = True
scale_GP_amp = True
cholesky_mode = 'banded'

# Prepare the wavelength separation and
# average squared error arrays and keep 
# in memory
#prepare_for_covariance = False
prepare_for_covariance = True

apply_high_pass_filter = False

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 50
n_iter_before_update = 5
