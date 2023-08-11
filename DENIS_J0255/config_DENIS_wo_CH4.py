import numpy as np

file_params = 'config_DENIS_wo_CH4.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'DENIS_J0255_wo_CH4'
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
T_std = 14700
log_g_std = 3.5
rv_std, vsini_std = 25.5, 10

slit = 'w_0.2'
lbl_opacity_sampling = 3

tell_threshold = 0.6

wave_range = (1900, 2500)

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Uncertainty scaling
    'a_1': [(0.1,0.8), r'$a_1$'], 
    'a_2': [(0.1,0.8), r'$a_2$'], 
    'a_3': [(0.1,0.8), r'$a_3$'], 
    'a_4': [(0.1,0.8), r'$a_4$'], 
    'a_5': [(0.1,0.8), r'$a_5$'], 
    'a_6': [(0.1,0.8), r'$a_6$'], 
    'a_7': [(0.1,0.8), r'$a_7$'],  
    'l': [(10,40), r'$l$'], 

    # General properties
    'R_p': [(0.4,1.5), r'$R_\mathrm{p}$'], 
    'log_g': [(4.5,6), r'$\log\ g$'], 
    'epsilon_limb': [(0.2,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(35,50), r'$v\ \sin\ i$'], 
    'rv': [(20,25), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10,3), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    'log_P_base_gray': [(-6,3), r'$\log\ P_{\mathrm{cl},0}$'], 
    'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 

    # Chemistry
    'log_12CO': [(-10,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_H2O': [(-10,-2), r'$\log\ \mathrm{H_{2}O}$'], 
    #'log_CH4': [(-10,-2), r'$\log\ \mathrm{CH_{4}}$'], 
    'log_NH3': [(-10,-2), r'$\log\ \mathrm{NH_{3}}$'], 
    'log_13CO': [(-10,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    'log_CO2': [(-10,-2), r'$\log\ \mathrm{CO_2}$'], 
    'log_HCN': [(-10,-2), r'$\log\ \mathrm{HCN}$'], 

    # PT profile
    'log_gamma': [(-4,4), r'$\log\ \gamma$'], 

    'T_0': [(0,6000), r'$T_0$'], 
    'T_1': [(0,4500), r'$T_1$'], 
    'T_2': [(0,3000), r'$T_2$'], 
    'T_3': [(0,2000), r'$T_3$'], 
    'T_4': [(0,2000), r'$T_4$'], 
    'T_5': [(0,2000), r'$T_5$'], 
    'T_6': [(0,2000), r'$T_6$'], 

    'd_log_P_01': [(0,2), r'$\Delta\log\ P_{01}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 205.4251,  # +/- 0.1857 mas

    # PT profile
    'log_P_knots': [-6, -1.25, -0.25, 0.5, 1, 1.5, 2], 

    'd_log_P_12': 0.5, 
    'd_log_P_23': 0.5, 
    'd_log_P_34': 0.75, 
    'd_log_P_45': 1, 
}

# Polynomial order of non-vertical abundance profile
chem_spline_order = 0

# Log-likelihood penalty
ln_L_penalty_order = 3
PT_interp_mode = 'log'
enforce_PT_corr = False


line_species = [
    'H2O_pokazatel_main_iso', 
    'CO_main_iso', 
    'CO_36', 
    #'CH4_hargreaves_main_iso', 
    'NH3_coles_main_iso', 
    'CO2_main_iso', 
    'HCN_main_iso', 
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
n_live_points = 400
n_iter_before_update = 400
