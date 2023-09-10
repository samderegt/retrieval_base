import numpy as np

file_params = 'config_fiducial.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = '2M_0559_fiducial'
prefix = f'./retrieval_outputs/{prefix}/test_'

file_target = './data/2M_0559_K.dat'
file_std    = './data/2M_0559_std_K.dat'
file_wave   = './data/2M_0559_std_K.dat'
file_skycalc_transm = './data/skycalc_transm.dat'

magnitudes = {
    '2MASS/2MASS.J': (13.802, 0.024), # Cutri et al. (2003)
    '2MASS/2MASS.H': (13.679, 0.044), 
    '2MASS/2MASS.Ks': (13.577, 0.052), 
}
filter_2MASS = '2MASS/2MASS.Ks'

pwv = 10.0

ra, dec = 89.834193, -14.08239
mjd = 59977.26137825

ra_std, dec_std = 90.460453, -10.59824
mjd_std = 59977.20176338
T_std = 16258
log_g_std = 3.5
rv_std, vsini_std = 39.00, 45

slit = 'w_0.4'
lbl_opacity_sampling = 3

tell_threshold = 0.8

sigma_clip_width = 21

#wave_range = (1900, 2500)
wave_range = (1980, 2500)

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Uncertainty scaling
    #'a_1': [(0.1,0.8), r'$a_1$'], 
    #'a_2': [(0.1,0.8), r'$a_2$'], 
    #'a_3': [(0.1,0.8), r'$a_3$'], 
    #'a_4': [(0.1,0.8), r'$a_4$'], 
    #'a_5': [(0.1,0.8), r'$a_5$'], 
    #'a_6': [(0.1,0.8), r'$a_6$'], 
    #'a_7': [(0.1,0.8), r'$a_7$'],  
    #'l': [(10,40), r'$l$'], 

    # General properties
    'R_p': [(0.4,1.5), r'$R_\mathrm{p}$'], 
    'log_g': [(4.5,6), r'$\log\ g$'], 
    'epsilon_limb': [(0.2,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(10,30), r'$v\ \sin\ i$'], 
    'rv': [(-40,40), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    #'log_opa_base_gray': [(-10,3), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    #'log_P_base_gray': [(-6,3), r'$\log\ P_{\mathrm{cl},0}$'], 
    #'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 

    # Chemistry
    'log_12CO': [(-10,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_H2O': [(-10,-2), r'$\log\ \mathrm{H_{2}O}$'], 
    'log_CH4': [(-10,-2), r'$\log\ \mathrm{CH_{4}}$'], 
    #'log_NH3': [(-10,-2), r'$\log\ \mathrm{NH_{3}}$'], 
    #'log_13CO': [(-10,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    #'log_CO2': [(-10,-2), r'$\log\ \mathrm{CO_2}$'], 
    #'log_HCN': [(-10,-2), r'$\log\ \mathrm{HCN}$'], 

    # PT profile
    'log_gamma': [(-4,4), r'$\log\ \gamma$'], 

    'T_0': [(0,6000), r'$T_0$'], 
    'T_1': [(0,4500), r'$T_1$'], 
    'T_2': [(0,3000), r'$T_2$'], 
    'T_3': [(0,2000), r'$T_3$'], 
    'T_4': [(0,2000), r'$T_4$'], 
    #'T_5': [(0,2000), r'$T_5$'], 
    #'T_6': [(0,2000), r'$T_6$'], 

    'd_log_P_01': [(0,2), r'$\Delta\log\ P_{01}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 205.4251,  # +/- 0.1857 mas

    # PT profile
    #'log_P_knots': [-6, -1.25, -0.25, 0.5, 1, 1.5, 2], 
    'log_P_knots': [-6, -1.25, -0.25, 0.5, 1], 

    'd_log_P_12': 0.5, 
    'd_log_P_23': 0.5, 
    #'d_log_P_34': 0.75, 
    #'d_log_P_45': 1, 
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
    #'CO_36', 
    'CH4_hargreaves_main_iso', 
    #'NH3_coles_main_iso', 
    #'CO2_main_iso', 
    #'HCN_main_iso', 
    ]
cloud_species = None

scale_flux = True
scale_err  = True
#scale_GP_amp = True
scale_GP_amp = False
cholesky_mode = 'banded'

# Prepare the wavelength separation and
# average squared error arrays and keep 
# in memory
prepare_for_covariance = False
#prepare_for_covariance = True

apply_high_pass_filter = False

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 50
n_iter_before_update = 2
