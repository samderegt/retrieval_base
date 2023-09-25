import numpy as np

file_params = 'config_fiducial_J_A.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'fiducial_J_A_ret_1'
prefix = f'./retrieval_outputs/{prefix}/test_'

wave_range = (1115, 1325)
#wave_range = (1115, 1338.3)
#wave_range = (1213, 1338.3)
wlen_setting = 'J1226'

file_target = './data/Luhman_16A_J.dat'
file_std    = './data/Luhman_16_std_J.dat'
file_wave   = './data/Luhman_16_std_J.dat'
file_skycalc_transm = f'./data/skycalc_transm_{wlen_setting}.dat'

magnitudes = {
    '2MASS/2MASS.J': (11.53, 0.04), # Burgasser et al. (2013)
    '2MASS/2MASS.Ks': (9.44, 0.07), 
}
filter_2MASS = '2MASS/2MASS.J'

pwv = 1.5

ra, dec = 162.299451, -53.31767
mjd = 59946.35286502

ra_std, dec_std = 161.738984, -56.75771
mjd_std = 59946.3601578
T_std = 15000
log_g_std = 2.3
rv_std, vsini_std = 31.00, 280

slit = 'w_0.4'
lbl_opacity_sampling = 3

tell_threshold = 0.6

sigma_clip_width = 8

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Uncertainty scaling
    'log_a': [(-18,-14), r'$\log\ a_1$'], 
    'log_l': [(-3,-0.8), r'$\log\ l$'], 

    # General properties
    'R_p': [(0.5,1.5), r'$R_\mathrm{p}$'], 
    'log_g': [(4,5.5), r'$\log\ g$'], 
    'epsilon_limb': [(0.2,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(10,50), r'$v\ \sin\ i$'], 
    'rv': [(10,25), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10,3), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    'log_P_base_gray': [(-6,3), r'$\log\ P_{\mathrm{cl},0}$'], 
    'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 

    # Chemistry
    'log_12CO': [(-10,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    #'log_13CO': [(-10,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    #'log_C18O': [(-10,-2), r'$\log\ \mathrm{C^{18}O}$'], 
    #'log_C17O': [(-10,-2), r'$\log\ \mathrm{C^{17}O}$'], 

    'log_H2O': [(-10,-2), r'$\log\ \mathrm{H_{2}O}$'], 
    'log_HDO': [(-10,-2), r'$\log\ \mathrm{HDO}$'], 

    'log_CH4': [(-10,-2), r'$\log\ \mathrm{CH_{4}}$'], 

    'log_NH3': [(-10,-2), r'$\log\ \mathrm{NH_{3}}$'], 
    #'log_CO2': [(-10,-2), r'$\log\ \mathrm{CO_2}$'], 
    'log_HCN': [(-10,-2), r'$\log\ \mathrm{HCN}$'], 
    'log_FeH': [(-10,-2), r'$\log\ \mathrm{FeH}$'], 
    #'log_H2S': [(-10,-2), r'$\log\ \mathrm{H_{2}S}$'], 
    'log_TiO': [(-10,-2), r'$\log\ \mathrm{^{48}TiO}$'], 
    
    'log_K': [(-10,-2), r'$\log\ \mathrm{K}$'], 
    'log_Na': [(-10,-2), r'$\log\ \mathrm{Na}$'], 
    #'log_Ti': [(-10,-2), r'$\log\ \mathrm{Ti}$'],
    'log_Fe': [(-10,-2), r'$\log\ \mathrm{Fe}$'], 

    # PT profile
    'log_gamma': [(-4,4), r'$\log\ \gamma$'], 

    'T_0': [(0,6000), r'$T_0$'], 
    #'dlnT_dlnP_0': [(-0.3,0.6), r'$\frac{\mathrm{d}\ln T}{\mathrm{d}\ln P}_{0}$'], 
    #'dlnT_dlnP_1': [(-0.3,0.6), r'$\frac{\mathrm{d}\ln T}{\mathrm{d}\ln P}_{1}$'], 
    #'dlnT_dlnP_2': [(-0.3,0.6), r'$\frac{\mathrm{d}\ln T}{\mathrm{d}\ln P}_{2}$'], 
    #'dlnT_dlnP_3': [(-0.3,0.6), r'$\frac{\mathrm{d}\ln T}{\mathrm{d}\ln P}_{3}$'], 
    #'dlnT_dlnP_4': [(-0.3,0.6), r'$\frac{\mathrm{d}\ln T}{\mathrm{d}\ln P}_{4}$'], 
    #'dlnT_dlnP_5': [(-0.3,0.6), r'$\frac{\mathrm{d}\ln T}{\mathrm{d}\ln P}_{5}$'], 

    'T_1': [(0,5000), r'$T_1$'], 
    'T_2': [(0,4000), r'$T_2$'], 
    'T_3': [(0,3000), r'$T_3$'], 
    'T_4': [(0,3000), r'$T_4$'], 
    'T_5': [(0,2000), r'$T_5$'], 
    'T_6': [(0,2000), r'$T_6$'], 
    'T_7': [(0,2000), r'$T_7$'], 

    'd_log_P_01': [(0,2), r'$\Delta\log\ P_{01}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 496,  # +/- 37 mas

    # PT profile
    #'log_P_knots': [-6, -1.25, -0.25, 0.5, 1, 1.5, 2], 
    'log_P_knots': [-6, -2.25, -1.25, -0.5, 0.0, 0.5, 1, 2], 

    #'d_log_P_01': 1.0, 
    'd_log_P_12': 0.5, 
    'd_log_P_23': 0.5, 
    'd_log_P_34': 0.5, 
    'd_log_P_45': 0.75, 
    'd_log_P_56': 1, 

    'epsilon_limb': 0.65, 
}

# Polynomial order of non-vertical abundance profile
chem_spline_order = 0

# Log-likelihood penalty
ln_L_penalty_order = 3
PT_interp_mode = 'log'
enforce_PT_corr = False


line_species = [
    'CO_main_iso', 
    #'CO_36', 
    #'CO_28', 
    #'CO_27', 

    'H2O_pokazatel_main_iso', 
    'HDO_voronin', 
    
    'CH4_hargreaves_main_iso', 

    'NH3_coles_main_iso', 
    #'CO2_main_iso', 
    'HCN_main_iso', 
    #'H2S_main_iso', 
    'FeH_main_iso', 
    'TiO_48_Exomol_McKemmish', 

    'K', 
    'Na_allard', 
    #'Ti', 
    'Fe', 
    ]
cloud_species = None
species_to_plot_VMR = ['12CO', 'H2O', 'CH4', 'NH3', 'FeH', 'TiO', 'Na', 'K']
species_to_plot_CCF = ['12CO', 'HCN', 'TiO', 'NH3', 'FeH']

scale_flux = True
scale_err  = True
scale_GP_amp = False
cholesky_mode = 'banded'
GP_trunc_dist = 3
GP_max_separation = GP_trunc_dist * 10**free_params['log_l'][0][1]

# Prepare the wavelength separation and
# average squared error arrays and keep 
# in memory
prepare_for_covariance = False

apply_high_pass_filter = False

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 200
n_iter_before_update = 400
