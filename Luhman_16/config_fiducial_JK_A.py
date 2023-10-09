import numpy as np

file_params = 'config_fiducial_JK_A.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'fiducial_JK_A_ret_1'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = {
    'J1226': {
        'w_set': 'J1226', 'wave_range': (1115, 1325), 
        #'w_set': 'J1226', 'wave_range': (1240, 1295), 

        'file_target': './data/Luhman_16A_J.dat', 
        'file_std': './data/Luhman_16_std_J.dat', 
        'file_wave': './data/Luhman_16_std_J.dat', 
        'file_skycalc_transm': f'./data/skycalc_transm_J1226.dat', 

        'filter_2MASS': '2MASS/2MASS.J', 
        'pwv': 1.5, 

        'ra': 162.299451, 'dec': -53.31767, 'mjd': 59946.35286502, 
        'ra_std': 161.738984, 'dec_std': -56.75771, 'mjd_std': 59946.3601578, 

        'T_std': 15000, 'log_g_std': 2.3, 'rv_std': 31.00, 'vsini_std': 280, 
        
        'slit': 'w_0.4', 'lbl_opacity_sampling': 3, 
        'tell_threshold': 0.6, 'sigma_clip_width': 8, 
        }, 

    'K2166': {
        'w_set': 'K2166', 'wave_range': (1900, 2500), 
        #'w_set': 'K2166', 'wave_range': (2300, 2400), 

        'file_target': './data/Luhman_16A_K.dat', 
        'file_std': './data/Luhman_16_std_K.dat', 
        'file_wave': './data/Luhman_16_std_K.dat', 
        'file_skycalc_transm': f'./data/skycalc_transm_K2166.dat', 

        'filter_2MASS': '2MASS/2MASS.Ks', 
        'pwv': 1.5, 

        'ra': 162.297895, 'dec': -53.31703, 'mjd': 59946.32563173, 
        'ra_std': 161.739683, 'dec_std': -56.75788, 'mjd_std': 59946.31615474, 

        'T_std': 15000, 'log_g_std': 2.3, 'rv_std': 31.00, 'vsini_std': 280, 
        
        'slit': 'w_0.4', 'lbl_opacity_sampling': 3, 
        'tell_threshold': 0.8, 'sigma_clip_width': 8, 
        }
    }

magnitudes = {
    '2MASS/2MASS.J': (11.53, 0.04), # Burgasser et al. (2013)
    '2MASS/2MASS.Ks': (9.44, 0.07), 
}

####################################################################################
# Model parameters
####################################################################################

PT_mode = 'free'
chem_mode  = 'free'
cloud_mode = 'gray'
cov_mode = 'GP'

# Define the priors of the parameters
free_params = {
    # Data resolution
    #'res': [(20000,200000), r'res'], 
    'log_res_K2166': [(4,5.4), r'$\log\ R_\mathrm{K}$'], 

    # Uncertainty scaling
    #'log_a': [(-18,-14), r'$\log\ a_1$'], 
    'log_a_J1226': [(-1,0.4), r'$\log\ a_\mathrm{J}$'], 
    'log_l_J1226': [(-2,-0.8), r'$\log\ l_\mathrm{J}$'], 
    'log_a_K2166': [(-1,0.4), r'$\log\ a_\mathrm{K}$'], 
    'log_l_K2166': [(-2,-0.8), r'$\log\ l_\mathrm{K}$'], 

    # General properties
    'R_p': [(0.5,1.5), r'$R_\mathrm{p}$'], 
    'log_g': [(4,6.0), r'$\log\ g$'], 
    'epsilon_limb': [(0.2,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(10,35), r'$v\ \sin\ i$'], 
    'rv': [(15,22), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10,3), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    'log_P_base_gray': [(-6,3), r'$\log\ P_{\mathrm{cl},0}$'], 
    'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 
    'cloud_slope': [(-10,10), r'$\xi_\mathrm{cl}$'], 

    # Chemistry
    'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'], 
    'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'], 
    'log_C18O': [(-12,-2), r'$\log\ \mathrm{C^{18}O}$'], 
    'log_C17O': [(-12,-2), r'$\log\ \mathrm{C^{17}O}$'], 

    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_{2}O}$'], 
    #'log_HDO': [(-12,-2), r'$\log\ \mathrm{HDO}$'], 

    'log_CH4': [(-12,-2), r'$\log\ \mathrm{CH_{4}}$'], 
    'log_NH3': [(-12,-2), r'$\log\ \mathrm{NH_{3}}$'], 
    'log_HCN': [(-12,-2), r'$\log\ \mathrm{HCN}$'], 
    'log_H2S': [(-12,-2), r'$\log\ \mathrm{H_{2}S}$'], 
    'log_FeH': [(-12,-2), r'$\log\ \mathrm{FeH}$'], 
    #'log_CrH': [(-12,-2), r'$\log\ \mathrm{CrH}$'], 
    #'log_NaH': [(-12,-2), r'$\log\ \mathrm{NaH}$'], 

    #'log_TiO': [(-12,-2), r'$\log\ \mathrm{^{48}TiO}$'], 
    #'log_VO': [(-12,-2), r'$\log\ \mathrm{VO}$'], 
    #'log_AlO': [(-12,-2), r'$\log\ \mathrm{AlO}$'], 
    #'log_CO2': [(-12,-2), r'$\log\ \mathrm{CO_2}$'], 
    'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'], 
    #'log_HCl': [(-12,-2), r'$\log\ \mathrm{HCl}$'], 
    
    'log_K': [(-12,-2), r'$\log\ \mathrm{K}$'], 
    'log_Na': [(-12,-2), r'$\log\ \mathrm{Na}$'], 
    #'log_Ti': [(-12,-2), r'$\log\ \mathrm{Ti}$'], 
    'log_Fe': [(-12,-2), r'$\log\ \mathrm{Fe}$'], 

    # PT profile
    'log_gamma': [(-4,4), r'$\log\ \gamma$'], 

    'T_0': [(1000,5000), r'$T_0$'], 
    'T_1': [(500,3500), r'$T_1$'], 
    'T_2': [(0,2000), r'$T_2$'], 
    'T_3': [(0,2000), r'$T_3$'], 
    'T_4': [(0,2000), r'$T_4$'], 
    'T_5': [(0,2000), r'$T_5$'], 
    #'T_6': [(0,2000), r'$T_6$'], 
    #'T_7': [(0,2000), r'$T_7$'], 

    'd_log_P_01': [(0,2), r'$\Delta\log\ P_{01}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 496,  # +/- 37 mas

    # PT profile
    #'log_P_knots': [-6, -1.25, -0.25, 0.5, 1, 1.5, 2], 
    #'log_P_knots': [-6, -2.25, -1.25, -0.5, 0.0, 0.5, 1, 2], 
    'log_P_knots': [-6, -1.25, -0.5, 0.0, 0.5, 1, 2], 

    #'d_log_P_01': 1.0, 
    'd_log_P_12': 0.5, 
    'd_log_P_23': 0.75, 
    'd_log_P_34': 1.0, 

    #'d_log_P_12': 0.75, 
    #'d_log_P_23': 1.0, 
    
    #'d_log_P_12': 0.5, 
    #'d_log_P_23': 0.5, 
    #'d_log_P_34': 0.5, 
    #'d_log_P_45': 0.75, 
    #'d_log_P_56': 1, 

    'epsilon_limb': 0.65, 
}

# Polynomial order of non-vertical abundance profile
chem_spline_order = 0

# Log-likelihood penalty
ln_L_penalty_order = 3
PT_interp_mode = 'log'
enforce_PT_corr = False
n_T_knots = 6


line_species = [
    'CO_main_iso', 
    'CO_36', 
    'CO_28', 
    'CO_27', 

    'H2O_pokazatel_main_iso', 
    'CH4_hargreaves_main_iso', 
    'NH3_coles_main_iso', 

    'HCN_main_iso', 
    #'H2S_main_iso', 
    'H2S_ExoMol_main_iso', 
    'FeH_main_iso', 
    #'CrH_main_iso', 
    #'NaH_main_iso', 

    #'TiO_48_Exomol_McKemmish', 
    #'VO_ExoMol_McKemmish', 
    #'CO2_main_iso', 
    'HF_main_iso', 
    #'HCl_main_iso', 

    'K', 
    'Na_allard', 
    #'Ti', 
    'Fe', 
    ]
cloud_species = None
species_to_plot_VMR = [
    '12CO', 'H2O', 'CH4', 'NH3', 'H2S', 'FeH', 'CrH', 'NaH', 'TiO', 'VO', 'HCl', 'K', 'Na', 'Ti', 'Fe', '13CO', 'C18O', 'C17O', 'HCN', 'HF', 
    ]
species_to_plot_CCF = [
    '12CO', 'H2O', 'CH4', 'NH3', 'H2S', 'FeH', 'CrH', 'NaH', 'TiO', 'VO', 'HCl', 'K', 'Na', 'Ti', 'Fe', '13CO', 'C18O', 'C17O', 'HCN', 'HF', 
    ]

scale_flux = True
scale_err  = True
#scale_GP_amp = False
scale_GP_amp = True
cholesky_mode = 'banded'
GP_trunc_dist = 3

GP_max_separation = 20
if free_params.get('log_l') is not None:
    GP_max_separation = GP_trunc_dist * 10**free_params['log_l'][0][1]
if free_params.get('l') is not None:
    GP_max_separation = GP_trunc_dist * free_params['l'][0][1]

if (free_params.get('log_l_K2166') is not None) and \
    (free_params.get('log_l_J1226') is not None):
    GP_max_separation = \
        GP_trunc_dist * max([10**free_params['log_l_K2166'][0][1], 
                             10**free_params['log_l_J1226'][0][1]])

# Prepare the wavelength separation and
# average squared error arrays and keep 
# in memory
prepare_for_covariance = True

apply_high_pass_filter = False

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
#n_live_points = 200
#n_iter_before_update = 400
n_live_points = 200
n_iter_before_update = 400
