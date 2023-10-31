import numpy as np
import os

file_params = 'config_fiducial_J_B.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'fiducial_J_B_ret_18'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = {
    'J1226': {
        'w_set': 'J1226', 'wave_range': (1115, 1325), 
        #'w_set': 'J1226', 'wave_range': (1115, 1139), 
        #'w_set': 'J1226', 'wave_range': (1240, 1295), 

        'wave_to_mask': np.array([[1241,1246], [1251,1255]]), # Mask K I lines

        'file_target': './data/Luhman_16B_J.dat', 
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

        'log_P_range': (-5,2), 'n_atm_layers': 50, 
        }, 
    }

magnitudes = {
    '2MASS/2MASS.J': (11.22, 0.04), # Burgasser et al. (2013)
    '2MASS/2MASS.Ks': (9.73, 0.09), 
}

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Data resolution
    #'res': [(20000,200000), r'res'], 
    #'log_res_J1226': [(4,5.2), r'$\log\ R_\mathrm{J}$'], 

    # Uncertainty scaling
    #'log_a': [(-18,-14), r'$\log\ a_1$'], 
    'log_a': [(-1,0.4), r'$\log\ a_\mathrm{J}$'], 
    'log_l': [(-2,-0.8), r'$\log\ l_\mathrm{J}$'], 
    #'log_a_K2166': [(-1,0.4), r'$\log\ a_\mathrm{K}$'], 
    #'log_l_K2166': [(-2,-0.8), r'$\log\ l_\mathrm{K}$'], 

    # General properties
    'R_p': [(0.5,1.2), r'$R_\mathrm{p}$'], 
    'log_g': [(4,6.0), r'$\log\ g$'], 
    #'epsilon_limb': [(0.1,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(20,30), r'$v\ \sin\ i$'], 
    'rv': [(16,22), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10,5), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    'log_P_base_gray': [(-5,2), r'$\log\ P_{\mathrm{cl},0}$'], 
    'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 
    #'cloud_slope': [(-10,10), r'$\xi_\mathrm{cl}$'], 

    # Chemistry
    'C/O': [(0.15,1), r'C/O'], 
    'Fe/H': [(-1,1), r'[Fe/H]'], 
    'log_P_quench_CO_CH4': [(-4,2), r'$\log\ P_\mathrm{quench}(\mathrm{C})$'], 
    'log_P_quench_N2_NH3': [(-4,2), r'$\log\ P_\mathrm{quench}(\mathrm{N})$'], 
    #'log_C13_12_ratio': [(-10,0), r'$\log\ \mathrm{^{13}C/^{12}C}$'], 
    'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'], 

    # PT profile
    'dlnT_dlnP_0': [(0.12, 0.5), r'$\nabla_{T,0}$'], 
    'dlnT_dlnP_1': [(0.13,0.32), r'$\nabla_{T,1}$'], 
    'dlnT_dlnP_2': [(0.03,0.23), r'$\nabla_{T,2}$'], 
    'dlnT_dlnP_3': [(0.0,0.2), r'$\nabla_{T,3}$'], 
    'dlnT_dlnP_4': [(-0.03,0.13), r'$\nabla_{T,4}$'], 
    'T_0': [(2000,5000), r'$T_0$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 496,  # +/- 37 mas
    'epsilon_limb': 0.65, 

    # PT profile
    'log_P_knots': [-5, -2, -2/3, 2/3, 2], 
}

####################################################################################
#
####################################################################################

scale_flux = True
scale_err  = True
apply_high_pass_filter = False

cloud_mode = 'gray'
cloud_species = None

####################################################################################
# Chemistry parameters
####################################################################################

chem_mode  = 'SONORAchem'

#import pyfastchem
#fastchem_path = os.path.dirname(pyfastchem.__file__)
chem_kwargs = dict(
    #spline_order   = 0

    quench_setup = {
        'P_quench_CO_CH4': ['12CO', 'CH4', 'H2O', '13CO', 'C18O', 'C17O'], 
        'P_quench_N2_NH3': ['N2', 'HCN', 'NH3'], 
        }, 
    
    path_SONORA_chem = '../SONORA_models/chemistry', 

    #abundance_file = f'{fastchem_path}/input/element_abundances/asplund_2020.dat', 
    #gas_data_file  = f'{fastchem_path}/input/logK/logK.dat', 
    #cond_data_file = f'{fastchem_path}/input/logK/logK_condensates.dat', 
    #verbose_level  = 1, 
    #use_eq_cond      = True, 
    ##use_eq_cond      = False, 
    #use_rainout_cond = True,
    ##use_rainout_cond = False,
)

line_species = [
    'H2O_pokazatel_main_iso', 
    'CH4_hargreaves_main_iso', 
    'NH3_coles_main_iso', 

    'H2S_ExoMol_main_iso', 
    'FeH_main_iso', 

    'TiO_48_Exomol_McKemmish', 
    'VO_ExoMol_McKemmish', 
    'HF_main_iso', 

    'K', 
    'Na_allard', 
    'Fe', 

    #'HCl_main_iso', 
    #'HCN_main_iso', 
    #'CO2_main_iso', 
    ]
species_to_plot_VMR = [
    'H2O', 'CH4', 'NH3', 'H2S', 'FeH', 'TiO', 'VO', 'K', 'Na', 'Fe', 'HF', 
    ]
species_to_plot_CCF = [
    'H2O', 'CH4', 'NH3', 'H2S', 'FeH', 'TiO', 'VO', 'K', 'Na', 'Fe', 'HF', 
    ]

####################################################################################
# Covariance parameters
####################################################################################

cov_mode = 'GP'

cov_kwargs = dict(
    trunc_dist   = 3, 
    scale_GP_amp = True, 
    max_separation = 20, 

    # Prepare the wavelength separation and
    # average squared error arrays and keep 
    # in memory
    prepare_for_covariance = True
)

if free_params.get('log_l') is not None:
    cov_kwargs['max_separation'] = \
        cov_kwargs['trunc_dist'] * 10**free_params['log_l'][0][1]
if free_params.get('l') is not None:
    cov_kwargs['max_separation'] = \
        cov_kwargs['trunc_dist'] * free_params['l'][0][1]

if (free_params.get('log_l_K2166') is not None) and \
    (free_params.get('log_l_J1226') is not None):
    cov_kwargs['max_separation'] = cov_kwargs['trunc_dist'] * \
        10**max([free_params['log_l_K2166'][0][1], \
                 free_params['log_l_J1226'][0][1]])

####################################################################################
# PT parameters
####################################################################################

PT_mode = 'free_gradient'

PT_kwargs = dict(
    conv_adiabat = True, 

    ln_L_penalty_order = 3, 
    PT_interp_mode = 'quadratic', 

    enforce_PT_corr = False, 
    n_T_knots = 5, 
)

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 200
n_iter_before_update = 200
