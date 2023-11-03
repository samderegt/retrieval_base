import numpy as np

file_params = 'config_fiducial_J.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'fiducial_K_ret_1'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = {
    'K2166': {
        'w_set': 'K2166', 'wave_range': (1900, 2500), 

        'file_target': './data/2M_0559_K.dat', 
        'file_std': './data/2M_0559_std_K.dat', 
        'file_wave': './data/2M_0559_K.dat', 
        'file_skycalc_transm': f'./data/skycalc_transm_K2166.dat', 

        'filter_2MASS': '2MASS/2MASS.Ks', 
        'pwv': 10.0, 

        'ra': 89.834193, 'dec': -14.08239, 'mjd': 59977.26137825, 
        'ra_std': 90.460453, 'dec_std': -10.59824, 'mjd_std': 59977.20176338, 

        'T_std': 16258, 'log_g_std': 3.5, 'rv_std': -39.00, 'vsini_std':45, 
        
        'slit': 'w_0.4', 'lbl_opacity_sampling': 3, 
        'tell_threshold': 0.8, 'sigma_clip_width': 8, 
    
        'log_P_range': (-5,3), 'n_atm_layers': 50, 
        }, 
    }

magnitudes = {
    '2MASS/2MASS.J': (13.802, 0.024), # Cutri et al. (2003)
    '2MASS/2MASS.H': (13.679, 0.044), 
    '2MASS/2MASS.Ks': (13.577, 0.052), 
}

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Data resolution
    #'log_res_J1226': [(4,5.2), r'$\log\ R_\mathrm{J}$'], 

    # Uncertainty scaling 
    #'log_a_K2166': [(-1,0.4), r'$\log\ a_\mathrm{K}$'], 
    #'log_l_K2166': [(-2,-0.8), r'$\log\ l_\mathrm{K}$'], 
    #'log_a_J1226': [(-1,0.4), r'$\log\ a_\mathrm{J}$'], 
    #'log_l_J1226': [(-2,-0.8), r'$\log\ l_\mathrm{J}$'], 
    #'log_a': [(-1,0.3), r'$\log\ a_\mathrm{J}$'], 
    #'log_l': [(-2,0.4), r'$\log\ l_\mathrm{J}$'], 

    # General properties
    'R_p': [(0.5,1.5), r'$R_\mathrm{p}$'], 
    'log_g': [(4,6.0), r'$\log\ g$'], 
    #'epsilon_limb': [(0.1,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(10,30), r'$v\ \sin\ i$'], 
    'rv': [(-20,-10), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10,5), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    'log_P_base_gray': [(-5,3), r'$\log\ P_{\mathrm{cl},0}$'], 
    'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 
    #'cloud_slope': [(-10,0), r'$\xi_\mathrm{cl}$'], 

    # Chemistry
    'C/O': [(0.15,1), r'C/O'], 
    'Fe/H': [(-1,1), r'[Fe/H]'], 
    'log_P_quench_CO_CH4': [(-5,3), r'$\log\ P_\mathrm{quench}(\mathrm{C})$'], 
    'log_P_quench_N2_NH3': [(-5,3), r'$\log\ P_\mathrm{quench}(\mathrm{N})$'], 
    #'log_C13_12_ratio': [(-10,0), r'$\log\ \mathrm{^{13}C/^{12}C}$'], 
    #'log_O18_16_ratio': [(-10,0), r'$\log\ \mathrm{^{18}O/^{16}O}$'], 
    #'log_O17_16_ratio': [(-10,0), r'$\log\ \mathrm{^{17}C/^{16}O}$'], 
    'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'], 
    'log_HCl': [(-12,-2), r'$\log\ \mathrm{HCl}$'], 

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
    'parallax': 95.2714,  # +/- 0.7179 mas
    'epsilon_limb': 0.65, 

    # PT profile
    #'log_P_knots': [-5, -2, -2/3, 2/3, 2], 
    'log_P_knots': [-5, -2, -1/3, 1.5, 3], 
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
    'CO_main_iso', 
    #'CO_36', 
    #'CO_28', 
    #'CO_27', 

    'H2O_pokazatel_main_iso', 
    'CH4_hargreaves_main_iso', 
    'NH3_coles_main_iso', 
    
    'H2S_ExoMol_main_iso', 
    'HF_main_iso', 
    'HCl_main_iso', 

    'HCN_main_iso', 
    'CO2_main_iso', 

    #'FeH_main_iso', 

    #'TiO_48_Exomol_McKemmish', 
    #'VO_ExoMol_McKemmish', 

    #'K', 
    #'Na_allard', 
    #'Fe', 
    ]
species_to_plot_VMR = [
    '12CO', 'HCN', 'CO2', 
    'H2O', 'CH4', 'NH3', 'H2S', 
    'HF', 'HCl', 
    #'12CO', '13CO', 'C18O', 'C17O', 'HCN', 'CO2', 
    #'H2O', 'CH4', 'NH3', 'H2S', 'FeH', #'TiO', 'VO', 
    #'K', 'Na', 'Fe', 
    #'HF', 'HCl', 
    ]
species_to_plot_CCF = species_to_plot_VMR

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