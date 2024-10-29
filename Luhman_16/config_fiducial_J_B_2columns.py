import numpy as np
import os

file_params = 'config_fiducial_J_B_2columns.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'fiducial_J_B_ret_53_2columns_n1000'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    J1226_A = {
        'w_set': 'J1226', 'wave_range': (1115, 1338), 

        'file_target': './data/Luhman_16B_J.dat', 
        'file_std': './data/Luhman_16_std_J.dat', 
        #'file_wave': './data/Luhman_16_std_J.dat', 
        'file_wave': './data/Luhman_16_std_J_molecfit_transm.dat', # Use molecfit wlen-solution

        # Telluric transmission
        'file_molecfit_transm': './data/Luhman_16B_J_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/Luhman_16_std_J_molecfit_transm.dat', 
        'file_std_molecfit_continuum': './data/Luhman_16_std_J_molecfit_continuum.dat', 

        'filter_2MASS': '2MASS/2MASS.J', # Magnitude used for flux-calibration
        'pwv': 1.5, # Precipitable water vapour

        # Telescope-pointing, used for barycentric velocity-correction
        'ra': 162.299451, 'dec': -53.31767, 'mjd': 59946.35286502, 
        'ra_std': 161.738984, 'dec_std': -56.75771, 'mjd_std': 59946.3601578, 
        
        # Some info on standard-observation
        'T_std': 15000, 

        # Slit-width, sets model resolution
        'slit': 'w_0.4', 

        # Ignore pixels with lower telluric transmission
        'tell_threshold': 0.7, 
        'sigma_clip_width': 5, # Remove outliers
        }, 
)
config_data['J1226_B'] = config_data['J1226_A'].copy()

magnitudes = {
    '2MASS/2MASS.J': (11.22, 0.04), # Burgasser et al. (2013)
    '2MASS/2MASS.Ks': (9.73, 0.09), 
}

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {

    # Covariance parameters
    'log_a': [(-0.7,0.3), r'$\log\ a$'], 
    'log_l': [(-3.0,-1.0), r'$\log\ l$'], 

    # General properties
    #'R_p': [(0.5,1.2), r'$R_\mathrm{p}$'], 
    'log_g': [(3.5,6.0), r'$\log\ g$'], 
    'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities #km/s
    'vsini': [(10.,30.), r'$v\ \sin\ i$'], 
    'rv': [(10.,30.), r'$v_\mathrm{rad}$'], 

    'cloud_fraction': [(0.,1.), r'cf'], 
    'J1226_A': {
        # Cloud properties
        'log_opa_base_gray_0': [(-10,3), r'$\log\ \kappa_{\mathrm{cl,0,1}}$'], # Cloud slab
        'log_P_base_gray_0': [(0.0,2.5), r'$\log\ P_{\mathrm{cl,0,1}}$'], 
        'f_sed_gray_0': [(1,20), r'$f_\mathrm{sed,1}$'], 
        'log_FeH': [(-14,-2), r'$\log\ \mathrm{FeH}$'], 
        'log_FeH_P': [(-5,3), r'$\log\ P_\mathrm{FeH}$'], 
        'FeH_alpha': [(0,20), r'$\alpha_\mathrm{FeH}$'], 
    }, 
    'J1226_B': {
        # Cloud properties
        'log_opa_base_gray_0': [(-10,3), r'$\log\ \kappa_{\mathrm{cl,0,1}}$'], # Cloud slab
        'log_P_base_gray_0': [(0.0,2.5), r'$\log\ P_{\mathrm{cl,0,1}}$'], 
        'f_sed_gray_0': [(1,20), r'$f_\mathrm{sed,1}$'], 
        'log_FeH': [(-14,-2), r'$\log\ \mathrm{FeH}$'], 
        'log_FeH_P': [(-5,3), r'$\log\ P_\mathrm{FeH}$'], 
        'FeH_alpha': [(0,20), r'$\alpha_\mathrm{FeH}$'], 
    }, 

    # Cloud properties
    #'log_opa_base_gray_0': [(-10,3), r'$\log\ \kappa_{\mathrm{cl,0,1}}$'], # Cloud slab
    #'log_P_base_gray_0': [(0.0,2.5), r'$\log\ P_{\mathrm{cl,0,1}}$'], 
    #'f_sed_gray_0': [(1,20), r'$f_\mathrm{sed,1}$'], 

    #'log_FeH': [(-14,-2), r'$\log\ \mathrm{FeH}$'], 
    #'log_FeH_P': [(-5,3), r'$\log\ P_\mathrm{FeH}$'], 
    #'FeH_alpha': [(0,20), r'$\alpha_\mathrm{FeH}$'], 

    # Chemistry
    'log_H2O': [(-14,-2), r'$\log\ \mathrm{H_2O}$'], 
    'log_HF': [(-14,-2), r'$\log\ \mathrm{HF}$'], 
    'log_K': [(-14,-2), r'$\log\ \mathrm{K}$'], 
    'log_Na': [(-14,-2), r'$\log\ \mathrm{Na}$'], 

    # Impact shifts: d = A*T^b * n/n_ref
    # 0.0033 pm 0.0008 # 0.0036 pm 0.0006
    #'A_d_0': [(0.0,0.007), r'$A_{d,0}$'], # 0.00158988 (K-H2) | 0.001943820 (K-He)
    #'A_d_1': [(0.0,0.007), r'$A_{d,1}$'], # 0.00211668 (K-H2) | 0.000462539 (K-He)

    # 0.93 pm 0.03 # 0.90 pm 0.03
    #'b_d_0': [(0.5,1.5), r'$b_{d,0}$'],   # 0.949254 (K-H2) | 0.89691 (K-He)
    #'b_d_1': [(0.5,1.5), r'$b_{d,1}$'],   # 0.933563 (K-H2) | 1.07284 (K-He)

    # Impact widths: w = A*T^b * n/n_ref
    # 0.38 pm 0.04 # 0.17 pm 0.04
    #'A_w_0': [(0.05,0.60), r'$A_{w,0}$'], # 0.352609 (K-H2) | 0.208190 (K-He)
    #'A_w_1': [(0.00,0.45), r'$A_{w,1}$'], # 0.245926 (K-H2) | 0.121448 (K-He)

    # 0.39 pm 0.02 # 0.50 pm 0.03
    #'b_w_0': [(0.30,0.60), r'$b_{w,0}$'], # 0.385961 (K-H2) | 0.452833 (K-He)
    #'b_w_1': [(0.30,0.65), r'$b_{w,1}$'], # 0.447971 (K-H2) | 0.531718 (K-He)

    # PT profile    
    'dlnT_dlnP_0': [(0.10,0.34), r'$\nabla_0$'], 
    'dlnT_dlnP_1': [(0.10,0.34), r'$\nabla_1$'], 
    'dlnT_dlnP_2': [(0.05,0.34), r'$\nabla_2$'], 
    'dlnT_dlnP_3': [(0.,0.34), r'$\nabla_3$'], 
    'dlnT_dlnP_4': [(0.,0.34), r'$\nabla_4$'], 

    'T_phot': [(1200.,2200.), r'$T_\mathrm{phot}$'], 
    'log_P_phot': [(-1.,1.), r'$\log\ P_\mathrm{phot}$'], 
    'd_log_P_phot+1': [(0.5,3.), r'$\log\ P_\mathrm{up}$'], 
    'd_log_P_phot-1': [(0.5,2.), r'$\log\ P_\mathrm{low}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # Define pRT's log-pressure grid
    'log_P_range': (-5,3), 
    'n_atm_layers': 50, 

    # Down-sample opacities for faster radiative transfer
    'lbl_opacity_sampling': 3, 

    # Data resolution
    'res': 65000, 

    # General properties
    'parallax': 496,  # +/- 37 mas
    'inclination': 26, # degrees

    'do_scat_emis': False, 

    # Custom line opacity
    'A_w_0_H2': 0.352609, 'b_w_0_H2': 0.385961, 
    'A_w_1_H2': 0.245926, 'b_w_1_H2': 0.447971, 
    'A_d_0_H2': 0.00158988, 'b_d_0_H2': 0.949254, 
    'A_d_1_H2': 0.00211668, 'b_d_1_H2': 0.933563, 

    'A_w_0_He': 0.208190, 'b_w_0_He': 0.452833, 
    'A_w_1_He': 0.121448, 'b_w_1_He': 0.531718, 
    'A_d_0_He': 0.001943820, 'b_d_0_He': 0.89691, 
    'A_d_1_He': 0.000462539, 'b_d_1_He': 1.07284, 
}

#'''
parent_dir = '/net/lem/data1/regt/retrieval_base/retrieval_base/custom_opacity_data/'
#old_result_dir = '/net/lem/data1/regt/retrieval_base/Luhman_16/retrieval_outputs/fiducial_J_B_ret_45_1column/'
line_opacity_kwargs = [
    {
    # --- Potassium (K I) --------------------
    ###
    #'exists_in_pRT_atm': f'{old_result_dir}test_data/pRT_atm_J1226_A.pkl', 
    ###

    'NIST_states_file': f'{parent_dir}/K_I_states.txt', 
    #'VALD_trans_file': f'{parent_dir}/K_I_transitions.txt', 
    'VALD_trans_file': f'{parent_dir}/K_I_transitions_Kurucz.txt', 
    'pRT_name': 'K_wo_J_doublets', 
    'mass': 39.0983, 
    'is_alkali': True, 
    'E_ion': 35009.8140, #Z=0, # Potassium (K I)
    
    #'line_cutoff': 200, 
    'line_cutoff': 1000, 
    #n_density_ref=1e20, 
    'log_gf_threshold': -2.0, 
    'log_gf_threshold_exact': -0.5, 
    #'pre_compute': True, 
    'pre_compute': False, 

    #'nu_0': [7983.67489, 8041.38112], 
    #'log_gf': [-0.139, -0.439], 
    #'E_low': [13042.876, 12985.170], 
    #'gamma_N': [7.790, 7.790], 
    #'gamma_vdW': [-7.021, -7.022], 
    'nu_0': [7983.655, 8041.365], 
    'log_gf': [-0.063, -0.361], 
    'E_low': [13042.896, 12985.186], 
    'gamma_N': [7.83, 7.83], 
    'gamma_vdW': [-7.46, -7.46], 
    }, 
    {
    # --- Sodium (Na I) ----------------------
    ###
    #'exists_in_pRT_atm': f'{old_result_dir}test_data/pRT_atm_J1226_A.pkl', 
    ###

    'NIST_states_file': f'{parent_dir}/Na_I_states.txt', 
    #'VALD_trans_file': f'{parent_dir}/Na_I_transitions.txt', 
    'VALD_trans_file': f'{parent_dir}/Na_I_transitions_Kurucz.txt', 
    'pRT_name': 'Na_wo_J_doublet', 
    'mass': 22.989769, 
    'is_alkali': True, 
    'E_ion': 41449.451, #Z=0, # Sodium (Na I)
    
    #'line_cutoff': 200, 
    'line_cutoff': 1000, 
    #n_density_ref=1e20, 
    'log_gf_threshold': -2.0, 
    'log_gf_threshold_exact': -0.5, 
    #'pre_compute': True, 
    'pre_compute': False, 
    }, 
]
#'''
####################################################################################
#
####################################################################################

sum_m_spec = len(config_data) > 1

scale_flux = True
scale_err  = True
apply_high_pass_filter = False

cloud_kwargs = {
    'cloud_mode': 'gray', 
    #'J1226_A': {'cloud_mode': None}, 
    #'J1226_B': {'cloud_mode': 'gray'}, 
    #'cloud_mode': 'EddySed', 'cloud_species': ['Mg2SiO4(c)_cd', 'MgSiO3(c)_cd', 'Fe(c)_cd'], 
}

#rotation_mode = 'integrate' # 'convolve'
rotation_mode = 'convolve'

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = {
    'chem_mode': 'free', #'SONORAchem' #'eqchem'

    'line_species': [
        'H2O_pokazatel_main_iso_Sam_new', 
        'HF_main_iso_new', 
        'FeH_main_iso_Sam', 
        'K_wo_J_doublets', 
        'Na_wo_J_doublet', 
        #'K', 'Na', # On-the-fly treatment
        #'K_param_shift', 'Na_Sam'
    ], 
}

species_to_plot_VMR = [
    #'H2O', 'CH4', 'HF', 'FeH', 'Cr', 'Mn', 'AlH', 'SH', 'K', 'Na', 
    'H2O', 'HF', 'FeH', 'K', 'Na', 
    ]
species_to_plot_CCF = species_to_plot_VMR

####################################################################################
# Covariance parameters
####################################################################################

cov_kwargs = dict(
    cov_mode = 'GP', #None, 
    
    trunc_dist   = 3, 
    scale_GP_amp = True, 

    # Prepare the wavelength separation and
    # average squared error arrays and keep 
    # in memory
    prepare_for_covariance = True, #False, 
)

if free_params.get('log_l') is not None:
    cov_kwargs['max_separation'] = \
        cov_kwargs['trunc_dist'] * 10**np.max(free_params['log_l'][0])

####################################################################################
# PT parameters
####################################################################################

PT_kwargs = dict(
    PT_mode   = 'free_gradient', 
    n_T_knots = 5, 
    PT_interp_mode = 'linear', 
    symmetric_around_P_phot = False, 
)

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 1000
n_iter_before_update = 400
