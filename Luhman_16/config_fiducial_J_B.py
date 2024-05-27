import numpy as np
import os

file_params = 'config_fiducial_J_B.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'K_H2He_J_B_ret_2'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    J1226_A = {
        #'w_set': 'J1226', 'wave_range': (1115, 1325), 
        'w_set': 'J1226', 'wave_range': (1240, 1267), 

        #'wave_to_mask': np.array([[1241,1246], [1251,1255]]), # Mask K I lines
        #'wave_to_mask': np.array([[1251,1255]]), # Mask K I lines

        'file_target': './data/Luhman_16B_J.dat', 
        'file_std': './data/Luhman_16_std_J.dat', 
        #'file_wave': './data/Luhman_16_std_J.dat', 
        'file_wave': './data/Luhman_16_std_J_molecfit_transm.dat', # Use molecfit wlen-solution

        # Telluric transmission
        'file_skycalc_transm': './data/skycalc_transm_J1226.dat', 
        'file_molecfit_transm': './data/Luhman_16B_J_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/Luhman_16_std_J_molecfit_transm.dat', 
        'file_std_molecfit_continuum': './data/Luhman_16_std_J_molecfit_continuum.dat', 

        'filter_2MASS': '2MASS/2MASS.J', 
        'pwv': 1.5, 

        'ra': 162.299451, 'dec': -53.31767, 'mjd': 59946.35286502, 
        'ra_std': 161.738984, 'dec_std': -56.75771, 'mjd_std': 59946.3601578, 

        'T_std': 15000, 'log_g_std': 2.3, 'rv_std': 31.00, 'vsini_std': 280, 
        
        # Slit-width, sets model resolution
        'slit': 'w_0.4', 

        # Down-sample opacities for faster radiative transfer
        'lbl_opacity_sampling': 3, 

        # Ignore pixels with lower telluric transmission
        'tell_threshold': 0.7, 
        'sigma_clip_width': 5, # Remove outliers
    
        # Define pRT's log-pressure grid
        'log_P_range': (-5,3), 
        'n_atm_layers': 50, 
        }, 
)

magnitudes = {
    '2MASS/2MASS.J': (11.22, 0.04), # Burgasser et al. (2013)
    '2MASS/2MASS.Ks': (9.73, 0.09), 
}

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {

    'res': [(40000,100000), r'$R$'], 

    # Covariance parameters
    'log_a': [(-0.7,0.3), r'$\log\ a$'], 
    'log_l': [(-3.0,-1.0), r'$\log\ l$'], 

    # General properties
    #'R_p': [(0.5,1.2), r'$R_\mathrm{p}$'], 
    'log_g': [(4,6.0), r'$\log\ g$'], 
    #'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities #km/s
    #'vsini': [(10,30), r'$v\ \sin\ i$'], 
    'rv': [(10,30), r'$v_\mathrm{rad}$'], 

    # Surface brightness
    #'lat_band': [(0,90), r'$\lambda_\mathrm{b}$'], 
    #'epsilon_band': [(-1,1), r'$\epsilon_\mathrm{b}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10,5), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    'log_P_base_gray': [(-5,3), r'$\log\ P_{\mathrm{cl},0}$'], 
    'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 

    # Chemistry
    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'],
    #'log_H2(18)O': [(-12,-2), r'$\log\ \mathrm{H_2^{18}O}$'],
    #'log_CH4': [(-12,-2), r'$\log\ \mathrm{CH_4}$'],
    #'log_NH3': [(-12,-2), r'$\log\ \mathrm{NH_3}$'],
    #'log_H2S': [(-12,-2), r'$\log\ \mathrm{H_2S}$'],
    'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'],
    'log_FeH': [(-12,-2), r'$\log\ \mathrm{FeH}$'],
    'log_TiO': [(-12,-2), r'$\log\ \mathrm{TiO}$'], 
    'log_VO': [(-12,-2), r'$\log\ \mathrm{VO}$'],     

    'log_K': [(-12,-2), r'$\log\ \mathrm{K}$'], 
    #'log_KshiftH2': [(-12,-2), r'$\log\ \mathrm{K}$'], 
    #'log_KshiftHe': [(-12,-2), r'$\log\ \mathrm{K}$'], 

    #'log_Na': [(-12,-2), r'$\log\ \mathrm{Na}$'], 
    #'log_Ti': [(-12,-2), r'$\log\ \mathrm{Ti}$'], 
    'log_Fe': [(-12,-2), r'$\log\ \mathrm{Fe}$'],
    #'log_Ca': [(-12,-2), r'$\log\ \mathrm{Ca}$'], 

    # PT profile    
    'dlnT_dlnP_0': [(0.0,0.4), r'$\nabla_{T,0}$'], 
    'dlnT_dlnP_1': [(0.0,0.4), r'$\nabla_{T,1}$'], 
    'dlnT_dlnP_2': [(0.0,0.4), r'$\nabla_{T,2}$'], 
    'dlnT_dlnP_3': [(-0.1,0.4), r'$\nabla_{T,3}$'], 
    'dlnT_dlnP_4': [(-0.2,0.2), r'$\nabla_{T,4}$'], 

    'T_phot': [(200,3000), r'$T_\mathrm{phot}$'], 
    'log_P_phot': [(-3,1), r'$\log\ P_\mathrm{phot}$'], 
    'd_log_P_phot+1': [(0.5,2), r'$\log\ P_\mathrm{up}$'], 
    'd_log_P_phot-1': [(0.5,2), r'$\log\ P_\mathrm{low}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 496,  # +/- 37 mas
    'inclination': 26, # degrees

    'vsini': 25.3, 

    # PT profile
    'log_P_knots': np.array([-5, -2, 0.5, 1.5, 3], dtype=np.float64), 
}

####################################################################################
#
####################################################################################

sum_m_spec = False

scale_flux = True
scale_err  = True
apply_high_pass_filter = False

cloud_kwargs = {
    'cloud_mode': 'gray', 
}

rotation_mode = 'integrate' # 'convolve'

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = {
    'chem_mode': 'free', #'SONORAchem' #'eqchem'

    'line_species': [
        'H2O_pokazatel_main_iso', 
        #'H2O_181', 
        #'CH4_hargreaves_main_iso', 
        #'NH3_coles_main_iso', 
        #'H2S_Sid_main_iso', 
        'HF_main_iso', 
        'FeH_main_iso', 
        'TiO_48_Exomol_McKemmish', 
        'VO_ExoMol_McKemmish', 

        #'K', 
        'KshiftH2',
        'KshiftHe',
        #'Na_allard_recomputed', 
        #'Ti', 
        'Fe', 
        #'Ca', 
    ], 
}

species_to_plot_VMR = [
    'H2O', 'HF', 'FeH', 'TiO', 'VO', 'Fe', 
    #'K', 
    'KshiftH2', 
    'KshiftHe', 

    #'H2O', 'H2(18)O', 'CH4', 'NH3', 'H2S', 'HF', 'FeH', 'TiO', 'VO', 
    #'K', 'Na', 'Ti', 'Fe', 'Ca', 
    #'K', 'Na', 'Fe', 
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
    n_T_knots = len(constant_params['log_P_knots']), 
    PT_interp_mode = 'linear', 
    symmetric_around_P_phot = False, 
)

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 100
n_iter_before_update = 200
