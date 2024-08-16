import numpy as np
import os

file_params = 'config_fiducial_J_A.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'fiducial_J_A_ret_7'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    J1226_A = {
        'w_set': 'J1226', 'wave_range': (1115, 1338), 

        #'wave_to_mask': np.array([[1241,1246], [1251,1255]]), # Mask K I lines
        #'wave_to_mask': np.array([[1251,1255]]), # Mask K I lines

        'file_target': './data/Luhman_16A_J.dat', 
        'file_std': './data/Luhman_16_std_J.dat', 
        #'file_wave': './data/Luhman_16_std_J.dat', 
        'file_wave': './data/Luhman_16_std_J_molecfit_transm.dat', # Use molecfit wlen-solution

        # Telluric transmission
        'file_molecfit_transm': './data/Luhman_16A_J_molecfit_transm.dat', 
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


magnitudes = {
    '2MASS/2MASS.J': (11.53, 0.04), # Burgasser et al. (2013)
    '2MASS/2MASS.Ks': (9.44, 0.07), 
}

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {

    #'res': [(40000,100000), r'$R$'], 

    # Covariance parameters
    'log_a': [(-0.7,0.3), r'$\log\ a$'], 
    'log_l': [(-3.0,-1.0), r'$\log\ l$'], 

    # General properties
    #'R_p': [(0.5,1.2), r'$R_\mathrm{p}$'], 
    'log_g': [(4.0,6.0), r'$\log\ g$'], 
    'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities #km/s
    'vsini': [(10,30), r'$v\ \sin\ i$'], 
    'rv': [(10,30), r'$v_\mathrm{rad}$'], 

    # Surface brightness
    #'lat_band': [(0,90), r'$\lambda_\mathrm{b}$'], 
    #'epsilon_band': [(-1,1), r'$\epsilon_\mathrm{b}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10,10), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    'log_P_base_gray': [(-1,2), r'$\log\ P_{\mathrm{cl},0}$'], 
    'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 
    #'log_X_Mg2SiO4(c)': [(-2.3,1.), r'$\log\ X_\mathrm{Mg2SiO4}$'], 
    #'f_sed_Mg2SiO4(c)': [(0.,10.), r'$f_\mathrm{sed,Mg2SiO4}$'], 
    #'log_X_MgSiO3(c)': [(-2.3,1.), r'$\log\ X_\mathrm{MgSiO3}$'], 
    #'f_sed_MgSiO3(c)': [(0.,10.), r'$f_\mathrm{sed,MgSiO3}$'], 
    #'log_X_Fe(c)': [(-2.3,1.), r'$\log\ X_\mathrm{Fe}$'], 
    #'f_sed_Fe(c)': [(0.,10.), r'$f_\mathrm{sed,Fe}$'], 
#
    #'log_K_zz': [(5.,13.), r'$\log\ K_\mathrm{zz}$'], 
    #'sigma_g': [(1.05,3), r'$\sigma_\mathrm{g}$'], 

    # Chemistry
    'log_H2O': [(-14,-2), r'$\log\ \mathrm{H_2O}$'], 
    'log_CH4': [(-14,-2), r'$\log\ \mathrm{CH_4}$'], 
    #'log_NH3': [(-14,-2), r'$\log\ \mathrm{NH_3}$'], 
    #'log_H2S': [(-14,-2), r'$\log\ \mathrm{H_2S}$'], 
    'log_HF': [(-14,-2), r'$\log\ \mathrm{HF}$'], 
    'log_FeH': [(-14,-2), r'$\log\ \mathrm{FeH}$'], 'log_FeH_P': [(-5,3), r'$\log\ P_\mathrm{FeH}$'], 
    #'log_TiO': [(-14,-2), r'$\log\ \mathrm{TiO}$'], 
    #'log_VO': [(-14,-2), r'$\log\ \mathrm{VO}$'], 

    'log_K': [(-14,-2), r'$\log\ \mathrm{K}$'], 'log_K_P': [(-5,3), r'$\log\ P_\mathrm{K}$'], 
    'log_Na': [(-14,-2), r'$\log\ \mathrm{Na}$'], 
    'log_Cr': [(-14,-2), r'$\log\ \mathrm{Cr}$'], 
    #'log_Fe': [(-14,-2), r'$\log\ \mathrm{Fe}$'], 
    'log_Mn': [(-14,-2), r'$\log\ \mathrm{Mn}$'], 

    'log_CrH': [(-14,-2), r'$\log\ \mathrm{CrH}$'], 

    # Impact shifts: d = A*T^b * n/n_ref
    'A_d_0': [(0.001,0.004), r'$A_{d,0}$'], # 0.00158988 (K-H2) | 0.001943820 (K-He)
    'A_d_1': [(0.001,0.004), r'$A_{d,1}$'], # 0.00211668 (K-H2) | 0.000462539 (K-He)
    'b_d_0': [(0.5,1.5), r'$b_{d,0}$'],     # 0.949254 (K-H2) | 0.89691 (K-He)
    'b_d_1': [(0.5,1.5), r'$b_{d,1}$'],     # 0.933563 (K-H2) | 1.07284 (K-He)

    # Impact widths: w = A*T^b * n/n_ref
    'A_w_0': [(0.10,0.45), r'$A_{w,0}$'], # 0.352609 (K-H2) | 0.208190 (K-He)
    'A_w_1': [(0.05,0.35), r'$A_{w,1}$'], # 0.245926 (K-H2) | 0.121448 (K-He)
    'b_w_0': [(0.30,0.50), r'$b_{w,0}$'], # 0.385961 (K-H2) | 0.452833 (K-He)
    'b_w_1': [(0.35,0.60), r'$b_{w,1}$'], # 0.447971 (K-H2) | 0.531718 (K-He)

    # PT profile    
    'dlnT_dlnP_0': [(0.,0.4), r'$\nabla_{T,0}$'], 
    'dlnT_dlnP_1': [(0.,0.4), r'$\nabla_{T,1}$'], 
    'dlnT_dlnP_2': [(0.,0.4), r'$\nabla_{T,2}$'], 
    'dlnT_dlnP_3': [(0.,0.4), r'$\nabla_{T,3}$'], 
    'dlnT_dlnP_4': [(0.,0.4), r'$\nabla_{T,4}$'], 

    'T_phot': [(700.,2500.), r'$T_\mathrm{phot}$'], 
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

    # General properties
    'parallax': 496,  # +/- 37 mas
    'inclination': 0, # degrees

    #'vsini': 25.3, 

    # PT profile
    'log_P_knots': np.array([-5, -2, 0.5, 1.5, 3], dtype=np.float64), 

    #'do_scat_emis': True, 
    'do_scat_emis': False, 

    # Custom line opacity
    #'A_w_0_H2': 0.352609, 'b_w_0_H2': 0.385961, 
    #'A_w_1_H2': 0.245926, 'b_w_1_H2': 0.447971, 
    #'A_d_0_H2': 0.00158988, 'b_d_0_H2': 0.949254, 
    #'A_d_1_H2': 0.00211668, 'b_d_1_H2': 0.933563, 

    #'A_w_0_He': 0.208190, 'b_w_0_He': 0.452833, 
    #'A_w_1_He': 0.121448, 'b_w_1_He': 0.531718, 
    #'A_d_0_He': 0.001943820, 'b_d_0_He': 0.89691, 
    #'A_d_1_He': 0.000462539, 'b_d_1_He': 1.07284, 
}

#'''
parent_dir = '/home/sdregt/retrieval_base/retrieval_base/custom_opacity_data/'
line_opacity_kwargs = [
    {
    # --- Potassium (K I) --------------------
    'NIST_states_file': f'{parent_dir}/K_I_states.txt', 
    'VALD_trans_file': f'{parent_dir}/K_I_transitions.txt', 
    'pRT_name': 'K', 
    'mass': 39.0983, 
    'is_alkali': True, 
    #E_ion=35009.8140, Z=0, # Potassium (K I)
    
    #'line_cutoff': 200, 
    'line_cutoff': 1000, 
    #n_density_ref=1e20, 
    'log_gf_threshold': -2.0, 
    'log_gf_threshold_exact': -0.5, 
    'pre_compute': True, 

    'nu_0': [7983.67489, 8041.38112], 
    'log_gf': [-0.139, -0.439], 
    'E_low': [13042.876, 12985.170], 
    'gamma_N': [7.790, 7.790], 
    'gamma_vdW': [-7.021, -7.022], 
    }, 
    {
    # --- Sodium (Na I) ----------------------
    'NIST_states_file': f'{parent_dir}/Na_I_states.txt', 
    'VALD_trans_file': f'{parent_dir}/Na_I_transitions.txt', 
    'pRT_name': 'Na_allard_recomputed', 
    'mass': 22.989769, 
    'is_alkali': True, 
    'E_ion': 41449.451, #Z=0, # Sodium (Na I)
    
    #'line_cutoff': 200, 
    'line_cutoff': 1000, 
    #n_density_ref=1e20, 
    'log_gf_threshold': -2.0, 
    'log_gf_threshold_exact': -0.5, 
    'pre_compute': True, 
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
    #'cloud_mode': 'EddySed', 
    #'cloud_species': ['Mg2SiO4(c)_cd', 'MgSiO3(c)_cd', 'Fe(c)_cd'], 
}

#rotation_mode = 'integrate' # 'convolve'
rotation_mode = 'convolve'

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = {
    'chem_mode': 'free', #'SONORAchem' #'eqchem'

    'line_species': [
        'H2O_pokazatel_main_iso', 
        'CH4_hargreaves_main_iso', 
        #'NH3_coles_main_iso', 
        #'H2S_Sid_main_iso', 
        'HF_main_iso', 
        'FeH_main_iso', 
        #'TiO_48_Exomol_McKemmish', 
        #'VO_HyVO_main_iso', 

        'Cr', 
        #'Fe', 
        'Mn', 
        'CrH_main_iso', 
    ], 
}

species_to_plot_VMR = [
    'H2O', 'CH4', 'HF', 'FeH', 'Cr', 'Mn', 'CrH', 'K', 'Na', 
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
n_iter_before_update = 100