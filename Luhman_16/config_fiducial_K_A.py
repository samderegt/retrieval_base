import numpy as np

file_params = 'config_fiducial_K_A.py'

####################################################################################
# Files and physical parameters
####################################################################################

# Where to store retrieval outputs
prefix = 'fiducial_K_A_ret_16'
prefix = f'./retrieval_outputs/{prefix}/test_'


config_data = dict(
    K2166_A = {
        'w_set': 'K2166', # Wavelength setting
        'wave_range': (1900, 2500), # Range to fit, doesn't have to be full spectrum

        # Data filenames
        'file_target': './data/Luhman_16A_K.dat', 
        'file_std': './data/Luhman_16_std_K.dat', 
        #'file_wave': './data/Luhman_16_std_K.dat', # Use std-observation wlen-solution
        'file_wave': './data/Luhman_16_std_K_molecfit_transm.dat', # Use molecfit wlen-solution
    
        # Telluric transmission
        'file_skycalc_transm': './data/skycalc_transm_K2166.dat', 
        'file_molecfit_transm': './data/Luhman_16A_K_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/Luhman_16_std_K_molecfit_transm.dat', 
        'file_std_molecfit_continuum': './data/Luhman_16_std_K_molecfit_continuum.dat', # Continuum fit by molecfit

        'filter_2MASS': '2MASS/2MASS.Ks', # Magnitude used for flux-calibration
        
        'pwv': 1.5, # Precipitable water vapour

        # Telescope-pointing, used for barycentric velocity-correction
        'ra': 162.297895, 'dec': -53.31703, 'mjd': 59946.32563173, 
        'ra_std': 161.739683, 'dec_std': -56.75788, 'mjd_std': 59946.31615474, 

        # Some info on standard-observation
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

# Magnitudes used for flux-calibration
magnitudes = {
    '2MASS/2MASS.J': (11.53, 0.04), # Burgasser et al. (2013)
    '2MASS/2MASS.Ks': (9.44, 0.07), 
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
    'R_p': [(0.5,1.2), r'$R_\mathrm{p}$'], 
    'log_g': [(4,6.0), r'$\log\ g$'], 
    #'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities #km/s
    'vsini': [(10,30), r'$v\ \sin\ i$'], 
    'rv': [(10,30), r'$v_\mathrm{rad}$'], 

    # Chemistry
    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'],
    'log_H2(18)O': [(-12,-2), r'$\log\ \mathrm{H_2^{18}O}$'],
    #'log_H2(17)O': [(-12,-2), r'$\log\ \mathrm{H_2^{17}O}$'],
    
    'log_CH4': [(-12,-2), r'$\log\ \mathrm{CH_4}$'],
    #'log_13CH4': [(-12,-2), r'$\log\ \mathrm{^{13}CH_4}$'],
    
    'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'],
    'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'],
    'log_C18O': [(-12,-2), r'$\log\ \mathrm{C^{18}O}$'],
    #'log_C17O': [(-12,-2), r'$\log\ \mathrm{C^{17}O}$'],
    
    #'log_CO2': [(-12,-2), r'$\log\ \mathrm{CO_2}$'],
    'log_NH3': [(-12,-2), r'$\log\ \mathrm{NH_3}$'],
    'log_H2S': [(-12,-2), r'$\log\ \mathrm{H_2S}$'],
    #'log_HCN': [(-12,-2), r'$\log\ \mathrm{HCN}$'],

    'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'],
    #'log_Ca': [(-12,-2), r'$\log\ \mathrm{Ca}$'],

    # PT profile    
    'dlnT_dlnP_0': [(0.0,0.4), r'$\nabla_{T,0}$'], 
    'dlnT_dlnP_1': [(0.0,0.4), r'$\nabla_{T,1}$'], 
    'dlnT_dlnP_2': [(0.0,0.4), r'$\nabla_{T,2}$'], 
    'dlnT_dlnP_3': [(-0.1,0.4), r'$\nabla_{T,3}$'], 
    'dlnT_dlnP_4': [(-0.2,0.2), r'$\nabla_{T,4}$'], 

    'T_phot': [(200,3000), r'$T_\mathrm{phot}$'], 
    'log_P_phot': [(-3,1), r'$\log\ P_\mathrm{phot}$'], 
    'd_log_P_phot+1': [(0,2), r'$\log\ P_\mathrm{up}$'], 
    #'d_log_P_phot-1': [(0,2), r'$\log\ P_\mathrm{low}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 496,  # +/- 37 mas
    'inclination': 0, # degrees

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
    'cloud_mode': None, 
}

rotation_mode = 'integrate' # 'convolve'

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = {
    'chem_mode': 'free', #'SONORAchem' #'eqchem'

    'line_species': [
        'H2O_pokazatel_main_iso', 
        'H2O_181', 
        #'H2O_171', 

        'CH4_hargreaves_main_iso', 
        #'13CH4_hargreaves', 

        'CO_main_iso', 
        'CO_36', 
        'CO_28', 
        #'CO_27', 

        #'CO2_main_iso', 
        'NH3_coles_main_iso', 
        'H2S_Sid_main_iso', 
        #'HCN_main_iso', 

        'HF_main_iso', 
        #'Ca', 
    ], 
}

species_to_plot_VMR = [
    'H2O', 'H2(18)O', #'H2(17)O', 
    'CH4', #'13CH4', 
    '12CO', '13CO', #'C18O', 'C17O', 
    #'CO2', 
    'NH3', 'H2S', #'HCN', 
    'HF', #'Ca', 
    ]
species_to_plot_CCF = species_to_plot_VMR

####################################################################################
# Covariance parameters
####################################################################################

cov_kwargs = dict(
    cov_mode = 'GP', #None, 
    
    trunc_dist   = 3, 
    scale_GP_amp = True, 
    #max_separation = 20, 

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
    symmetric_around_P_phot = True, 
)

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 100
n_iter_before_update = 200
