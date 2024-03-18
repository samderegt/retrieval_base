import numpy as np

file_params = 'config_fiducial_K_AB.py'

####################################################################################
# Files and physical parameters
####################################################################################

# Where to store retrieval outputs
prefix = 'fiducial_K_AB_ret_2'
prefix = f'./retrieval_outputs/{prefix}/test_'


config_data = dict(
    K2166_A = {
        'w_set': 'K2166', # Wavelength setting
        'wave_range': (1900, 2500), # Range to fit, doesn't have to be full spectrum

        # Data filenames
        'file_target': './data/2022_12_31/Luhman_16AB_K.dat', 
        'file_std': './data/2022_12_31/Luhman_16_std_K.dat', 
        'file_wave': './data/2022_12_31/Luhman_16_std_K.dat', # Use std-observation wlen-solution
    
        # Telluric transmission
        'file_skycalc_transm': './data/2022_12_31/skycalc_transm_K2166.dat', 
        'file_molecfit_transm': './data/2022_12_31/Luhman_16AB_K_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/2022_12_31/Luhman_16_std_K_molecfit_transm.dat', 
        'file_std_molecfit_continuum': './data/2022_12_31/Luhman_16_std_K_molecfit_continuum.dat', # Continuum fit by molecfit

        'filter_2MASS': '2MASS/2MASS.Ks', # Magnitude used for flux-calibration
        
        'pwv': 1.5, # Precipitable water vapour

        # Telescope-pointing, used for barycentric velocity-correction
        'ra': 162.297895, 'dec': -53.31703, 'mjd': 59945.22453985, 
        'ra_std': 161.739683, 'dec_std': -56.75788, 'mjd_std': 59945.2197638, 

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
config_data['K2166_B'] = config_data['K2166_A'].copy()

# Magnitudes used for flux-calibration
magnitudes = {
    '2MASS/2MASS.Ks': (8.841, 0.02), # Burgasser et al. (2013)
}

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {

    # Parameters specific to one wavelength-setting
    'K2166_A': {
        # General properties
        'R_p': [(0.5,1.2), r'$R_\mathrm{p}$'], 
        'log_g': [(4,6.0), r'$\log\ g$'], 
        'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

        # Velocities #km/s
        'vsini': [(10,30), r'$v\ \sin\ i$'], 
        'rv': [(10,30), r'$v_\mathrm{rad}$'], 

        # Chemistry
        'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'],
        'log_CH4': [(-12,-2), r'$\log\ \mathrm{CH_4}$'],
        'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'],
        'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'],
        'log_NH3': [(-12,-2), r'$\log\ \mathrm{NH_3}$'],
        'log_H2S': [(-12,-2), r'$\log\ \mathrm{H_2S}$'],
        'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'],

        # PT profile
        'dlnT_dlnP_0': [(0.1,0.4), r'$\nabla_{T,0}$'], 
        'dlnT_dlnP_1': [(0.1,0.4), r'$\nabla_{T,1}$'], 
        'dlnT_dlnP_2': [(0.1,0.4), r'$\nabla_{T,2}$'], 
        'dlnT_dlnP_3': [(0.03,0.4), r'$\nabla_{T,3}$'], 
        'dlnT_dlnP_4': [(0.0,0.4), r'$\nabla_{T,4}$'], 
        'dlnT_dlnP_5': [(-0.1,0.2), r'$\nabla_{T,5}$'],

        'T_0': [(2000,15000), r'$T_0$'], 
    }, 

    'K2166_B': {
        # General properties
        'R_p': [(0.5,1.2), r'$R_\mathrm{p}$'], 
        'log_g': [(4,6.0), r'$\log\ g$'], 
        'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

        # Velocities #km/s
        'vsini': [(10,30), r'$v\ \sin\ i$'], 
        'rv': [(10,30), r'$v_\mathrm{rad}$'], 

        # Chemistry
        'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'],
        'log_CH4': [(-12,-2), r'$\log\ \mathrm{CH_4}$'],
        'log_12CO': [(-12,-2), r'$\log\ \mathrm{^{12}CO}$'],
        'log_13CO': [(-12,-2), r'$\log\ \mathrm{^{13}CO}$'],
        'log_NH3': [(-12,-2), r'$\log\ \mathrm{NH_3}$'],
        'log_H2S': [(-12,-2), r'$\log\ \mathrm{H_2S}$'],
        'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'],

        # PT profile
        'dlnT_dlnP_0': [(0.1,0.4), r'$\nabla_{T,0}$'], 
        'dlnT_dlnP_1': [(0.1,0.4), r'$\nabla_{T,1}$'], 
        'dlnT_dlnP_2': [(0.1,0.4), r'$\nabla_{T,2}$'], 
        'dlnT_dlnP_3': [(0.03,0.4), r'$\nabla_{T,3}$'], 
        'dlnT_dlnP_4': [(0.0,0.4), r'$\nabla_{T,4}$'], 
        'dlnT_dlnP_5': [(-0.1,0.2), r'$\nabla_{T,5}$'], 

        'T_0': [(2000,15000), r'$T_0$'], 
    }, 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 496,  # +/- 37 mas
    'inclination': 0, # degrees

    # PT profile
    'log_P_knots': np.array([-5, -2, -0.5, 0.5, 1.5, 3], dtype=np.float64), 
}

####################################################################################
#
####################################################################################

sum_m_spec = True

scale_flux = True
scale_err  = True
apply_high_pass_filter = False

cloud_kwargs = dict(
    K2166_A = {'cloud_mode': None}, 
    K2166_B = {'cloud_mode': None}, 
)

rotation_mode = 'convolve' #'integrate'

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = dict(   
    K2166_A = {
        'chem_mode': 'free', #'SONORAchem' #'eqchem'

        'line_species': [
            'H2O_pokazatel_main_iso', 
            'CH4_hargreaves_main_iso', 
            'CO_main_iso', 
            'CO_36', 
            'NH3_coles_main_iso', 
            'H2S_Sid_main_iso', 
            'HF_main_iso', 
        ], 
    }, 

    K2166_B = {
        'chem_mode': 'free', #'SONORAchem' #'eqchem'

        'line_species': [
            'H2O_pokazatel_main_iso', 
            'CH4_hargreaves_main_iso', 
            'CO_main_iso', 
            'CO_36', 
            'NH3_coles_main_iso', 
            'H2S_Sid_main_iso', 
            'HF_main_iso', 
        ], 
    }, 
)

species_to_plot_VMR = [
    'H2O', 'CH4', '12CO', '13CO', 'NH3', 'H2S', 'HF'
    ]
species_to_plot_CCF = species_to_plot_VMR

####################################################################################
# Covariance parameters
####################################################################################

cov_kwargs = dict(
    cov_mode = None, # 'GP'
    
    # Prepare the wavelength separation and
    # average squared error arrays and keep 
    # in memory
    prepare_for_covariance = False
)

'''
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
'''

####################################################################################
# PT parameters
####################################################################################

PT_kwargs = dict(
    PT_mode = 'free_gradient', 
    n_T_knots = len(constant_params['log_P_knots']), 
)

####################################################################################
# Multinest parameters
####################################################################################

const_efficiency_mode = True
sampling_efficiency = 0.05
evidence_tolerance = 0.5
n_live_points = 200
n_iter_before_update = 200
