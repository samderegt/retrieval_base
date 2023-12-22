import numpy as np

file_params = 'config_J.py'

####################################################################################
# Files and physical parameters
####################################################################################

# Where to store retrieval outputs
prefix = 'test'
prefix = f'./retrieval_outputs/{prefix}/test_'


config_data = {
    'J1226': {
        'w_set': 'J1226', # Wavelength setting
        #'wave_range': (1115, 1325), 
        'wave_range': (1239, 1267), # Range to fit, doesn't have to be full spectrum

        # Data filenames
        'file_target': './data/LSPM_J0036_J.dat', 
        'file_std': './data/LSPM_J0036_std_J.dat', 
        'file_wave': './data/LSPM_J0036_std_J.dat', # Use std-observation wlen-solution

        # Telluric transmission
        'file_skycalc_transm': './data/skycalc_transm_J1226.dat', 
        'file_molecfit_transm': './data/LSPM_J0036_J_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/LSPM_J0036_std_J_molecfit_transm.dat', 
        'file_std_molecfit_continuum': './data/LSPM_J0036_std_J_molecfit_continuum.dat', # Continuum fit by molecfit

        'filter_2MASS': '2MASS/2MASS.J', # Magnitude used for flux-calibration
        
        'pwv': 2.5, # Precipitable water vapour

        # Telescope-pointing, used for barycentric velocity-correction
        'ra': 9.073374, 'dec': 18.35260, 'mjd': 59888.11823169,
        'ra_std': 9.196928, 'dec_std': 15.23084, 'mjd_std': 59888.07684836, 

        # Some info on standard-observation
        'T_std': 19000, 'log_g_std': 2.3, 'rv_std': -8.00, 'vsini_std': 40, 
        
        # Slit-width, sets model resolution
        'slit': 'w_0.2', 

        # Down-sample opacities for faster radiative transfer
        'lbl_opacity_sampling': 3, 

        # Ignore pixels with lower telluric transmission
        'tell_threshold': 0.7, 
        'sigma_clip_width': 5, # Remove outliers
    
        # Define pRT's log-pressure grid
        'log_P_range': (-5,3), 
        'n_atm_layers': 50, 
        }, 
    }

# Magnitudes used for flux-calibration
magnitudes = {
    '2MASS/2MASS.J': (12.466, 0.027), # Cutri et al. (2003)
    '2MASS/2MASS.Ks': (11.058, 0.021), 
}

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Data resolution
    #'log_res_K2166': [(4,5.2), r'$\log\ R_\mathrm{K}$'], 

    # Uncertainty scaling
    #'log_a': [(-0.7,0.25), r'$\log\ a_\mathrm{K}$'], 
    #'log_l': [(-3,-1), r'$\log\ l_\mathrm{K}$'], 

    # General properties
    'R_p': [(0.5,1.5), r'$R_\mathrm{p}$'], 
    'log_g': [(4,6.0), r'$\log\ g$'], 
    'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities
    'vsini': [(30,50), r'$v\ \sin\ i$'], 
    'rv': [(10,30), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    #'log_opa_base_gray': [(-10,5), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    #'log_P_base_gray': [(-5,3), r'$\log\ P_{\mathrm{cl},0}$'], 
    #'f_sed_gray': [(0,20), r'$f_\mathrm{sed}$'], 

    # Chemistry
    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'], 
    'log_CH4': [(-12,-2), r'$\log\ \mathrm{CH_4}$'], 
    'log_NH3': [(-12,-2), r'$\log\ \mathrm{NH_3}$'], 
    'log_FeH': [(-12,-2), r'$\log\ \mathrm{FeH}$'], 
    'log_TiO': [(-12,-2), r'$\log\ \mathrm{TiO}$'], 
    
    'log_K': [(-12,-2), r'$\log\ \mathrm{K}$'], 
    'log_Na': [(-12,-2), r'$\log\ \mathrm{Na}$'], 
    'log_Fe': [(-12,-2), r'$\log\ \mathrm{Fe}$'], 

    # PT profile
    'dlnT_dlnP_0': [(0.12, 0.4), r'$\nabla_{T,0}$'], 
    'dlnT_dlnP_1': [(0.13,0.4), r'$\nabla_{T,1}$'], 
    'dlnT_dlnP_2': [(0.03,0.4), r'$\nabla_{T,2}$'], 
    'dlnT_dlnP_3': [(0.0,0.4), r'$\nabla_{T,3}$'], 
    'dlnT_dlnP_4': [(-0.2,0.2), r'$\nabla_{T,4}$'], 

    'T_0': [(3000,15000), r'$T_0$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 114.4735,  # +/- 0.1381 mas
    'inclination': 0, # degrees

    # PT profile
    'log_P_knots': np.array([-5, -1.5, 0, 1.5, 3], dtype=np.float64), 
}

####################################################################################
#
####################################################################################

scale_flux = True
scale_err  = True
apply_high_pass_filter = False

cloud_mode = None #'gray'
cloud_species = None

rotation_mode = 'convolve' #'integrate'

####################################################################################
# Chemistry parameters
####################################################################################

chem_mode  = 'free' #'SONORAchem' #'eqchem'

chem_kwargs = dict(
    spline_order = 0, 

    quench_setup = {
        'P_quench_CO_CH4': ['12CO', 'CH4', 'H2O', '13CO', 'C18O', 'C17O'], 
    #    #'P_quench_N2_NH3': ['N2', 'HCN', 'NH3'], 
    #    'P_quench_N2_NH3': ['N2', 'NH3'], 
        }, 
)

line_species = [
    'H2O_pokazatel_main_iso', 
    'CH4_hargreaves_main_iso', 
    'NH3_coles_main_iso', 
    'FeH_main_iso', 
    'TiO_48_Exomol_McKemmish',  

    'K', 
    'Na_allard', 
    'Fe', 
    ]
species_to_plot_VMR = [
    'H2O', 'CH4', 'NH3', 'FeH', 'TiO', 'K', 'Na', 'Fe', 
    ]
species_to_plot_CCF = species_to_plot_VMR

####################################################################################
# Covariance parameters
####################################################################################

cov_mode = None #'GP'

cov_kwargs = dict(
    trunc_dist   = 3, 
    scale_GP_amp = True, 
    max_separation = 20, 

    # Prepare the wavelength separation and
    # average squared error arrays and keep 
    # in memory
    prepare_for_covariance = True
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
n_live_points = 100
n_iter_before_update = 100
