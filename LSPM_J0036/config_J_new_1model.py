import numpy as np

file_params = 'config_J_new_1model.py'

####################################################################################
# Files and physical parameters
####################################################################################

# Where to store retrieval outputs
prefix = 'test_13'
prefix = f'./retrieval_outputs/{prefix}/test_'


config_data = dict(
    J1226_cloudy = {
        'w_set': 'J1226', # Wavelength setting
        #'wave_range': (1115, 1360), 
        #'wave_range': (1316.5, 1325), # Range to fit, doesn't have to be full spectrum
        'wave_range': (1296, 1325), 

        #'wave_to_mask': np.array([[1168, 1179]]), #mask K I lines 

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
)

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

    # General properties
    'R_p': [(0.5,8), r'$R_\mathrm{p}$'], 
    'log_g': [(3,6), r'$\log\ g$'], 
    'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities #km/s
    'vsini': [(20,50), r'$v\ \sin\ i$'], 
    'rv': [(0,40), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10,5), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    'log_P_base_gray': [(-5,2), r'$\log\ P_{\mathrm{cl},0}$'], 
    'f_sed_gray': [(1,20), r'$f_\mathrm{sed}$'], 

    # Chemistry
    'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'], 
    'log_TiO': [(-12,-2), r'$\log\ \mathrm{TiO}$'], 
    'log_VO': [(-12,-2), r'$\log\ \mathrm{VO}$'], 
    #'log_H2S': [(-12,-2), r'$\log\ \mathrm{H_2S}$'], 
    #'log_Fe': [(-12,-2), r'$\log\ \mathrm{Fe}$'], 
    #'log_Ti': [(-12,-2), r'$\log\ \mathrm{Ti}$'], 
    'log_Cr': [(-12,-2), r'$\log\ \mathrm{Cr}$'], 
    #'log_OH': [(-12,-2), r'$\log\ \mathrm{OH}$'], 
    #'log_CH': [(-12,-2), r'$\log\ \mathrm{CH}$'], 
    #'log_AlH': [(-12,-2), r'$\log\ \mathrm{AlH}$'], 
    'log_FeH': [(-12,-2), r'$\log\ \mathrm{FeH}$'], 
    'log_MgH': [(-12,-2), r'$\log\ \mathrm{MgH}$'], 
    'log_CrH': [(-12,-2), r'$\log\ \mathrm{CrH}$'], 
    'log_TiH': [(-12,-2), r'$\log\ \mathrm{TiH}$'], 
    #'log_CaH': [(-12,-2), r'$\log\ \mathrm{CaH}$'], 
    #'log_NaH': [(-12,-2), r'$\log\ \mathrm{NaH}$'], 
    #'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'],

    #'log_H2O': [(-12,-2), r'$\log\ \mathrm{H_2O}$'],
    #'log_FeH': [(-12,-2), r'$\log\ \mathrm{FeH}$'],
    #'log_TiO': [(-12,-2), r'$\log\ \mathrm{TiO}$'],
    #'log_VO': [(-12,-2), r'$\log\ \mathrm{VO}$'],
    #'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'],
    #'log_H2lines': [(-12,0), r'$\log\ \mathrm{H_2}$'],

    # PT profile
    #'dlnT_dlnP_0': [(0.12, 0.4), r'$\nabla_{T,0}$'], 
    #'dlnT_dlnP_1': [(0.13,0.4), r'$\nabla_{T,1}$'], 
    #'dlnT_dlnP_2': [(0.03,0.4), r'$\nabla_{T,2}$'], 
    #'dlnT_dlnP_3': [(0.0,0.4), r'$\nabla_{T,3}$'], 
    #'dlnT_dlnP_4': [(-0.1,0.2), r'$\nabla_{T,4}$'], 

    #'T_0': [(2000,15000), r'$T_0$'], 

    'dlnT_dlnP_0': [(0.0,0.4), r'$\nabla_{T,0}$'], 
    'dlnT_dlnP_1': [(0.0,0.4), r'$\nabla_{T,1}$'], 
    'dlnT_dlnP_2': [(0.0,0.4), r'$\nabla_{T,2}$'], 
    'dlnT_dlnP_3': [(-0.1,0.4), r'$\nabla_{T,3}$'], 
    'dlnT_dlnP_4': [(-0.2,0.2), r'$\nabla_{T,4}$'], 

    'T_phot': [(200,3000), r'$T_\mathrm{phot}$'], 
    'log_P_phot': [(-3,1), r'$\log\ P_\mathrm{phot}$'], 
    'd_log_P_phot+1': [(0,2), r'$\log\ P_\mathrm{up}$'],
    'd_log_P_phot-1': [(0,2), r'$\log\ P_\mathrm{low}$'], 

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

sum_m_spec = False

scale_flux = True
scale_err  = True
apply_high_pass_filter = False

cloud_kwargs = dict(
    cloud_mode = 'gray', 
)

rotation_mode = 'convolve' #'integrate'

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = dict(
    chem_mode = 'free', #'SONORAchem' #'eqchem'
    #chem_mode = 'SONORAchem', path_SONORA_chem = '/net/lem/data1/regt/retrieval_base/SONORA_models/chemistry', 

    line_species = [
        'H2O_pokazatel_main_iso', 
        'TiO_48_Exomol_McKemmish', 
        'VO_ExoMol_McKemmish', 
        #'H2S_Sid_main_iso', 
        #'Fe', 
        #'Ti', 
        'Cr', 
        #'AlO_main_iso', 
        #'OH_main_iso', 
        #'CH_main_iso', 
        #'AlH_main_iso', 
        'FeH_main_iso', 
        'MgH_main_iso', 
        'CrH_main_iso', 
        'TiH_main_iso', 
        #'CaH_main_iso', 
        #'NaH_main_iso', 
        #'HF_main_iso', 
    ], 
)

species_to_plot_VMR = [
    #'H2O', 'FeH', 'TiO', 'VO', 'HF', #'H2lines'
    'H2O', 
    'TiO', 
    'VO', 
    #'H2S', 
    #'Fe', 
    #'Ti', 
    'Cr', 
    #'AlO', 
    #'OH', 
    #'CH'
    'FeH', 
    'MgH', 
    'CrH', 
    'TiH', 
    #'CaH', 
    #'NaH', 
    #'HF', 
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

####################################################################################
# PT parameters
####################################################################################

PT_kwargs = dict(
    PT_mode = 'free_gradient', 
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
