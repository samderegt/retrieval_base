import numpy as np

file_params = 'config_fiducial_K.py'

####################################################################################
# Files and physical parameters
####################################################################################

# Where to store retrieval outputs
prefix = 'fiducial_J_ret_2'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    J1226_cloudy = {
        'w_set': 'J1226', # Wavelength setting
        'wave_range': (1155.3, 1325), # Range to fit, doesn't have to be full spectrum

        # Data filenames
        'file_target': './data/2M_0559_J.dat', 
        'file_std': './data/2M_0559_std_J.dat', 
        'file_wave': './data/2M_0559_J.dat', # Use direct observation (metrology wasn't used correctly)
    
        # Telluric transmission
        'file_molecfit_transm': './data/2M_0559_J_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/2M_0559_std_J_molecfit_transm.dat', 
        'file_std_molecfit_continuum': './data/2M_0559_std_J_molecfit_continuum.dat', # Continuum fit by molecfit

        'filter_2MASS': '2MASS/2MASS.J', # Magnitude used for flux-calibration
        'pwv': 10., # Precipitable water vapour

        # Telescope-pointing, used for barycentric velocity-correction
        'ra': 89.834193, 'dec': -14.08239, 'mjd': 59977.26137825, 
        'ra_std': 90.460453, 'dec_std': -10.59824, 'mjd_std': 59977.20176338, 

        # Some info on standard-observation
        'T_std': 16258, 

        # Slit-width, sets model resolution
        'slit': 'w_0.4', 

        # Ignore pixels with lower telluric transmission
        'tell_threshold': 0.8, 
        'sigma_clip_width': 5, # Remove outliers
        }, 
)
#config_data['K2166_band'] = config_data['K2166_cloudy'].copy()
#config_data['K2166_spot'] = config_data['K2166_cloudy'].copy()

# Magnitudes used for flux-calibration
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
    # Covariance parameters
    'log_a': [(-0.3,0.5), r'$\log\ a$'], 
    'log_l': [(-2.5,-1.2), r'$\log\ l$'], 

    # General properties
    #'R_p': [(0.5,1.5), r'$R_\mathrm{p}$'], 
    'log_g': [(4.,6.), r'$\log\ g$'], 
    'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities #km/s
    'vsini': [(15.,40.), r'$v\ \sin\ i$'], 
    'rv': [(-20.,0.), r'$v_\mathrm{rad}$'], 

    # Cloud properties
    'log_opa_base_gray': [(-10.,5.), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    'log_P_base_gray': [(-5.,3.), r'$\log\ P_{\mathrm{cl},0}$'], 
    'f_sed_gray': [(0.,20.), r'$f_\mathrm{sed}$'], 

    # Chemistry
    'log_H2O':     [(-12.,-2.), r'$\log\ \mathrm{H_2O}$'],
    'log_CH4':     [(-12.,-2.), r'$\log\ \mathrm{CH_4}$'],
    'log_NH3':     [(-12.,-2.), r'$\log\ \mathrm{NH_3}$'],
    'log_HF':      [(-12.,-2.), r'$\log\ \mathrm{HF}$'],

    'log_HF': [(-12,-2), r'$\log\ \mathrm{HF}$'],
    'log_FeH': [(-12,-2), r'$\log\ \mathrm{FeH}$'],
    'log_TiO': [(-12,-2), r'$\log\ \mathrm{TiO}$'], 
    'log_VO': [(-12,-2), r'$\log\ \mathrm{VO}$'],   
    'log_KshiftH2': [(-12,-2), r'$\log\ \mathrm{K}$'], 
    'log_Na': [(-12,-2), r'$\log\ \mathrm{Na}$'], 
    'log_Fe': [(-12,-2), r'$\log\ \mathrm{Fe}$'],

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

    # Data resolution
    'res': 50000, 

    # General properties
    'parallax': 95.2714,  # +/- 0.7179 mas
    'inclination': 0., # degrees

    'relative_scaling': False, 
}

####################################################################################
#
####################################################################################

sum_m_spec = len(config_data) > 1

#scale_flux = True
scale_flux = True
scale_err  = True
apply_high_pass_filter = False

cloud_kwargs = {
    'cloud_mode': 'gray', 
    #'cloud_mode': None, 
    #'K2166_cloudy': {'cloud_mode': 'gray'}, 
    #'K2166_clear': {'cloud_mode': None}, 
}

rotation_mode = 'convolve' # 'integrate' 

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = {
    'chem_mode': 'free', #'SONORAchem' #'eqchem'

    'line_species': [
        'H2O_pokazatel_main_iso', 
        'CH4_hargreaves_main_iso', 
        'NH3_coles_main_iso', 
        'HF_main_iso', 

        'FeH_main_iso', 
        'TiO_48_Exomol_McKemmish', 
        'VO_ExoMol_McKemmish', 
        'KshiftH2',
        'Na_allard_recomputed', 
        'Fe', 
    ], 
}

species_to_plot_VMR = [
    #'H2O', 'CH4', '12CO', '13CO', 'NH3', 'H2S', 'HF', 
    'H2O', 'CH4', 'NH3', 'HF', 'FeH', 'TiO', 'VO', 'KshiftH2', 'Na', 'Fe'
    ]
species_to_plot_CCF = species_to_plot_VMR

####################################################################################
# Covariance parameters
####################################################################################

cov_kwargs = dict(
    cov_mode = 'GP', #None, 
    
    trunc_dist   = 3, 
    scale_GP_amp = True, 

    # Prepare the wavelength separation and average squared error- 
    # arrays and keep in memory
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
n_live_points = 100
n_iter_before_update = 100