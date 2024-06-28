import numpy as np

file_params = 'config_fiducial_K_B.py'

####################################################################################
# Files and physical parameters
####################################################################################

# Where to store retrieval outputs
prefix = 'no_bands_K_B_ret_11'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    K2166_cloudy = {
        'w_set': 'K2166', # Wavelength setting
        'wave_range': (2040, 2500), # Range to fit, doesn't have to be full spectrum

        # Data filenames
        'file_target': './data/Luhman_16B_K.dat', 
        'file_std': './data/Luhman_16_std_K.dat', 
        #'file_wave': './data/Luhman_16_std_K.dat', # Use std-observation wlen-solution
        'file_wave': './data/Luhman_16_std_K_molecfit_transm.dat', # Use molecfit wlen-solution
    
        # Telluric transmission
        'file_molecfit_transm': './data/Luhman_16B_K_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/Luhman_16_std_K_molecfit_transm.dat', 
        'file_std_molecfit_continuum': './data/Luhman_16_std_K_molecfit_continuum.dat', # Continuum fit by molecfit

        'filter_2MASS': '2MASS/2MASS.Ks', # Magnitude used for flux-calibration
        'pwv': 1.5, # Precipitable water vapour

        # Telescope-pointing, used for barycentric velocity-correction
        'ra': 162.297895, 'dec': -53.31703, 'mjd': 59946.32563173, 
        'ra_std': 161.739683, 'dec_std': -56.75788, 'mjd_std': 59946.31615474, 

        # Some info on standard-observation
        'T_std': 15000, 

        # Slit-width, sets model resolution
        'slit': 'w_0.4', 

        # Ignore pixels with lower telluric transmission
        'tell_threshold': 0.7, 
        'sigma_clip_width': 5, # Remove outliers
        }, 
)
#config_data['K2166_band'] = config_data['K2166_cloudy'].copy()
#config_data['K2166_spot'] = config_data['K2166_cloudy'].copy()

# Magnitudes used for flux-calibration
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
    'log_a': [(-0.3,0.2), r'$\log\ a$'], 
    'log_l': [(-2.5,-1.2), r'$\log\ l$'], 

    # General properties
    #'R_p': [(0.5,1.5), r'$R_\mathrm{p}$'], 
    'log_g': [(4.,6.), r'$\log\ g$'], 
    #'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Velocities #km/s
    'vsini': [(20.,30.), r'$v\ \sin\ i$'], 
    'rv': [(15.,25.), r'$v_\mathrm{rad}$'], 

    # Surface brightness.
    #'lat_band_cen': [(-30.,30.), r'$\phi_\mathrm{b,cen}$'], 
    #'lat_band': [(0.,90.), r'$\phi_\mathrm{b}$'], 
    #'epsilon_band': [(0.,1.5), r'$\epsilon_\mathrm{b}$'], 
    #'vsini_band': [(20.,30.), r'$v\ \sin\ i_\mathrm{b}$'], 

    #'lat_band_cen_1': [(-30.,30.), r'$\phi_\mathrm{b,cen}$'], 
    #'lat_band_1': [(0.,90.), r'$\phi_\mathrm{b}$'], 
    #'epsilon_band_1': [(0.,1.5), r'$\epsilon_\mathrm{b}$'], 

    #'r_spot': [(0.,1.), r'$r_\mathrm{s}$'], 
    #'theta_spot': [(-180.,180.), r'$\theta_\mathrm{s}$'], 
    #'radius_spot': [(0.1,0.5), r'$\sigma_\mathrm{s}$'], 
    #'epsilon_spot': [(0.,5.), r'$\epsilon_\mathrm{s}$'], 

    # Cloud properties
    #'log_opa_base_gray': [(-10.,15.), r'$\log\ \kappa_{\mathrm{cl},0}$'], 
    #'log_P_base_gray': [(-5.,3.), r'$\log\ P_{\mathrm{cl},0}$'], 
    #'f_sed_gray': [(0.,20.), r'$f_\mathrm{sed}$'], 
    #'cloud_slope': [(0.,10.), r'$\alpha_\mathrm{cl}$'], 
    #'wave_cloud_0': [(), r'$\lambda_{\mathrm{cl},0}$'], 
    
    'log_X_Mg2SiO4(c)': [(-2.3,1.), r'$\log\ X_\mathrm{Mg2SiO4}$'], 
    'f_sed_Mg2SiO4(c)': [(0.,10.), r'$f_\mathrm{sed,Mg2SiO4}$'], 
    'log_X_MgSiO3(c)': [(-2.3,1.), r'$\log\ X_\mathrm{MgSiO3}$'], 
    'f_sed_MgSiO3(c)': [(0.,10.), r'$f_\mathrm{sed,MgSiO3}$'], 
    'log_X_Fe(c)': [(-2.3,1.), r'$\log\ X_\mathrm{Fe}$'], 
    'f_sed_Fe(c)': [(0.,10.), r'$f_\mathrm{sed,Fe}$'], 

    'log_K_zz': [(5.,13.), r'$\log\ K_\mathrm{zz}$'], 
    'sigma_g': [(1.05,3), r'$\sigma_\mathrm{g}$'], 

    # Parameters specific to model-settings
    #'K2166_spot': {
    #    # Chemistry
    #    'log_H2O':  [(-12.,-2.), r'$\log\ \mathrm{H_2O}_\mathrm{s}$'],
    #    'log_CH4':  [(-12.,-2.), r'$\log\ \mathrm{CH_4}_\mathrm{s}$'],
    #    'log_12CO': [(-12.,-2.), r'$\log\ \mathrm{^{12}CO}_\mathrm{s}$'],
    #    }, 
    #'K2166_cloudy': {
    #    # Chemistry
    #    'log_H2O':  [(-12.,-2.), r'$\log\ \mathrm{H_2O}$'],
    #    'log_CH4':  [(-12.,-2.), r'$\log\ \mathrm{CH_4}$'],
    #    'log_12CO': [(-12.,-2.), r'$\log\ \mathrm{^{12}CO}$'],
    #    }, 
    #'K2166_band': {
    #    # Chemistry
    #    'log_H2O':  [(-12.,-2.), r'$\log\ \mathrm{H_2O}_\mathrm{b}$'],
    #    'log_CH4':  [(-12.,-2.), r'$\log\ \mathrm{CH_4}_\mathrm{b}$'],
    #    #'log_12CO': [(-12.,-2.), r'$\log\ \mathrm{^{12}CO}_\mathrm{b}$'],
    #    }, 

    # Chemistry
    'log_H2O':     [(-12.,-2.), r'$\log\ \mathrm{H_2O}$'],
    'log_CH4':     [(-12.,-2.), r'$\log\ \mathrm{CH_4}$'],
    'log_12CO':    [(-12.,-2.), r'$\log\ \mathrm{^{12}CO}$'],
    
    'log_H2(18)O': [(-12.,-2.), r'$\log\ \mathrm{H_2^{18}O}$'],
    'log_13CO':    [(-12.,-2.), r'$\log\ \mathrm{^{13}CO}$'],
    'log_C18O':    [(-12.,-2.), r'$\log\ \mathrm{C^{18}O}$'],
    'log_NH3':     [(-12.,-2.), r'$\log\ \mathrm{NH_3}$'],
    'log_H2S':     [(-12.,-2.), r'$\log\ \mathrm{H_2S}$'],
    'log_HF':      [(-12.,-2.), r'$\log\ \mathrm{HF}$'],

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
    'res': 60000, 

    # General properties
    'parallax': 496.,  # +/- 37 mas
    'inclination': 26., # degrees
    #'inclination': 0., # degrees

    'relative_scaling': False, 
    #'K2166_cloudy': {'is_not_in_band': True, 'is_not_in_spot': True}, 
    #'K2166_band':   {'is_in_band': True, 'is_not_in_spot': True}, 
    #'K2166_spot':   {'is_in_spot': True, 'is_not_in_band': True}, 

    #'K2166_cloudy': {'is_not_in_spot': True}, 'K2166_spot':   {'is_in_spot': True}, 
    'do_scat_emis': True, 
}

####################################################################################
#
####################################################################################

sum_m_spec = len(config_data) > 1

scale_flux = True
scale_err  = True
apply_high_pass_filter = True

cloud_kwargs = {
    #'cloud_mode': 'gray', 
    'cloud_mode': 'EddySed', 
    'cloud_species': ['Mg2SiO4(c)_cd', 'MgSiO3(c)_cd', 'Fe(c)_cd'], 

    #'K2166_cloudy': {'cloud_mode': 'gray'}, 
    #'K2166_clear': {'cloud_mode': None}, 
}

rotation_mode = 'integrate' # 'convolve'

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = {
    'chem_mode': 'free', #'SONORAchem' #'eqchem'

    'line_species': [
        'H2O_pokazatel_main_iso', 
        'H2O_181_HotWat78', #'H2O_181', 
        'CH4_hargreaves_main_iso', 
        'CO_main_iso', 
        'CO_36', 
        'CO_28', 
        'NH3_coles_main_iso', 
        'H2S_Sid_main_iso', 
        'HF_main_iso', 
    ], 
}

species_to_plot_VMR = [
    #'H2O', 'CH4', 'NH3', 'H2S', 
    #'H2O', 'H2(18)O', 'CH4', '12CO', '13CO', 'C18O', 'C17O', 'NH3', 'H2S', 'HF', #'Ca', 
    'H2O', 'H2(18)O', 'CH4', '12CO', '13CO', 'C18O', 'NH3', 'H2S', 'HF', 
    #'H2O', 'H2(18)O', 'CH4', 'NH3', 'H2S', #'HF', 
    #'H2O', 'CO', 'CH4'
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
n_iter_before_update = 200
