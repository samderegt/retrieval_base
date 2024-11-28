import numpy as np
import os

file_params = 'config_fiducial_K_B.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'new_fiducial_K_B_ret_7'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    K2166_B = {
        'w_set': 'K2166', 'wave_range': (2300, 2400), 

        # Data filenames
        'file_target': './data/Luhman_16B_K.dat', 
        'file_std': './data/Luhman_16_std_K.dat', 
        #'file_wave': './data/Luhman_16_std_K.dat', # Use std-observation wlen-solution
        'file_wave': './data/Luhman_16_std_K_molecfit_transm.dat', # Use molecfit wlen-solution

        # Telluric transmission
        'file_molecfit_transm': './data/Luhman_16B_K_molecfit_transm.dat', 
        'file_std_molecfit_transm': './data/Luhman_16_std_K_molecfit_transm.dat', 
        'file_std_molecfit_continuum': './data/Luhman_16_std_K_molecfit_continuum.dat', 

        'filter_2MASS': '2MASS/2MASS.Ks', # Magnitude used for flux-calibration
        #'filter_2MASS': 'MKO/NSFCam.K', # Magnitude used for flux-calibration
        'pwv': 1.5, # Precipitable water vapour

        # Telescope-pointing, used for barycentric velocity-correction
        'ra': 162.297895, 'dec': -53.31703, 'mjd': 59946.32563173, 
        'ra_std': 161.739683, 'dec_std': -56.75788, 'mjd_std': 59946.31615474, 

        # Some info on standard-observation
        'T_std': 15000, 

        # Slit-width, sets model resolution
        'slit': 'w_0.4', 

        # Ignore pixels with lower telluric transmission
        'tell_threshold': 0.8, 
        'sigma_clip_width': 5, # Remove outliers
        }, 
)

magnitudes = {
    #'2MASS/2MASS.J': (11.22, 0.04), # Burgasser et al. (2013)
    #'2MASS/2MASS.Ks': (9.73, 0.09), 
    #'MKO/NSFCam.J': (11.22, 0.04), # Burgasser et al. (2013)
    #'MKO/NSFCam.K': (9.73, 0.09), 
    '2MASS/2MASS.J': (11.40, 0.05), # Faherty et al. (2014)
    '2MASS/2MASS.Ks': (9.71, 0.10), 
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
    #'log_g': [(3.5,6.0), r'$\log\ g$'], 
    'gaussian_M_p': [(28.6,0.3), r'$\mathrm{M_p}$'], 
    'gaussian_R_p': [(1.0,0.1), r'$\mathrm{R_p}$'], 
    'rv': [(10.,30.), r'$v_\mathrm{rad}$'], 

    # Broadening
    'vsini':        [(10.,30.), r'$v\ \sin\ i$'], 
    'epsilon_limb': [(0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Cloud properties
    #'log_opa_base_gray_0': [(-10,3), r'$\log\ \kappa_{\mathrm{cl,0,1}}$'], # Cloud slab
    #'log_P_base_gray_0':   [(-0.5,2.5), r'$\log\ P_{\mathrm{cl,0,1}}$'], 
    #'f_sed_gray_0':        [(1,20), r'$f_\mathrm{sed,1}$'], 
    #'cloud_slope_0':       [(-6,1), r'$\xi_\mathrm{cl,1}$'], 

    #'log_opa_base_gray_1': [(-10,3), r'$\log\ \kappa_{\mathrm{cl,0,2}}$'], # Cloud slab
    #'log_P_base_gray_1':   [(-0.5,2.5), r'$\log\ P_{\mathrm{cl,0,2}}$'], 
    #'f_sed_gray_1':        [(1,20), r'$f_\mathrm{sed,2}$'], 
    #'cloud_slope_1':       [(-6,1), r'$\xi_\mathrm{cl,2}$'], 

    # Chemistry
#    'log_H2O':     [(-14,-2), r'$\log\ \mathrm{H_2O}$'], 
#    'log_H2(18)O': [(-14,-2), r'$\log\ \mathrm{H_2^{18}O}$'], 

#    'log_12CO':    [(-14,-2), r'$\log\ \mathrm{^{12}CO}$'], 
#    'log_13CO':    [(-14,-2), r'$\log\ \mathrm{^{13}CO}$'], 
#    'log_C18O':    [(-14,-2), r'$\log\ \mathrm{C^{18}O}$'], 

#    'log_CH4':     [(-14,-2), r'$\log\ \mathrm{CH_4}$'], 
#    'log_NH3':     [(-14,-2), r'$\log\ \mathrm{NH_3}$'], 
#    'log_H2S':     [(-14,-2), r'$\log\ \mathrm{H_2S}$'], 
#    'log_HF':      [(-14,-2), r'$\log\ \mathrm{HF}$'], 

    'C/O': [(0.1,1.0), r'C/O'], 
    'N/O': [(0.05,0.5), r'N/O'], 
    'Fe/H': [(-1.0,1.0), r'Fe/H'], 
    'log_13CO_ratio': [(0,5), r'$\log\ \mathrm{^{12}/^{13}C}$'], 

    #'log_C18O_ratio': [(0,5), r'$\mathrm{C^{18}/^{16}O}$'], 
    #'log_H2(18)O_ratio': [(0,5), r'$\mathrm{H_2^{18}/^{16}O}$'], 

    #'log_13CH4':   [(-14,-2), r'$\log\ \mathrm{^{13}CH_4}$'], 
    #'log_15NH3':   [(-14,-2), r'$\log\ \mathrm{^{15}NH_3}$'], 

    'log_Kzz_chem': [(5,15), r'$\log\ K_\mathrm{zz}$'], 

    # PT profile    
    'dlnT_dlnP_0': [(0.10,0.34), r'$\nabla_0$'], 
    'dlnT_dlnP_1': [(0.10,0.34), r'$\nabla_1$'], 
    'dlnT_dlnP_2': [(0.05,0.34), r'$\nabla_2$'], 
    'dlnT_dlnP_3': [(0.,0.34), r'$\nabla_3$'], 
    'dlnT_dlnP_4': [(0.,0.34), r'$\nabla_4$'], 

    'T_phot':         [(900.,1900.), r'$T_\mathrm{phot}$'], 
    'log_P_phot':     [(-1.,1.), r'$\log\ P_\mathrm{phot}$'], 
    'd_log_P_phot+1': [(0.5,2.5), r'$\Delta\ P_\mathrm{+1}$'], 
    'd_log_P_phot-1': [(0.5,2.), r'$\Delta\ P_\mathrm{-1}$'], 
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
    'parallax': 496,  # +/- 37 mas
    'inclination': 26, # degrees

    'do_scat_emis': False, 

    #'R_p': 1.0, 
}

####################################################################################
#
####################################################################################

sum_m_spec = len(config_data) > 1

scale_flux = True
#scale_rel_to_ij = (3,0)
scale_rel_to_ij = (0,0)
#scale_flux = False
scale_err  = True
apply_high_pass_filter = False

cloud_kwargs = {
    #'cloud_mode': 'gray', 
    'cloud_mode': None, 
    'wave_cloud_0': 2.0, 
}

rotation_mode = 'convolve' # 'integrate'

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = {
    #'chem_mode': 'free', #'SONORAchem' #'eqchem'
    #'chem_mode': 'fastchem_table', 'path_fastchem_tables': '/net/lem/data2/regt/fastchem_tables/', 
    #'chem_mode': 'pRT_table', 
    'chem_mode': 'fastchem', 
    'abundance_file': '/net/lem/data1/regt/fastchem/input/element_abundances/asplund_2020_extended.dat', 
    'gas_data_file': '/net/lem/data1/regt/fastchem/input/logK/logK_extended.dat', 
    'cond_data_file': '/net/lem/data1/regt/fastchem/input/logK/logK_condensates.dat', 

    'line_species': [
        'H2O_pokazatel_main_iso_Sam_new', 
        #'H2O_181_HotWat78', 

        'CO_high_Sam', 
        'CO_36_high_Sam', 
        #'CO_28_high_Sam', 

        'CH4_MM_main_iso', #'CH4_hargreaves_main_iso_Sam', 
        'NH3_coles_main_iso_Sam', 
        #'H2S_Sid_main_iso', 
        'HF_main_iso_new', 
    ], 
}

species_to_plot_VMR = [
    #'H2O', 'H2(18)O', '12CO', '13CO', 'C18O', 'CH4', 'NH3', 'H2S', 'HF', 
    'H2O', '12CO', 'CH4', 'NH3', 'HF', 
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
n_live_points = 30
n_iter_before_update = 5
