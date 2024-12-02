import numpy as np
import os

file_params = 'config_fiducial_K_A_new.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'test'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    K2166_B = dict(
        target_kwargs={
            # Data filenames
            'file':      './data/Luhman_16A_K.dat', 
            'file_wave': './data/Luhman_16_std_K_molecfit_transm.dat', 
            'file_molecfit_transm': './data/Luhman_16A_K_molecfit_transm.dat', 

            # Mask pixels with lower telluric transmission
            'telluric_threshold': 0.8, 

            # Telescope-pointing, used for barycentric velocity-correction
            'ra': 162.297895, 'dec': -53.31703, 'mjd': 59946.32563173, 

            # Flux-calibration filter-name
            #'filter_name': '2MASS/2MASS.J', 'magnitude': 11.68, # Faherty et al. (2014)
            'filter_name': '2MASS/2MASS.Ks', 'magnitude': 9.46, 
        }, 

        std_kwargs={
            # Data filenames
            'file':      './data/Luhman_16_std_K.dat',
            'file_wave': './data/Luhman_16_std_K_molecfit_transm.dat', 
            'file_molecfit_transm':    './data/Luhman_16A_K_molecfit_transm.dat', 
            'file_molecfit_continuum': './data/Luhman_16_std_K_molecfit_continuum.dat',
            'T_BB': 15000., # Blackbody temperature of the standard-star

            # Telescope-pointing, used for barycentric velocity-correction
            'ra': 161.739683, 'dec': -56.75788, 'mjd': 59946.31615474, 
        }, 

        kwargs={
            # Observation info
            #'wave_range': (1900, 2500), 'w_set': 'K2166', 
            'wave_range': (2300, 2450), 'w_set': 'K2166', 
            'slit': 'w_0.4', 'resolution': 60000,

            # Outlier clipping
            'sigma_clip_width': 5, 'sigma_clip_sigma': 5, 
        },
    )
)

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Covariance parameters
    'log_a': ['U', (-0.7,0.3), r'$\log\ a$'], 
    'log_l': ['U', (-3.0,-1.0), r'$\log\ l$'], 

    # General properties
    'M_p': ['G', (33.5,0.3), r'$\mathrm{M_p}$'], 
    'R_p': ['G', (1.0,0.1), r'$\mathrm{R_p}$'], 
    'rv':  ['U', (10.,30.), r'$v_\mathrm{rad}$'], 

    # Broadening
    'vsini':        ['U', (10.,30.), r'$v\ \sin\ i$'], 
    'epsilon_limb': ['U', (0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Cloud properties
    'log_opa_base_gray_0': ['U', (-10,3), r'$\log\ \kappa_{\mathrm{cl,0,1}}$'], # Cloud slab
    'log_P_base_gray_0':   ['U', (-0.5,2.5), r'$\log\ P_{\mathrm{cl,0,1}}$'], 
    'f_sed_gray_0':        ['U', (1,20), r'$f_\mathrm{sed,1}$'], 
    'cloud_slope_0':       ['U', (-6,1), r'$\xi_\mathrm{cl,1}$'], 

    # Chemistry
    'C/O':  ['U', (0.1,1.0), r'C/O'], 
    'Fe/H': ['U', (-1.0,1.0), r'Fe/H'], 

    'log_13CO_ratio':    ['U', (0,5), r'$\log\ \mathrm{^{12}/^{13}C}$'], 
    'log_C18O_ratio':    ['U', (0,5), r'$\mathrm{C^{18}/^{16}O}$'], 
    'log_H2(18)O_ratio': ['U', (0,5), r'$\mathrm{H_2^{18}/^{16}O}$'], 
    'log_Kzz_chem':      ['U', (5,15), r'$\log\ K_\mathrm{zz}$'], 

    # PT profile    
    'dlnT_dlnP_0': ['U', (0.10,0.34), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['U', (0.10,0.34), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['U', (0.05,0.34), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['U', (0.,0.34), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['U', (0.,0.34), r'$\nabla_4$'], 

    'T_phot':         ['U', (900.,1900.), r'$T_\mathrm{phot}$'], 
    'log_P_phot':     ['U', (-1.,1.), r'$\log\ P_\mathrm{phot}$'], 
    'd_log_P_phot+1': ['U', (0.5,2.5), r'$\Delta\ P_\mathrm{+1}$'], 
    'd_log_P_phot-1': ['U', (0.5,2.), r'$\Delta\ P_\mathrm{-1}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # Define pRT's log-pressure grid
    'log_P_range': (-5,3), 
    'n_atm_layers': 50, 

    # Down-sample opacities for faster radiative transfer
    'lbl_opacity_sampling': 3, 

    # Data resolution
    #'res': 60000, 

    # General properties
    'parallax': 496,  # +/- 37 mas
    'inclination': 18, # degrees

    'do_scat_emis': False, 

    #'R_p': 1.0, 
}

####################################################################################
#
####################################################################################

sum_m_spec = len(config_data) > 1

scale_flux = True; scale_rel_to_ij = (3,0)
#scale_flux = False
scale_err  = True
apply_high_pass_filter = False

cloud_kwargs = {
    'cloud_mode': 'gray', 
    'wave_cloud_0': 2.0, 
}

rotation_mode = 'convolve' # 'integrate'

####################################################################################
# Chemistry parameters
####################################################################################

chem_kwargs = {
    #'chem_mode': 'free', 
    'chem_mode': 'fastchem_table', 'path_fastchem_tables': '/net/lem/data2/regt/fastchem_tables/', 
    #'chem_mode': 'pRT_table', 
    #'chem_mode': 'fastchem', 
    #'abundance_file': '/net/lem/data1/regt/fastchem/input/element_abundances/asplund_2020_extended.dat', 
    #'gas_data_file': '/net/lem/data1/regt/fastchem/input/logK/logK_extended.dat', 
    #'cond_data_file': '/net/lem/data1/regt/fastchem/input/logK/logK_condensates.dat', 

    'line_species': [
        'H2O_pokazatel_main_iso_Sam_new', 
        'H2O_181_HotWat78', 

        'CO_high_Sam', 
        'CO_36_high_Sam', 
        'CO_28_high_Sam', 

        'CH4_MM_main_iso', #'CH4_hargreaves_main_iso_Sam', 
        'NH3_coles_main_iso_Sam', 
        'H2S_Sid_main_iso', 
        'HF_main_iso_new', 
    ], 
}

species_to_plot_VMR = [
    'H2O', 'H2(18)O', '12CO', '13CO', 'C18O', 'CH4', 'NH3', 'H2S', 'HF', 
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
        cov_kwargs['trunc_dist'] * 10**np.max(free_params['log_l'][1])

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
n_iter_before_update = 400
