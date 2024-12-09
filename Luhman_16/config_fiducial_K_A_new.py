import numpy as np

file_params = 'config_fiducial_K_A_new.py'

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'test2'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    K2166_1 = dict(
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
            #'wave_range': (2300, 2338), 'w_set': 'K2166', 
            'wave_range': (2300, 2400), 'w_set': 'K2166', 
            #'wave_range': (2300, 2500), 'w_set': 'K2166', 
            'slit': 'w_0.4', 'resolution': 60000,

            # Outlier clipping
            'sigma_clip_width': 5, 'sigma_clip_sigma': 5, 
        },
    )
)
#import copy
#config_data['K2166_2'] = copy.deepcopy(config_data['K2166_1'])
#config_data['K2166_2']['kwargs']['wave_range'] = (2200, 2300)

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

    'rv':     ['U', (10.,30.), r'$v_\mathrm{rad}$'], 
    'T_phot': ['U', (900.,1900.), r'$T_\mathrm{phot}$'], 

    # Broadening
    'vsini':        ['U', (10.,30.), r'$v\ \sin\ i$'], 
    'epsilon_limb': ['U', (0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Cloud properties
    #'log_opa_base_gray': ['U', (-10,3), r'$\log\ \kappa_{\mathrm{cl,0}}$'], # Cloud slab
    #'log_P_base_gray':   ['U', (-0.5,2.5), r'$\log\ P_{\mathrm{cl,0}}$'], 
    #'f_sed_gray':        ['U', (1,20), r'$f_\mathrm{sed}$'], 
    #'cloud_slope':       ['U', (-6,1), r'$\xi_\mathrm{cl}$'], 

    # Chemistry
    'C/O':            ['U', (0.1,1.0), r'C/O'], 
    'Fe/H':           ['U', (-1.0,1.0), r'Fe/H'], 
    #'log_13CO_ratio': ['U', (0,5), r'$\log\ \mathrm{^{12}/^{13}C}$'], 
    'log_Kzz_chem':   ['U', (5,15), r'$\log\ K_\mathrm{zz}$'], 

    # PT profile
    'dlnT_dlnP_0': ['U', (0.10,0.34), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['U', (0.10,0.34), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['U', (0.05,0.34), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['U', (0.,0.34), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['U', (0.,0.34), r'$\nabla_4$'], 

    #'T_phot':         ['U', (900.,1900.), r'$T_\mathrm{phot}$'], 
    'log_P_phot':     ['U', (-1.,1.), r'$\log\ P_\mathrm{phot}$'], 
    'd_log_P_phot+1': ['U', (0.5,2.5), r'$\Delta P_\mathrm{+1}$'], 
    'd_log_P_phot-1': ['U', (0.5,2.), r'$\Delta P_\mathrm{-1}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 496,  # +/- 37 mas
}

####################################################################################
#
####################################################################################

#apply_high_pass_filter = False

####################################################################################
# Physical model keyword-arguments
####################################################################################

PT_kwargs = dict(
    PT_mode   = 'free_gradient', 
    n_T_knots = 5, 
    interp_mode = 'linear', 

    log_P_range = (-5.,3.), 
    n_atm_layers = 50,
)

chem_kwargs = dict(
    chem_mode = 'fastchem_table', path_fastchem_tables='/net/lem/data2/regt/fastchem_tables/', 

    line_species = [
        'H2O_pokazatel_main_iso_Sam_new', 
        'CO_high_Sam', 
        #'CO_36_high_Sam',

        'CH4_MM_main_iso', 
        'NH3_coles_main_iso_Sam', 
        'H2S_Sid_main_iso', 
        'HF_main_iso_new', 
    ], 
)

cloud_kwargs = dict(
    cloud_mode = None, 
    #cloud_mode = 'gray', 
    wave_cloud_0 = 2.0, 
)

rotation_kwargs = dict(
    rotation_mode = 'convolve', 
    inclination   = 18, # Degreees
)

pRT_Radtrans_kwargs = dict(
    line_species        = chem_kwargs['line_species'],
    rayleigh_species    = ['H2','He'],
    continuum_opacities = ['H2-H2','H2-He'],
    cloud_species       = cloud_kwargs.get('cloud_species'), 
    
    mode                 = 'lbl',
    lbl_opacity_sampling = 3, # Faster radiative transfer by down-sampling
    do_scat_emis         = False, 
)

line_opacity_kwargs = dict(
    states_file = '/net/lem/data1/regt/retrieval_base/retrieval_base/custom_opacity_data/K_I_states.txt', 
    transitions_file = '/net/lem/data1/regt/retrieval_base/retrieval_base/custom_opacity_data/K_I_transitions_Kurucz.txt',

    custom_transitions = [
        {'nu_0':4310.3, 'log_gf':-0.063, 'E_low':12985.186, 'log_gamma_N':7.83, 'log_gamma_vdW':-7.46}, 
        {'nu_0':7983.655}, 
        {'nu_0':8041.365}, 
        ], 
    log_gf_cutoff = -2., 
    line_cutoff = 1000, 
    log_gf_cutoff_exact = -0.5, 
    
    is_alkali = True, 
    mass = 39.0983, E_ion = 35009.8140, 
    line_species = 'K_wo_J_doublets', 
)

####################################################################################
# Log-likelihood, covariance keyword-arguments
####################################################################################

loglike_kwargs = dict(
    scale_flux = True, #scale_relative_to_chip = 9, 
    scale_err = True, 
    sum_model_settings = True, 
)

cov_kwargs = dict(
    trunc_dist = 3, 
    scale_amp  = True, 
    max_wave_sep = 3 * 10**free_params.get('log_l', [None,[None,np.inf]])[1][1], 
)

all_model_kwargs = dict(
    PT_kwargs=PT_kwargs, 
    chem_kwargs=chem_kwargs, 
    cloud_kwargs=cloud_kwargs, 
    rotation_kwargs=rotation_kwargs,
    pRT_Radtrans_kwargs=pRT_Radtrans_kwargs, 
    line_opacity_kwargs=line_opacity_kwargs,

    cov_kwargs=cov_kwargs, 
    loglike_kwargs=loglike_kwargs,
)

####################################################################################
# Multinest parameters
####################################################################################

pymultinest_kwargs = dict(    
    verbose = True, 

    const_efficiency_mode = True, 
    sampling_efficiency   = 0.05, 
    evidence_tolerance    = 0.5, 
    n_live_points         = 50, 
    n_iter_before_update  = 200, 
)