import numpy as np

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'eqchem_K_B_ret_1'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    K2166_1 = dict(
        instrument='CRIRES',
        
        target_kwargs={
            # Data filenames
            'file':      './data/Luhman_16B_K.dat', 
            'file_wave': './data/Luhman_16_std_K_molecfit_transm.dat', 
            'file_molecfit_transm': './data/Luhman_16B_K_molecfit_transm.dat', 

            # Mask pixels with lower telluric transmission
            'telluric_threshold': 0.8, 

            # Telescope-pointing, used for barycentric velocity-correction
            'ra': 162.297895, 'dec': -53.31703, 'mjd': 59946.32563173, 

            # Flux-calibration filter-name
            #'filter_name': '2MASS/2MASS.J', 'magnitude': 11.40, # Faherty et al. (2014)
            'filter_name': '2MASS/2MASS.Ks', 'magnitude': 9.71, 
        }, 

        std_kwargs={
            # Data filenames
            'file':      './data/Luhman_16_std_K.dat',
            'file_wave': './data/Luhman_16_std_K_molecfit_transm.dat', 
            'file_molecfit_transm':    './data/Luhman_16B_K_molecfit_transm.dat', 
            'file_molecfit_continuum': './data/Luhman_16_std_K_molecfit_continuum.dat',
            'T_BB': 15000., # Blackbody temperature of the standard-star

            # Telescope-pointing, used for barycentric velocity-correction
            'ra': 161.739683, 'dec': -56.75788, 'mjd': 59946.31615474, 
        }, 

        kwargs={
            # Observation info
            'wave_range': (1900, 2500), 'w_set': 'K2166', 
            #'wave_range': (2300, 2400), 'w_set': 'K2166', 
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
    'M_p': ['G', (28.6,0.3), r'$\mathrm{M_p}$'], 
    'R_p': ['G', (1.0,0.1), r'$\mathrm{R_p}$'], 
    'rv':  ['U', (10.,30.), r'$v_\mathrm{rad}$'], 

    # Broadening
    'vsini':        ['U', (10.,30.), r'$v\ \sin\ i$'], 
    'epsilon_limb': ['U', (0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Cloud properties
    'log_opa_base_gray': ['U', (-10,3), r'$\log\ \kappa_{\mathrm{cl,0}}$'], # Cloud slab
    'log_P_base_gray':   ['U', (-0.5,2.5), r'$\log\ P_{\mathrm{cl,0}}$'], 
    'f_sed_gray':        ['U', (1,20), r'$f_\mathrm{sed}$'], 
    'cloud_slope':       ['U', (-6,1), r'$\xi_\mathrm{cl}$'], 

    # Chemistry
    'C/O':               ['U', (0.1,1.0), r'C/O'], 
    'N/O':               ['U', (0.05,0.5), r'N/O'], 
    'Fe/H':              ['U', (-1.0,1.0), r'Fe/H'], 
    'log_Kzz_chem':      ['U', (5,15), r'$\log\ K_\mathrm{zz}$'], 

    'log_13CO_ratio':    ['U', (0,5), r'$\log\ \mathrm{^{12}/^{13}CO}$'], 
    'log_C18O_ratio':    ['U', (0,5), r'$\log\ \mathrm{C^{16}/^{18}O}$'], 
    'log_C17O_ratio':    ['U', (0,5), r'$\log\ \mathrm{C^{16}/^{17}O}$'], 
    'log_H2(18)O_ratio': ['U', (0,5), r'$\log\ \mathrm{H_2^{16}/^{18}O}$'], 
    'log_H2(17)O_ratio': ['U', (0,5), r'$\log\ \mathrm{H_2^{16}/^{17}O}$'], 

    # PT profile
    'dlnT_dlnP_0': ['U', (0.10,0.34), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['U', (0.10,0.34), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['U', (0.05,0.34), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['U', (0.,0.34), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['U', (0.,0.34), r'$\nabla_4$'], 

    'T_phot':         ['U', (900.,1900.), r'$T_\mathrm{phot}$'], 
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
    chem_mode = 'fastchem_table', 
    #path_fastchem_tables='/net/lem/data2/regt/fastchem_tables/', 
    path_fastchem_tables='/projects/0/prjs1096/fastchem_tables/', 
    grid_ranges={
        'P_grid': [10**PT_kwargs['log_P_range'][0], 10**PT_kwargs['log_P_range'][1]], 
        'T_grid': [150,4000], 'CO_grid': [0.1,1.0], #'NO_grid': [0.05,0.5], 
        },
    line_species = [
        '1H2-16O__POKAZATEL', 
        '1H2-18O__HotWat78', 
        '1H2-17O__HotWat78', 

        '12C-16O__HITEMP', 
        '13C-16O__HITEMP', 
        '12C-18O__HITEMP', 
        '12C-17O__HITEMP', 
        
        '12C-1H4__MM', 
        #'13C-1H4__HITRAN', 

        '14N-1H3__CoYuTe', 
        #'15N-1H3__CoYuTe-15', 

        '12C-16O2__AMES', 
        '1H-12C-14N__Harris', 
        '1H-19F__Coxon-Hajig', 
        '1H2-32S__AYT2', 
        '23Na__Kurucz', 
        '39K__Kurucz',
    ], 
)

cloud_kwargs = dict(
    cloud_mode = 'gray', 
    wave_cloud_0 = 2.0, 
)

rotation_kwargs = dict(
    rotation_mode = 'convolve', 
    inclination   = 26, # Degreees
)

pRT_Radtrans_kwargs = dict(
    line_species               = chem_kwargs['line_species'],
    rayleigh_species           = ['H2','He'],
    gas_continuum_contributors = ['H2-H2','H2-He'],
    cloud_species              = cloud_kwargs.get('cloud_species'), 
    
    line_opacity_mode             = 'lbl',
    line_by_line_opacity_sampling = 3, # Faster radiative transfer by down-sampling
    scattering_in_emission        = False, 
    
    pRT_input_data_path = '/projects/0/prjs1096/pRT3/input_data'
)

####################################################################################
# Log-likelihood, covariance keyword-arguments
####################################################################################

loglike_kwargs = dict(
    scale_flux = True, scale_relative_to_chip = 9, 
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
    n_live_points         = 200,
    n_iter_before_update  = 200, 
)