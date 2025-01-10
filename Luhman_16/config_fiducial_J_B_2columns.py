import numpy as np

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'J_B_ret_59_2column_n1000'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    J1226_1 = dict(
        instrument='CRIRES',
        
        target_kwargs={
            # Data filenames
            'file':      './data/Luhman_16B_J_blend_corr.dat', 
            'file_wave': './data/Luhman_16_std_J_molecfit_transm.dat', 
            'file_molecfit_transm': './data/Luhman_16B_J_molecfit_transm.dat', 

            # Mask pixels with lower telluric transmission
            'telluric_threshold': 0.7, 

            # Telescope-pointing, used for barycentric velocity-correction
            'ra': 162.299451, 'dec': -53.31767, 'mjd': 59946.35286502, 

            # Flux-calibration filter-name
            'filter_name': '2MASS/2MASS.J', 'magnitude': 11.22, # Burgasser et al. (2013)
        }, 

        std_kwargs={
            # Data filenames
            'file':      './data/Luhman_16_std_J.dat',
            'file_wave': './data/Luhman_16_std_J_molecfit_transm.dat', 
            'file_molecfit_transm':    './data/Luhman_16_std_J_molecfit_transm.dat', 
            'file_molecfit_continuum': './data/Luhman_16_std_J_molecfit_continuum.dat',
            'T_BB': 15000., # Blackbody temperature of the standard-star

            # Telescope-pointing, used for barycentric velocity-correction
            'ra': 161.738984, 'dec': -56.75771, 'mjd': 59946.3601578, 
        }, 

        kwargs={
            # Observation info
            'wave_range': (1115, 1338), 'w_set': 'J1226',  
            #'wave_range': (1240, 1267), 'w_set': 'J1226',  
            'slit': 'w_0.4', 'resolution': 65000,

            # Outlier clipping
            'sigma_clip_width': 5, 'sigma_clip_sigma': 5, 
        },
    ),
)

# Add the second column as another model setting
config_data['J1226_2'] = config_data['J1226_1'].copy()

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Covariance parameters
    'log_a': ['U', (-0.7,0.3), r'$\log\ a$'], 
    'log_l': ['U', (-3.0,-1.0), r'$\log\ l$'], 

    # General properties
    'M_p': ['G', (29.4,0.2), r'$\mathrm{M_p}$'], # (Bedin et al. 2024)
    'R_p': ['G', (1.0,0.1), r'$\mathrm{R_p}$'], 
    'rv':  ['U', (10.,30.), r'$v_\mathrm{rad}$'], 

    # Broadening
    'vsini':        ['U', (10.,30.), r'$v\ \sin\ i$'], 
    'epsilon_limb': ['U', (0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Chemistry
    'log_H2O':   ['U', (-14,-2), r'$\log\ \mathrm{H_2O}$'],
    'log_HF':    ['U', (-14,-2), r'$\log\ \mathrm{HF}$'],
    'log_K':     ['U', (-14,-2), r'$\log\ \mathrm{K}$'],
    'log_Na':    ['U', (-14,-2), r'$\log\ \mathrm{Na}$'],

    # PT profile
    'dlnT_dlnP_0': ['U', (0.10,0.34), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['U', (0.10,0.34), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['U', (0.05,0.34), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['U', (0.,0.34), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['U', (0.,0.34), r'$\nabla_4$'], 

    'T_phot':         ['U', (1200.,2200.), r'$T_\mathrm{phot}$'], 
    'log_P_phot':     ['U', (-1.,1.), r'$\log\ P_\mathrm{phot}$'], 
    'd_log_P_phot+1': ['U', (0.5,3.), r'$\Delta P_\mathrm{+1}$'], 
    'd_log_P_phot-1': ['U', (0.5,2.), r'$\Delta P_\mathrm{-1}$'], 

    'coverage_fraction': ['U', (0,1), 'cf'],
    'J1226_1': {
        # Cloud properties
        'log_opa_base_gray': ['U', (-10,3), r'$\log\ \kappa_{\mathrm{cl,0}}$'], # Cloud slab
        'log_P_base_gray':   ['U', (-0.5,2.5), r'$\log\ P_{\mathrm{cl,0}}$'], 
        'f_sed_gray':        ['U', (1,20), r'$f_\mathrm{sed}$'], 
        # Chemistry
        'log_FeH':   ['U', (-14,-2), r'$\log\ \mathrm{FeH}$'], 
        'log_FeH_P': ['U', (-5,3), r'$\log\ P_\mathrm{FeH}$'], 
        'FeH_alpha': ['U', (0,20), r'$\alpha_\mathrm{FeH}$'], 
    }, 
    'J1226_2': {
        # Cloud properties
        'log_opa_base_gray': ['U', (-10,3), r'$\log\ \kappa_{\mathrm{cl,0}}$'], # Cloud slab
        'log_P_base_gray':   ['U', (-0.5,2.5), r'$\log\ P_{\mathrm{cl,0}}$'], 
        'f_sed_gray':        ['U', (1,20), r'$f_\mathrm{sed}$'], 
        # Chemistry
        'log_FeH':   ['U', (-14,-2), r'$\log\ \mathrm{FeH}$'], 
        'log_FeH_P': ['U', (-5,3), r'$\log\ P_\mathrm{FeH}$'], 
        'FeH_alpha': ['U', (0,20), r'$\alpha_\mathrm{FeH}$'], 
    }, 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    #'parallax': 496,  # +/- 37 mas
    'distance': 1.9960, # pc +/- 50 AU (Bedin et al. 2024)

    # Custom line opacity
    'A_w_0_H2': 0.352609, 'b_w_0_H2': 0.385961, 
    'A_w_1_H2': 0.245926, 'b_w_1_H2': 0.447971, 
    'A_d_0_H2': 0.00158988, 'b_d_0_H2': 0.949254, 
    'A_d_1_H2': 0.00211668, 'b_d_1_H2': 0.933563, 

    'A_w_0_He': 0.208190, 'b_w_0_He': 0.452833, 
    'A_w_1_He': 0.121448, 'b_w_1_He': 0.531718, 
    'A_d_0_He': 0.001943820, 'b_d_0_He': 0.89691, 
    'A_d_1_He': 0.000462539, 'b_d_1_He': 1.07284, 
}

####################################################################################
# Physical model keyword-arguments
####################################################################################

PT_kwargs = dict(
    shared_between_m_set = True, 

    PT_mode   = 'free_gradient', 
    n_T_knots = 5, 
    interp_mode = 'linear', 

    log_P_range = (-5.,3.), 
    n_atm_layers = 50,
)

chem_kwargs = dict(
    shared_between_m_set = False, 

    chem_mode = 'free', 
    line_species = [
        '1H2-16O__POKAZATEL', 
        '1H-19F__Coxon-Hajig', 
        '23Na__Kurucz', 
        '39K__Kurucz',
        '56Fe-1H__MoLLIST',
    ], 
)

cloud_kwargs = dict(
    shared_between_m_set = False, 

    cloud_mode = 'gray', 
)

rotation_kwargs = dict(
    shared_between_m_set = True, 

    rotation_mode = 'convolve', 
    inclination   = 26, # Degreees
)

#custom_opacity_path = '/home/sdregt/retrieval_base/Luhman_16/custom_opacity_data/'
custom_opacity_path = '/net/lem/data1/regt/retrieval_base/Luhman_16/custom_opacity_data/'
line_opacity_kwargs = {
    'shared_between_m_set': True,

    '39K__Kurucz': {
        'states_file': f'{custom_opacity_path}/K_I_states.txt',
        'transitions_file': f'{custom_opacity_path}/K_I_transitions_Kurucz.txt', 

        'mass': 39.0983, 'is_alkali': True, 'E_ion': 35009.8140, 
        'custom_transitions': [{'nu_0': 7983.655}, {'nu_0': 8041.365}], 'n_ref': 1e20,
        'line_cutoff': 1000, 'log_gf_cutoff': -2.5, 'log_gf_cutoff_exact': -0.5,
    }, 
    '23Na__Kurucz': {
        'states_file': f'{custom_opacity_path}/Na_I_states.txt',
        'transitions_file': f'{custom_opacity_path}/Na_I_transitions_Kurucz.txt', 

        'mass': 22.989769, 'is_alkali': True, 'E_ion': 41449.451,
        'line_cutoff': 1000, 'log_gf_cutoff': -2.5, 'log_gf_cutoff_exact': -0.5,
    }
}

pRT_Radtrans_kwargs = dict(
    #line_species               = chem_kwargs['line_species'],
    line_species               = ['1H2-16O__POKAZATEL', '1H-19F__Coxon-Hajig', '56Fe-1H__MoLLIST'], 
    rayleigh_species           = ['H2','He'],
    gas_continuum_contributors = ['H2-H2','H2-He'],
    cloud_species              = cloud_kwargs.get('cloud_species'), 
    
    line_opacity_mode             = 'lbl',
    line_by_line_opacity_sampling = 3, # Faster radiative transfer by down-sampling
    scattering_in_emission        = False, 
    
    #pRT_input_data_path = '/projects/0/prjs1096/pRT3/input_data', 
    pRT_input_data_path = '/net/schenk/data2/regt/pRT3_input_data/input_data', 

    shared_line_opacities = True, # Share line opacities between same PT profiles
)

####################################################################################
# Log-likelihood, covariance keyword-arguments
####################################################################################

loglike_kwargs = dict(
    scale_flux = True, scale_relative_to_chip = 19, 
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
    line_opacity_kwargs=line_opacity_kwargs, 
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
    n_live_points         = 1000,
    n_iter_before_update  = 400, 
)