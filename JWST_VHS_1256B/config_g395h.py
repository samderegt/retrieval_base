import numpy as np

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'g395h_ret_2'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    #nirspec_g140h_1 = dict(
    #    instrument='JWST', kwargs={'file':'./data/nirspec_140h_1.dat', 'wave_range':(900,1900), 'grating':'G140H', 'n_chunks':4}
    #), 
    #nirspec_g140h_2 = dict(
    #    instrument='JWST', kwargs={'file':'./data/nirspec_140h_2.dat', 'wave_range':(900,1900), 'grating':'G140H', 'n_chunks':4}
    #), 
    #nirspec_g235h_1 = dict(
    #    instrument='JWST', kwargs={'file':'./data/nirspec_235h_1.dat', 'wave_range':(1600,3200), 'grating':'G235H', 'n_chunks':4}
    #),
    #nirspec_g235h_2 = dict(
    #    instrument='JWST', kwargs={'file':'./data/nirspec_235h_2.dat', 'wave_range':(1600,3200), 'grating':'G235H', 'n_chunks':4}
    #),
    nirspec_g395h_1 = dict(
        instrument='JWST', kwargs={'file':'./data/nirspec_395h_1.dat', 'wave_range':(2800,5400), 'grating':'G395H', 'n_chunks':4, 'min_SNR':3}
    ),
    nirspec_g395h_2 = dict(
        instrument='JWST', kwargs={'file':'./data/nirspec_395h_2.dat', 'wave_range':(2800,5400), 'grating':'G395H', 'n_chunks':4, 'min_SNR':3}
    ),
)

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Covariance parameters
    #'log_a': ['U', (-0.7,0.3), r'$\log\ a$'], 
    #'log_l': ['U', (-3.0,-1.0), r'$\log\ l$'], 

    # General properties
    'M_p': ['G', (19.0,5.0), r'$\mathrm{M_p}$'], 
    'R_p': ['G', (1.1,0.2), r'$\mathrm{R_p}$'], 
    'rv':  ['U', (-10.,7.), r'$v_\mathrm{rad}$'], 

    # Broadening
    'vsini':        ['U', (0.,20.), r'$v\ \sin\ i$'], 
    #'epsilon_limb': ['U', (0,1), r'$\epsilon_\mathrm{limb}$'], 

    # Cloud properties
    'log_opa_base_gray': ['U', (-10,3), r'$\log\ \kappa_{\mathrm{cl,0}}$'], # Cloud slab
    'log_P_base_gray':   ['U', (-0.5,2.5), r'$\log\ P_{\mathrm{cl,0}}$'], 
    'f_sed_gray':        ['U', (1,20), r'$f_\mathrm{sed}$'], 
    'cloud_slope':       ['U', (-6,1), r'$\xi_\mathrm{cl}$'], 

    # Chemistry
    'log_H2O':     ['U', (-14,-2), r'$\log\ \mathrm{H_2O}$'],
    'log_H2(18)O': ['U', (-14,-2), r'$\log\ \mathrm{H_2^{18}O}$'],
    #'log_H2(17)O': ['U', (-14,-2), r'$\log\ \mathrm{H_2^{17}O}$'],

    'log_12CO':    ['U', (-14,-2), r'$\log\ \mathrm{^{12}CO}$'],
    'log_13CO':    ['U', (-14,-2), r'$\log\ \mathrm{^{13}CO}$'],
    'log_C18O':    ['U', (-14,-2), r'$\log\ \mathrm{C^{18}O}$'],
    'log_C17O':    ['U', (-14,-2), r'$\log\ \mathrm{C^{17}O}$'],

    'log_CO2':     ['U', (-14,-2), r'$\log\ \mathrm{CO_2}$'],
    'log_13CO2':   ['U', (-14,-2), r'$\log\ \mathrm{^{13}CO_2}$'],
    #'log_CO(18)O': ['U', (-14,-2), r'$\log\ \mathrm{CO^{18}O}$'],
    #'log_CO(17)O': ['U', (-14,-2), r'$\log\ \mathrm{CO^{17}O}$'],

    'log_CH4':     ['U', (-14,-2), r'$\log\ \mathrm{CH_4}$'],
    'log_13CH4':   ['U', (-14,-2), r'$\log\ \mathrm{^{13}CH_4}$'],

    'log_NH3':     ['U', (-14,-2), r'$\log\ \mathrm{NH_3}$'],
    'log_HCN':     ['U', (-14,-2), r'$\log\ \mathrm{HCN}$'],
    #'log_HF':      ['U', (-14,-2), r'$\log\ \mathrm{HF}$'],
    #'log_HCl':     ['U', (-14,-2), r'$\log\ \mathrm{HCl}$'],
    'log_H2S':     ['U', (-14,-2), r'$\log\ \mathrm{H_2S}$'],
    
    #'log_K':       ['U', (-14,-2), r'$\log\ \mathrm{K}$'],
    #'log_Na':      ['U', (-14,-2), r'$\log\ \mathrm{Na}$'],
    #'log_Ti':      ['U', (-14,-2), r'$\log\ \mathrm{Ti}$'],
    #'log_Fe':      ['U', (-14,-2), r'$\log\ \mathrm{Fe}$'],

    # PT profile
    'dlnT_dlnP_0': ['U', (0.1,0.34), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['U', (0.1,0.34), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['U', (0.05,0.34), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['U', (0.,0.34), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['U', (-0.1,0.34), r'$\nabla_4$'], 
    #'dlnT_dlnP_5': ['U', (0.,0.34), r'$\nabla_5$'], 

    'T_phot':         ['U', (500.,2500.), r'$T_\mathrm{phot}$'], 
    'log_P_phot':     ['U', (-2.,1.5), r'$\log\ P_\mathrm{phot}$'], 
    'd_log_P_phot+1': ['U', (0.5,2.5), r'$\Delta P_\mathrm{+1}$'], 
    #'d_log_P_phot+2': ['U', (0.5,3.), r'$\Delta P_\mathrm{+2}$'], 
    'd_log_P_phot-1': ['U', (0.5,1.5), r'$\Delta P_\mathrm{-1}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 47.2733,  # +/- 37 mas

    'epsilon_limb': 0.6, # Limb-darkening
}

####################################################################################
#
####################################################################################

#apply_high_pass_filter = False

####################################################################################
# Physical model keyword-arguments
####################################################################################

PT_kwargs = dict(
    shared_between_m_set = True,

    PT_mode = 'free_gradient', 
    n_knots = 5, 
    interp_mode = 'linear', 

    log_P_range = (-5.,2.), 
    n_atm_layers = 50,
)

chem_kwargs = dict(
    shared_between_m_set = True,

    chem_mode = 'free', 
    #chem_mode = 'fastchem_table', 
    ##path_fastchem_tables='/net/lem/data2/regt/fastchem_tables/', 
    #path_fastchem_tables='/net/lem/data2/regt/fastchem_interpolation/fastchem_tables_NO_0.05_0.20/', 
    #grid_ranges={
    #    'P_grid': [10**PT_kwargs['log_P_range'][0], 10**PT_kwargs['log_P_range'][1]], 
    #    'T_grid': [150,4000], 'CO_grid': [0.1,1.0], 'NO_grid': [0.05,0.2], 
    #    },
    line_species = [
        '1H2-16O__POKAZATEL', 
        '1H2-18O__HotWat78', 
        #'1H2-17O__HotWat78', 

        '12C-16O__HITEMP', 
        '13C-16O__HITEMP', 
        '12C-18O__HITEMP', 
        '12C-17O__HITEMP', 

        '12C-1H4__MM', 
        #'13C-1H4__HITRAN', 

        '12C-16O2__HITEMP', 
        '13C-16O2__HITEMP', 
        #'12C-16O-18O__HITEMP', 
        #'12C-16O-17O__HITEMP', 

        '14N-1H3__CoYuTe', 
        '1H-12C-14N__Harris', 
        #'1H-19F__Coxon-Hajig', 
        #'1H-35Cl__HITRAN-HCl', 
        '1H2-32S__AYT2', 
        ##'12C2-1H2__aCeTY', 
        
        #'39K__Kurucz', 
        #'23Na__Kurucz', 
        #'48Ti__Kurucz', 
        #'56Fe__Kurucz',
        #'1H2__RACPPK', 
    ], 
)

cloud_kwargs = dict(
    shared_between_m_set = True,

    #cloud_mode = None, 
    cloud_mode = 'gray', 
    #wave_cloud_0 = 3.95, 
    wave_cloud_0 = 1.0, 
)

rotation_kwargs = dict(
    shared_between_m_set = True,

    rotation_mode = 'convolve', 
)

pRT_Radtrans_kwargs = dict(
    line_species               = chem_kwargs['line_species'],
    rayleigh_species           = ['H2','He'],
    gas_continuum_contributors = ['H2-H2','H2-He'],
    cloud_species              = cloud_kwargs.get('cloud_species'), 
    
    line_opacity_mode             = 'lbl',
    line_by_line_opacity_sampling = 10, # Faster radiative transfer by down-sampling
    scattering_in_emission        = False, 

    pRT_input_data_path = '/net/schenk/data2/regt/pRT3_input_data/input_data', 
)

####################################################################################
# Log-likelihood, covariance keyword-arguments
####################################################################################

loglike_kwargs = dict(
    scale_flux = False, 
    scale_err = True, 
    sum_model_settings = False, 
)

cov_kwargs = dict(
    trunc_dist = 3, 
    scale_amp  = True, 
    #max_wave_sep = 3 * 10**free_params.get('log_l', [None,[None,np.inf]])[1][1], 
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
    n_live_points         = 100, 
    n_iter_before_update  = 100, 
)