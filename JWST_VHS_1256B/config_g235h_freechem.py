import numpy as np

####################################################################################
# Files and physical parameters
####################################################################################

prefix = 'g235h_ret_8'
prefix = f'./retrieval_outputs/{prefix}/test_'

config_data = dict(
    #nirspec_g140h_1 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_140h_1.dat', 'grating':'G140H', 'min_SNR':3}), 
    #nirspec_g140h_2 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_140h_2.dat', 'grating':'G140H', 'min_SNR':3}), 
    #nirspec_g235h_1 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_235h_1.dat', 'grating':'G235H', 'min_SNR':3}),
    nirspec_g235h_2 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_235h_2.dat', 'grating':'G235H', 'min_SNR':3}),
    #nirspec_g395h_1 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_395h_1.dat', 'grating':'G395H', 'min_SNR':3}),
    #nirspec_g395h_2 = dict(instrument='JWST', kwargs={'file':'./data/nirspec_395h_2.dat', 'grating':'G395H', 'min_SNR':3}), 
)

####################################################################################
# Model parameters
####################################################################################

# Define the priors of the parameters
free_params = {
    # Covariance parameters
    #'log_a': ['U', (-0.5,0.5), r'$\log\ a$'], 
    #'log_l': ['U', (-2.0,0.5), r'$\log\ l$'], 

    # General properties
    'M_p': ['G', (19.0,5.0), r'$\mathrm{M_p}$'], 
    'R_p': ['G', (1.1,0.2), r'$\mathrm{R_p}$'], 
    'rv':  ['U', (-10.,7.), r'$v_\mathrm{rad}$'], 

    # Broadening
    'vsini': ['U', (0.,20.), r'$v\ \sin\ i$'], 

    # Cloud properties
    'log_K_zz': ['U', (5,13), r'$\log\ K_\mathrm{zz,cl}$'],
    'sigma_g':  ['U', (1.05,3), r'$\sigma_g$'],
    'log_X_Mg2SiO4(s)_amorphous__Mie': ['U', (-2.3,1), r'$\log\ X_\mathrm{Mg2SiO4}$'],
    'f_sed_Mg2SiO4(s)_amorphous__Mie': ['U', (0,10), r'$f_\mathrm{sed,Mg2SiO4}$'],
    'log_X_MgSiO3(s)_amorphous__Mie':  ['U', (-2.3,1), r'$\log\ X_\mathrm{MgSiO3}$'],
    'f_sed_MgSiO3(s)_amorphous__Mie':  ['U', (0,10), r'$f_\mathrm{sed,MgSiO3}$'],
    'log_X_Fe(s)_amorphous__Mie':      ['U', (-2.3,1), r'$\log\ X_\mathrm{Fe}$'],
    'f_sed_Fe(s)_amorphous__Mie':      ['U', (0,10), r'$f_\mathrm{sed,Fe}$'],

    # Chemistry
    'log_H2O':     ['U', (-14,-2), r'$\log\ \mathrm{H_2O}$'],
    'log_12CO':    ['U', (-14,-2), r'$\log\ \mathrm{^{12}CO}$'],
    'log_CH4':     ['U', (-14,-2), r'$\log\ \mathrm{CH_4}$'],
    'log_HF':      ['U', (-14,-2), r'$\log\ \mathrm{HF}$'],
    'log_H2S':     ['U', (-14,-2), r'$\log\ \mathrm{H_2S}$'],
    'log_CO2':     ['U', (-14,-2), r'$\log\ \mathrm{CO_2}$'],
    'log_HCN':     ['U', (-14,-2), r'$\log\ \mathrm{HCN}$'],
    'log_NH3':     ['U', (-14,-2), r'$\log\ \mathrm{NH_3}$'],
    
    'log_12/13C_ratio': ['U', (0,5), r'$\log\ \mathrm{^{12}/^{13}C}$'],
    'log_16/18O_ratio': ['U', (0,5), r'$\log\ \mathrm{^{16}/^{18}O}$'],

    # PT profile
    'dlnT_dlnP_0': ['U', (0.1,0.34), r'$\nabla_0$'], 
    'dlnT_dlnP_1': ['U', (0.05,0.34), r'$\nabla_1$'], 
    'dlnT_dlnP_2': ['U', (0.0,0.34), r'$\nabla_2$'], 
    'dlnT_dlnP_3': ['U', (0.0,0.34), r'$\nabla_3$'], 
    'dlnT_dlnP_4': ['U', (0.0,0.34), r'$\nabla_4$'], 
    'dlnT_dlnP_5': ['U', (-0.1,0.34), r'$\nabla_5$'], 

    'T_phot':         ['U', (800.,2000.), r'$T_\mathrm{phot}$'], 
    #'log_P_phot':     ['U', (-2.,0.5), r'$\log\ P_\mathrm{phot}$'], 
    #'d_log_P_phot+1': ['U', (0.5,2.5), r'$\Delta P_\mathrm{+1}$'], 
    #'d_log_P_phot-1': ['U', (0.5,1.5), r'$\Delta P_\mathrm{-1}$'], 
}

# Constants to use if prior is not given
constant_params = {
    # General properties
    'parallax': 47.2733,  # +/- 37 mas

    'epsilon_limb': 0.6, # Limb-darkening
    'C/O': 0.59, # Solar ratios
    'Fe/H': 0.0,
    'N/O': 0.14,

    'log_P_phot': 0., 
    'd_log_P_phot+2': 2.,
    'd_log_P_phot+1': 1., 
    'd_log_P_phot-1': 1.,
}

####################################################################################
# Physical model keyword-arguments
####################################################################################

PT_kwargs = dict(
    shared_between_m_set = True,
    PT_mode = 'free_gradient', 
    n_knots = 6, 
    interp_mode = 'linear', 

    log_P_range = (-5.,2.), 
    n_atm_layers = 50,
    #log_P_knots = np.array([-5.,-2.,-1.,0.,1.,2.]),
)

chem_kwargs = dict(
    shared_between_m_set = False,
    chem_mode = 'free', 
    nirspec_g235h_2 = dict(
        line_species = [
            '1H2-16O__POKAZATEL',
            '1H2-18O__HotWat78',
            '12C-16O__HITEMP',
            '13C-16O__HITEMP',
            '12C-1H4__MM',
            '13C-1H4__HITRAN',
            '1H-19F__Coxon-Hajig',
            '1H2-32S__AYT2',
            '12C-16O2__HITEMP',
            '1H-12C-14N__Harris',
            '14N-1H3__CoYuTe',
        ], 
    ), 
)

cloud_kwargs = dict(
    shared_between_m_set = False,
    #cloud_mode = None, 
    #cloud_mode = 'gray', wave_cloud_0 = 1.0, 
    cloud_mode = 'EddySed', cloud_species = ['Mg2SiO4(s)_amorphous__Mie', 'MgSiO3(s)_amorphous__Mie', 'Fe(s)_amorphous__Mie'], 
)

rotation_kwargs = dict(
    shared_between_m_set = True,
    rotation_mode = 'convolve', 
)

pRT_Radtrans_kwargs = dict(
    #line_species               = chem_kwargs['line_species'],
    nirspec_g235h_2 = dict(line_species=chem_kwargs['nirspec_g235h_2']['line_species']),

    rayleigh_species           = ['H2','He'],
    gas_continuum_contributors = ['H2-H2','H2-He'],
    cloud_species              = cloud_kwargs.get('cloud_species'), 
    
    line_opacity_mode             = 'lbl',
    line_by_line_opacity_sampling = 10, # Faster radiative transfer by down-sampling
    scattering_in_emission        = True, 
    #scattering_in_emission        = False, 

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
    n_live_points         = 200, 
    n_iter_before_update  = 100, 
)