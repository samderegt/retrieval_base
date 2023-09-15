import numpy as np
import pymultinest

import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline

import json

import corner

import retrieval_base.auxiliary_functions as af

def read_results(prefix, n_params):

    # Set-up analyzer object
    analyzer = pymultinest.Analyzer(
        n_params=n_params, 
        outputfiles_basename=prefix
        )
    stats = analyzer.get_stats()

    # Load the equally-weighted posterior distribution
    posterior = analyzer.get_equal_weighted_posterior()
    posterior = posterior[:,:-1]

    # Read the parameters of the best-fitting model
    bestfit = np.array(stats['modes'][0]['maximum a posterior'])

    PT = af.pickle_load(prefix+'data/bestfit_PT.pkl')
    Chem = af.pickle_load(prefix+'data/bestfit_Chem.pkl')

    m_spec = af.pickle_load(prefix+'data/bestfit_m_spec.pkl')
    d_spec = af.pickle_load(prefix+'data/d_spec.pkl')

    LogLike = af.pickle_load(prefix+'data/bestfit_LogLike.pkl')

    try:
        Cov = af.pickle_load(prefix+'data/bestfit_Cov.pkl')
    except:
        Cov = None

    int_contr_em           = np.load(prefix+'data/bestfit_int_contr_em.npy')
    int_contr_em_per_order = np.load(prefix+'data/bestfit_int_contr_em_per_order.npy')
    int_opa_cloud          = np.load(prefix+'data/bestfit_int_opa_cloud.npy')

    f = open(prefix+'data/bestfit.json')
    bestfit_params = json.load(f)
    f.close()

    print(posterior.shape)
    return posterior, bestfit, PT, Chem, int_contr_em, int_contr_em_per_order, int_opa_cloud, m_spec, d_spec, LogLike, Cov, bestfit_params

res = read_results(
    prefix='./retrieval_outputs/order_6_ret_4/test_', n_params=36
)
posterior, bestfit, PT, Chem, _, _, _, m_spec, d_spec, LogLike, _, bestfit_params = res

from petitRADTRANS import Radtrans
import petitRADTRANS.nat_cst as nc
from PyAstronomy import pyasl
from scipy.ndimage import gaussian_filter, generic_filter

def opa_cross_sections(
        species, 
        species_mass, 
        T, 
        P, 
        epsilon_limb=0.4, 
        vsini=14, 
        rv=17.3, 
        out_res=5e4, 
        #wlen_bords_micron=[(d_spec.wave.min()-5)/1e3,(d_spec.wave.max()+5)/1e3]
        wlen_bords_micron=[1.9,2.5]
        ):
    print('?')
    pRT_atm = Radtrans(
        line_species=[species], wlen_bords_micron=wlen_bords_micron, mode='lbl'
        )
    print('?')
    res = pRT_atm.plot_opas(
        species=[species], temperature=T, pressure_bar=P, return_opacities=True
        )
    opa = res[species][1].flatten() * (species_mass*1.66053892e-24)

    wave = (nc.c / pRT_atm.freq * 1e7) 
    wave = wave * (1 + rv/(nc.c*1e-5))

    wave_even = np.linspace(wave.min(), wave.max(), wave.size)

    opa_even = np.interp(wave_even, xp=wave, fp=opa)
    opa_rot_broad = pyasl.fastRotBroad(
        wave_even, opa_even, epsilon=epsilon_limb, vsini=vsini
        )
    opa_rot_broad = instr_broadening(wave_even, opa_rot_broad, out_res=out_res, in_res=1e6)

    return wave_even, opa_rot_broad

def instr_broadening(wave, flux, out_res=1e6, in_res=1e6):

    # Delta lambda of resolution element is FWHM of the LSF's standard deviation
    sigma_LSF = np.sqrt(1/out_res**2 - 1/in_res**2) / \
                (2*np.sqrt(2*np.log(2)))

    spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))

    # Calculate the sigma to be used in the gauss filter in pixels
    sigma_LSF_gauss_filter = sigma_LSF / spacing
    
    # Apply gaussian filter to broaden with the spectral resolution
    flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter, 
                                mode='nearest'
                                )
    return flux_LSF

fig, ax = plt.subplots(figsize=(12,4))
for i in range(d_spec.n_orders):
    for j in range(d_spec.n_dets):
        ax.plot(d_spec.wave[i,j,:], d_spec.flux[i,j,:] - LogLike.f[i,j]*m_spec.flux[i,j,:], c='k', lw=0.5)

ax_opa = ax.twinx()
#ax_opa.plot(
#    *opa_cross_sections(
#        'H2_12', species_mass=3, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )

#ax_opa.plot(
#    *opa_cross_sections(
#        'Na_allard', species_mass=23, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )

#ax_opa.plot(
#    *opa_cross_sections(
#        'Ti', species_mass=48, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )

#ax_opa.plot(
#    *opa_cross_sections(
#        'K', species_mass=39, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )
ax_opa.plot(
    *opa_cross_sections(
        'K_allard_cold', species_mass=39, T=1200, P=1, 
        ), 
    lw=0.5, alpha=1
    )

#ax_opa.plot(
#    *opa_cross_sections(
#        'HDO_voronin', species_mass=19, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )
#ax_opa.plot(
#    *opa_cross_sections(
#        'H2O_pokazatel_main_iso', species_mass=18, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )
#ax_opa.plot(
#    *opa_cross_sections(
#        'H2O_main_iso', species_mass=18, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )

#ax_opa.plot(
#    *opa_cross_sections(
#        'CO_36', species_mass=13.003355 + 15.999, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )
#ax_opa.plot(
#    *opa_cross_sections(
#        'CO_28', species_mass=12.011 + 17.9991610, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )
#ax_opa.plot(
#    *opa_cross_sections(
#        'CO_27', species_mass=12.011 + 16.999131, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )

#ax_opa.plot(
#    *opa_cross_sections(
#        'NH3_coles_main_iso', species_mass=17, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )

#ax_opa.plot(
#    *opa_cross_sections(
#        'CH4_hargreaves_main_iso', species_mass=16, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )

#ax_opa.plot(
#    *opa_cross_sections(
#        'HCN_main_iso', species_mass=27, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )

#ax_opa.plot(
#    *opa_cross_sections(
#        'H2S_main_iso', species_mass=34.082, T=1200, P=1, 
#        ), 
#    lw=0.5, alpha=1
#    )

#ax.set(xlim=(d_spec.wave[0].min(), d_spec.wave[0].max()))

#plt.savefig('./plots/opa_cross_sec.pdf')
plt.show()
plt.close()