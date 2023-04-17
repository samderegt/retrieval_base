import matplotlib.pyplot as plt
import numpy as np

from config_DENIS import prefix

def fig_flux_calib_2MASS(wave, 
                         calib_flux, 
                         calib_flux_wo_tell_corr, 
                         transm, 
                         poly_model, 
                         wave_2MASS, 
                         transm_2MASS, 
                         tell_threshold=0.2, 
                         order_wlen_ranges=None, 
                         ):

    fig, ax = plt.subplots(
        figsize=(10,4), nrows=2, sharex=True, 
        gridspec_kw={'hspace':0, 'height_ratios':[1,0.5], 
                     'left':0.1, 'right':0.95, 'top':0.92, 'bottom':0.15, 
                     }
        )

    ax[0].plot(wave, calib_flux_wo_tell_corr, c='k', lw=0.5, alpha=0.4, 
               label=r'$F_\mathrm{CRIRES}$'
               )
    ax[0].plot(wave, calib_flux, c='k', lw=0.5, 
               label=r'$F_\mathrm{CRIRES}/T_\mathrm{CRIRES}$'
               )
    ax[0].set(ylim=(0, 1.5*np.nanmedian(calib_flux)), 
              ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 
              )
    ax[0].legend(loc='upper left')
    
    ax[1].plot(wave, transm, c='k', lw=0.5, label=r'$T_\mathrm{CRIRES}$')
    #ax[1].plot(wave, transm/poly_model, c='k', lw=0.5, alpha=0.5)
    ax[1].plot(wave, poly_model, c='gray', lw=1)
    ax[1].plot(wave, tell_threshold*poly_model, c='gray', lw=1, ls='--')
    ax[1].plot(wave_2MASS, transm_2MASS, c='r', lw=1, label=r'$T_\mathrm{2MASS}$')
    ax[1].set(xlim=(wave.min()-50, wave.max()+50), xlabel=r'Wavelength (nm)', 
              ylim=(0,1.1), ylabel=r'Transmissivity'
              )
    ax[1].legend(loc='upper left')

    plt.savefig(prefix+'plots/flux_calib_tell_corr.pdf')
    #plt.show()
    plt.close()

    if order_wlen_ranges is not None:
        # Plot zoom-ins of the telluric correction
        n_orders = order_wlen_ranges.shape[0]

        fig, ax = plt.subplots(
            figsize=(10,1.8*n_orders), nrows=n_orders, 
            gridspec_kw={'hspace':0.22, 'left':0.1, 'right':0.95, 'top':0.95, 'bottom':0.05, }
            )
        
        for i in range(n_orders):
            # Only plot within a wavelength range
            wave_min = order_wlen_ranges[i,:].min() - 2
            wave_max = order_wlen_ranges[i,:].max() + 2

            mask_wave = (wave > wave_min) & (wave < wave_max)

            ax[i].plot(
                wave[mask_wave], 
                (calib_flux_wo_tell_corr/poly_model)[mask_wave], 
                c='k', lw=0.5, alpha=0.4
                )
            ax[i].plot(wave[mask_wave], calib_flux[mask_wave], c='k', lw=0.5)

            ax[i].set(xlim=(wave_min, wave_max))

        ax[n_orders//2].set(ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$')
        ax[-1].set(xlabel=r'Wavelength (nm)')
        
        plt.savefig(prefix+'plots/tell_corr_zoom_ins.pdf')
        #plt.show()
        plt.close()

def fig_sigma_clip(wave, flux, flux_wo_clip, sigma_clip_bounds, order_wlen_ranges, sigma):

    # Plot zoom-ins of the sigma-clipping procedure
    n_orders = order_wlen_ranges.shape[0]

    fig, ax = plt.subplots(
        figsize=(10,1.8*n_orders), nrows=n_orders, 
        gridspec_kw={'hspace':0.22, 'left':0.1, 'right':0.95, 'top':0.95, 'bottom':0.05, }
        )
    
    for i in range(n_orders):
        # Only plot within a wavelength range
        wave_min = order_wlen_ranges[i,:].min() - 2
        wave_max = order_wlen_ranges[i,:].max() + 2

        mask_wave = (wave > wave_min) & (wave < wave_max)

        ax[i].plot(wave[mask_wave], flux_wo_clip[mask_wave], c='r', lw=0.3)
        ax[i].plot(wave[mask_wave], flux[mask_wave], c='k', lw=0.5)

        ax[i].fill_between(
            wave[mask_wave], y1=sigma_clip_bounds[0,i], y2=sigma_clip_bounds[2,i], 
            fc='C0', alpha=0.5, ec='none', label=f'{sigma}'+r'$\sigma$'
            )
        ax[i].plot(wave[mask_wave], sigma_clip_bounds[1,i], c='C0')

        ax[i].set(xlim=(wave_min, wave_max))
        
    ax[n_orders//2].set(ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$')
    ax[-1].set(xlabel=r'Wavelength (nm)')
    ax[-1].legend()
    
    plt.savefig(prefix+'plots/sigma_clip_zoom_ins.pdf')
    #plt.show()
    plt.close()