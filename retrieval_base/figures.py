import matplotlib.pyplot as plt
import numpy as np

def fig_order_subplots(n_orders, ylabel, xlabel=r'Wavelength (nm)'):

    fig, ax = plt.subplots(
        figsize=(10,2.5*n_orders), nrows=n_orders, 
        gridspec_kw={'hspace':0.22, 'left':0.1, 'right':0.95, 
                     'top':(1-0.02*7/n_orders), 'bottom':0.035*7/n_orders, 
                     }
        )
    if n_orders == 1:
        ax = np.array([ax])

    ax[n_orders//2].set(ylabel=ylabel)
    ax[-1].set(xlabel=xlabel)

    return fig, ax

def fig_flux_calib_2MASS(wave, 
                         calib_flux, 
                         calib_flux_wo_tell_corr, 
                         transm, 
                         poly_model, 
                         wave_2MASS, 
                         transm_2MASS, 
                         tell_threshold=0.2, 
                         order_wlen_ranges=None, 
                         prefix=None, 
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

    if prefix is not None:
        plt.savefig(prefix+'plots/flux_calib_tell_corr.pdf')
    #plt.show()
    plt.close(fig)

    if order_wlen_ranges is not None:
        # Plot zoom-ins of the telluric correction
        n_orders = order_wlen_ranges.shape[0]

        fig, ax = fig_order_subplots(
            n_orders, 
            ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
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
        
        if prefix is not None:
            plt.savefig(prefix+'plots/tell_corr_zoom_ins.pdf')
        #plt.show()
        plt.close(fig)

def fig_sigma_clip(wave, flux, flux_wo_clip, sigma_clip_bounds, order_wlen_ranges, sigma, prefix=None):

    # Plot zoom-ins of the sigma-clipping procedure
    n_orders = order_wlen_ranges.shape[0]

    fig, ax = fig_order_subplots(
        n_orders, 
        ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
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

    ax[-1].legend()
    
    if prefix is not None:
        plt.savefig(prefix+'plots/sigma_clip_zoom_ins.pdf')
    #plt.show()
    plt.close(fig)

def fig_spec_to_fit(d_spec, prefix=None):

    ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
    if d_spec.high_pass_filtered:
        ylabel = r'$F_\lambda$ (high-pass filtered)'

    fig, ax = fig_order_subplots(d_spec.n_orders, ylabel=ylabel)

    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            ax[i].plot(d_spec.wave[i,j], d_spec.flux[i,j], c='k', lw=0.5)
        
        ax[i].set(xlim=(d_spec.order_wlen_ranges[i].min()-2, 
                        d_spec.order_wlen_ranges[i].max()+2)
                  )

    if prefix is not None:
        plt.savefig(prefix+'plots/spec_to_fit.pdf')
    #plt.show()
    plt.close(fig)

def fig_bestfit_model(d_spec, m_spec, LogLike, bestfit_color='C1', ax_spec=None, ax_res=None, prefix=None):

    if (ax_spec is None) and (ax_res is None):
        # Create a new figure
        is_new_fig = True
        n_orders = d_spec.n_orders

        fig, ax = plt.subplots(
            figsize=(10,2.5*n_orders*2), nrows=n_orders*3, 
            gridspec_kw={'hspace':0, 'height_ratios':[1,1/3,1/5]*n_orders, 
                        'left':0.1, 'right':0.95, 
                        'top':(1-0.02*7/(n_orders*3)), 
                        'bottom':0.035*7/(n_orders*3), 
                        }
            )
    else:
        is_new_fig = False

    ylabel_spec = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
    if d_spec.high_pass_filtered:
        ylabel_spec = r'$F_\lambda$ (high-pass filtered)'

    # Use the same ylim, also for multiple axes
    ylim_spec = (np.nanmean(d_spec.flux)-4*np.nanstd(d_spec.flux), 
                 np.nanmean(d_spec.flux)+4*np.nanstd(d_spec.flux)
                )
    ylim_res = (1/3*(ylim_spec[0]-np.nanmean(d_spec.flux)), 
                1/3*(ylim_spec[1]-np.nanmean(d_spec.flux))
                )

    for i in range(d_spec.n_orders):

        if is_new_fig:
            # Spectrum and residual axes
            ax_spec = ax[i*3]
            ax_res  = ax[i*3+1]

            # Remove the temporary axis
            ax[i*3+2].remove()

            # Use a different xlim for the separate figures
            xlim = (d_spec.wave[i,:].min()-2, 
                    d_spec.wave[i,:].max()+2)
        else:
            xlim = (d_spec.wave.min()-2, 
                    d_spec.wave.max()+2)

        ax_spec.set(xlim=xlim, xticks=[], ylim=ylim_spec)
        ax_res.set(xlim=xlim, ylim=ylim_res)

        for j in range(d_spec.n_dets):
        
            mask_ij = d_spec.mask_isfinite[i,j]
            if mask_ij.any():
                # Show the observed and model spectra
                ax_spec.plot(
                    d_spec.wave[i,j], d_spec.flux[i,j], 
                    c='k', lw=0.5, label='Observation'
                    )

            label = 'Best-fit model ' + \
                    r'$(\chi^2_\mathrm{red}$ (w/o $\sigma$-model)$=' + \
                    '{:.2f}'.format(LogLike.chi_squared_red) + \
                    r')$'
            ax_spec.plot(
                d_spec.wave[i,j], LogLike.f[i,j]*m_spec.flux[i,j], 
                c=bestfit_color, lw=1, label=label
                )

            if mask_ij.any():

                # Plot the residuals
                res_ij = d_spec.flux[i,j] - LogLike.f[i,j]*m_spec.flux[i,j]
                ax_res.plot(d_spec.wave[i,j], res_ij, c='k', lw=0.5)
                ax_res.plot(
                    [d_spec.wave[i,j].min(), d_spec.wave[i,j].max()], 
                    [0,0], c=bestfit_color, lw=1
                )

                # Show the mean error
                mean_err_ij = np.mean(d_spec.err[i,j][d_spec.mask_isfinite[i,j]])
                ax_res.errorbar(
                    d_spec.wave[i,j].min()-0.2, 0, yerr=1*mean_err_ij, 
                    fmt='none', lw=1, ecolor='k', capsize=2, color='k', 
                    label=r'$\langle\sigma_{ij}\rangle$'
                    )

                # Show the mean error after scaling with beta
                #mean_err_ij += np.sqrt(LogLike.params['a'][i,j])*mean_err_ij
                mean_scaled_err_ij = LogLike.beta[i,j]*mean_err_ij
                ax_res.errorbar(
                    d_spec.wave[i,j].min()-0.4, 0, yerr=1*mean_scaled_err_ij, 
                    fmt='none', lw=1, ecolor=bestfit_color, capsize=2, color=bestfit_color, 
                    #label=r'$\beta_{ij}\langle\sigma_{ij}\rangle$'
                    label=r'Modelled $\langle\sigma_{ij}\rangle$'
                    )

            if i==0 and j==0:
                ax_spec.legend(
                    loc='upper right', ncol=2, fontsize=8, handlelength=1, 
                    framealpha=0.7, handletextpad=0.3, columnspacing=0.8
                    )
                ax_res.legend(
                    loc='upper right', ncol=2, fontsize=8, handlelength=1, 
                    framealpha=0.7, handletextpad=0.3, columnspacing=0.8
                    )

    # Set the labels for the final axis
    ax_spec.set(ylabel=ylabel_spec)
    ax_res.set(xlabel='Wavelength (nm)', ylabel='Residuals')

    if is_new_fig and (prefix is not None):
        plt.savefig(prefix+'plots/live_bestfit_spec.pdf')
        plt.close(fig)
    else:
        return ax_spec, ax_res

def fig_cov(LogLike, d_spec, cmap, prefix=None):

    all_cov = np.zeros(
        (d_spec.n_orders, d_spec.n_dets, 
         d_spec.n_pixels, d_spec.n_pixels)
        )
    vmax = np.zeros((d_spec.n_orders, d_spec.n_dets))
    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            
            # Only store the valid pixels
            mask_ij = d_spec.mask_isfinite[i,j]

            # Get the covariance matrix for this order/detector
            cov_ij = LogLike.cov[i,j]

            if cov_ij.is_matrix:
                if cov_ij.cholesky_mode == 'sparse':
                    cov = cov_ij.cov.todense()
                elif cov_ij.cholesky_mode == 'banded':
                    cov = cov_ij.cov
            else:
                cov = np.diag(cov_ij.cov)

            # Scale with the optimal uncertainty scaling
            cov *= LogLike.beta[i,j]

            # Insert the masked rows into the covariance matrix
            indices = np.arange(0, d_spec.n_pixels, 1)[~mask_ij]
            for idx in indices:
                cov = np.insert(cov, idx, np.zeros(mask_ij.sum()), axis=0)
            for idx in indices:
                cov = np.insert(cov, idx, np.zeros(d_spec.n_pixels), axis=1)

            # Add to the complete array
            all_cov[i,j,:,:] = cov

            # Store the median of the diagonal
            vmax[i,j] = np.median(np.diag(cov))

    # Use a single range in the matshows
    vmin, vmax = 0, np.max(vmax)


    fig, ax = plt.subplots(
        figsize=(10*d_spec.n_dets/3, 3.5*d_spec.n_orders), 
        nrows=d_spec.n_orders, ncols=d_spec.n_dets, 
        gridspec_kw={
            'wspace':0.1, 'hspace':0.1, 
            'left':0.08, 'right':0.95, 
            'top':0.95, 'bottom':0.08
            }
        )
    if d_spec.n_orders == 1:
        ax = np.array([ax])

    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):

            extent = [
                d_spec.wave[i,j].min(), d_spec.wave[i,j].max(), 
                d_spec.wave[i,j].max(), d_spec.wave[i,j].min(), 
                ]
            ax[i,j].matshow(
                all_cov[i,j], aspect=1, extent=extent, cmap=cmap, 
                interpolation='none', vmin=vmin, vmax=vmax
                )
            ticks = np.linspace(d_spec.wave[i,j].min()+2, d_spec.wave[i,j].max()-2, num=4)
            ax[i,j].set_xticks(
                ticks, labels=['{:.0f}'.format(t_i) for t_i in ticks]
                )
            ax[i,j].set_yticks(
                ticks, labels=['{:.0f}'.format(t_i) for t_i in ticks], 
                rotation=90, va='center'
                )
            ax[i,j].tick_params(
                axis='x', which='both', bottom=False, top=True, labelbottom=False
                )
            ax[i,j].grid(True, alpha=0.1)

    ax[-1,1].set(xlabel='Wavelength (nm)')
    ax[d_spec.n_orders//2,0].set(ylabel='Wavelength (nm)')

    if prefix is not None:
        plt.savefig(prefix+'plots/live_cov.pdf')
        plt.close(fig)