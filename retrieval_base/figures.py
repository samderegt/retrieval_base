import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import generic_filter, gaussian_filter1d

import os

import petitRADTRANS.nat_cst as nc

import retrieval_base.auxiliary_functions as af

def fig_order_subplots(n_orders, ylabel, xlabel=r'Wavelength (nm)'):

    fig, ax = plt.subplots(
        figsize=(10,2.8*n_orders), nrows=n_orders, 
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

    #'''
    ax[0].plot(wave, calib_flux_wo_tell_corr, c='k', lw=0.5, alpha=0.4, 
               label=r'$F_\mathrm{CRIRES}$'
               )
    #'''
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
            fc='C0', alpha=0.4, ec='none', label=f'{sigma}'+r'$\sigma$'
            )
        ax[i].plot(wave[mask_wave], sigma_clip_bounds[1,i], c='C0', lw=0.5)

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

def fig_bestfit_model(d_spec, m_spec, LogLike, Cov, bestfit_color='C1', ax_spec=None, ax_res=None, prefix=None):

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
            if m_spec.flux_envelope is not None:
                ax_spec.plot(
                    d_spec.wave[i,j], m_spec.flux_envelope[3,i,j], c='C0', lw=1
                    )

            if mask_ij.any():

                # Plot the residuals
                res_ij = d_spec.flux[i,j] - LogLike.f[i,j]*m_spec.flux[i,j]
                ax_res.plot(d_spec.wave[i,j], res_ij, c='k', lw=0.5)
                '''
                ax_res.plot(
                    d_spec.wave[i,j], 
                    np.random.normal(0, d_spec.err[i,j], size=len(d_spec.err[i,j]))*res_ij/res_ij, 
                    c='k', lw=0.5
                    )
                '''
                ax_res.plot(
                    [d_spec.wave[i,j].min(), d_spec.wave[i,j].max()], 
                    [0,0], c=bestfit_color, lw=1
                )

                if m_spec.flux_envelope is not None:
                    ax_res.plot(
                        d_spec.wave[i,j], m_spec.flux_envelope[3,i,j] - LogLike.f[i,j]*m_spec.flux[i,j], 
                        c='C0', lw=1
                        )

                # Show the mean error
                mean_err_ij = np.mean(d_spec.err[i,j][d_spec.mask_isfinite[i,j]])
                ax_res.errorbar(
                    d_spec.wave[i,j].min()-0.2, 0, yerr=1*mean_err_ij, 
                    fmt='none', lw=1, ecolor='k', capsize=2, color='k', 
                    label=r'$\langle\sigma_{ij}\rangle$'
                    )

                # Get the covariance matrix
                cov = Cov[i,j].cov
                if Cov[i,j].is_matrix:
                    if Cov[i,j].cholesky_mode == 'sparse':
                        cov = cov.todense()
                else:
                    cov = np.diag(cov)
                
                # Scale with the optimal uncertainty-scaling
                cov *= LogLike.beta[i,j]

                # Get the mean error from the trace
                #mean_err_ij = np.sqrt(np.trace(cov)/len(cov))
                #mean_scaled_err_ij = LogLike.beta[i,j]*mean_err_ij
                mean_scaled_err_ij = np.mean(np.diag(np.sqrt(LogLike.beta[i,j]**2 * cov)))

                ax_res.errorbar(
                    d_spec.wave[i,j].min()-0.4, 0, yerr=1*mean_scaled_err_ij, 
                    fmt='none', lw=1, ecolor=bestfit_color, capsize=2, color=bestfit_color, 
                    #label=r'$\beta_{ij}\langle\sigma_{ij}\rangle$'
                    label=r'$\beta_{ij}\cdot\langle\mathrm{diag}(\sqrt{\Sigma_{ij}})\rangle$'
                    )

            if i==0 and j==0:
                ax_spec.legend(
                    loc='upper right', ncol=2, fontsize=8, handlelength=1, 
                    framealpha=0.7, handletextpad=0.3, columnspacing=0.8
                    )
                '''
                ax_res.legend(
                    loc='upper right', ncol=2, fontsize=8, handlelength=1, 
                    framealpha=0.7, handletextpad=0.3, columnspacing=0.8
                    )
                '''

    # Set the labels for the final axis
    ax_spec.set(ylabel=ylabel_spec)
    ax_res.set(xlabel='Wavelength (nm)', ylabel='Residuals')

    if is_new_fig and (prefix is not None):
        plt.savefig(prefix+'plots/bestfit_spec.pdf')
        plt.close(fig)
    else:
        return ax_spec, ax_res

def fig_cov(LogLike, Cov, d_spec, cmap, prefix=None):

    all_cov = np.zeros(
        (d_spec.n_orders, d_spec.n_dets, 
         d_spec.n_pixels, d_spec.n_pixels)
        )
    vmax = np.zeros((d_spec.n_orders, d_spec.n_dets))
    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            
            # Only store the valid pixels
            mask_ij = d_spec.mask_isfinite[i,j]

            # Get the covariance matrix
            cov = Cov[i,j].cov
            if Cov[i,j].is_matrix:
                if Cov[i,j].cholesky_mode == 'sparse':
                    cov = cov.todense()
            else:
                cov = np.diag(cov)

            # Scale with the optimal uncertainty scaling
            cov *= LogLike.beta[i,j]**2

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
    vmin, vmax = 0, 0.3*np.max(vmax)


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
        plt.savefig(prefix+'plots/cov_matrices.pdf')
        plt.close(fig)

    return all_cov

def fig_PT(PT, 
           integrated_contr_em=None, 
           integrated_contr_em_per_order=None, 
           integrated_opa_cloud=None, 
           ax_PT=None, 
           envelope_colors=None, 
           posterior_color='C0', 
           bestfit_color='C1', 
           ylabel=r'$P\ \mathrm{(bar)}$', 
           yticks=np.logspace(-6, 2, 9), 
           xlim=(1,3500), 
           show_ln_L_penalty=False, 
           prefix=None
           ):
    
    if ax_PT is None:
        fig, ax_PT = plt.subplots(
            figsize=(4.5,4.5), 
            gridspec_kw={'left':0.16, 'right':0.94, 
                         'top':0.87, 'bottom':0.15
                         }
            )

        is_new_fig = True
    else:
        is_new_fig = False

    if PT.temperature_envelopes is not None:
        # Plot the PT confidence envelopes
        for i in range(3):
            ax_PT.fill_betweenx(
                y=PT.pressure, x1=PT.temperature_envelopes[i], 
                x2=PT.temperature_envelopes[-i-1], 
                color=envelope_colors[i+1], ec='none', 
                )

        # Plot the median PT
        ax_PT.plot(
            PT.temperature_envelopes[3], PT.pressure, 
            c=posterior_color, lw=1
        )
    
    # Plot the best-fitting PT profile and median
    if show_ln_L_penalty:
        label = r'$\ln\ \mathrm{L\ penalty}=' + \
                '{:.0f}'.format(np.sign(PT.ln_L_penalty)*10) + '^{' + \
                '{:.2f}'.format(np.log10(np.abs(PT.ln_L_penalty))) + '}$'
    else:
        label = None
    
    ax_PT.plot(
        PT.temperature, PT.pressure, c=bestfit_color, lw=1, label=label
        )
    ax_PT.plot(
        PT.T_knots, PT.P_knots, c=bestfit_color, ls='', marker='o', markersize=3
        )

    if show_ln_L_penalty:
        ax_PT.legend(
            loc='upper right', handlelength=0.5, 
            handletextpad=0.5, framealpha=0.7
            )
        
    ax_PT.set(
        xlabel=r'$T\ \mathrm{(K)}$', xlim=xlim, 
        ylabel=ylabel, ylim=(PT.pressure.min(), PT.pressure.max()), 
        yscale='log', yticks=yticks
        )
    ax_PT.invert_yaxis()

    if (integrated_contr_em != 0).any():
        # Add the integrated emission contribution function
        ax_contr = ax_PT.twiny()
        fig_contr_em(
            ax_contr, integrated_contr_em, 
            integrated_contr_em_per_order, 
            PT.pressure, bestfit_color=bestfit_color
            )
    
    if (integrated_opa_cloud != 0).any():
        # Add the integrated emission contribution function
        ax_opa_cloud = ax_PT.twiny()
        fig_opa_cloud(
            ax_opa_cloud, integrated_opa_cloud, PT.pressure, 
            xlim=(1e0, 1e-10), color='grey'
            )
    
    # Save or return the axis
    if is_new_fig and (prefix is not None):
        fig.savefig(prefix+'plots/PT_profile.pdf')
        plt.close(fig)
    else:
        return ax_PT

def fig_contr_em(ax_contr, integrated_contr_em, integrated_contr_em_per_order, pressure, bestfit_color='C1'):
    
    if integrated_contr_em_per_order is not None:
        
        if len(integrated_contr_em_per_order) != 1:        
            # Plot the emission contribution functions per order
            #cmap_per_order = mpl.cm.get_cmap('coolwarm')
            cmap_per_order = mpl.colors.LinearSegmentedColormap.from_list(
                'cmap_per_order', colors=['b', 'r']
                )
            
            for i in range(len(integrated_contr_em_per_order)):
                color_i = cmap_per_order(i/(len(integrated_contr_em_per_order)-1))
                ax_contr.plot(
                    integrated_contr_em_per_order[i], pressure, 
                    c=color_i, alpha=0.5, lw=1, 
                    )

    ax_contr.plot(
        integrated_contr_em, pressure, 
        c=bestfit_color, ls='--', alpha=0.7
        )
    ax_contr.set(xlim=(0, 1.1*np.nanmax(integrated_contr_em)))

    ax_contr.tick_params(
        axis='x', which='both', top=False, labeltop=False
        )

    return ax_contr

def fig_opa_cloud(ax_opa_cloud, integrated_opa_cloud, pressure, xlim=(1e0, 1e-10), color='grey'):

    ax_opa_cloud.plot(
        integrated_opa_cloud, pressure, c=color, ls='--', alpha=0.7
        )
    
    # Set the color of the upper axis-spine
    ax_opa_cloud.tick_params(
        axis='x', which='both', top=True, labeltop=True, colors=color
        )
    ax_opa_cloud.spines['top'].set_color(color)
    ax_opa_cloud.xaxis.label.set_color(color)

    ax_opa_cloud.set(
        xlabel=r'$\kappa_\mathrm{cloud}\ (\mathrm{cm^2\ g^{-1}})$', 
        xlim=xlim, xscale='log', 
        )


def fig_VMR(ax_VMR, 
            Chem, 
            species_to_plot, 
            pressure, 
            ylabel=r'$P\ \mathrm{(bar)}$', 
            yticks=np.logspace(-6, 2, 9), 
            xlim=(1e-8, 1e-2), 
            ):

    MMW = Chem.mass_fractions['MMW']

    for species_i in Chem.species_info.keys():
        
        if species_i not in species_to_plot:
            continue

        # Check if the line species was included in the model
        line_species_i = Chem.read_species_info(species_i, info_key='pRT_name')
        if line_species_i in Chem.line_species:

            # Read the mass, color and label
            mass_i  = Chem.read_species_info(species_i, info_key='mass')
            color_i = Chem.read_species_info(species_i, info_key='color')
            label_i = Chem.read_species_info(species_i, info_key='label')

            # Convert the mass fraction to a VMR
            mass_fraction_i = Chem.mass_fractions[line_species_i]
            VMR_i = mass_fraction_i * MMW/mass_i

            # Plot volume-mixing ratio as function of pressure
            ax_VMR.plot(VMR_i, pressure, c=color_i, lw=1, label=label_i)
            
            if hasattr(Chem, 'P_quench') and Chem.read_species_info(species_i, info_key='C') > 0:

                P_quench_i   = pressure[pressure < Chem.P_quench][-1]
                VMR_quench_i = VMR_i[pressure < Chem.P_quench][-1]
                # Interpolate to find the quenched VMR
                ax_VMR.scatter(VMR_quench_i, P_quench_i, c=color_i, s=20, marker='_')

            if Chem.mass_fractions_envelopes is not None:
                # Plot the VMR envelope as well
                cmap_i = mpl.colors.LinearSegmentedColormap.from_list(
                    'cmap_i', colors=['w', color_i]
                    )
                VMR_envelope_colors_i = cmap_i([0.0,0.2,0.4,0.6,0.8])

                for j in range(3):
                    ax_VMR.fill_betweenx(
                        y=pressure, 
                        x1=MMW/mass_i * Chem.mass_fractions_envelopes[line_species_i][j], 
                        x2=MMW/mass_i * Chem.mass_fractions_envelopes[line_species_i][-j-1], 
                        color=VMR_envelope_colors_i[j+1], ec='none', 
                        )

    ax_VMR.legend(
        loc='upper left', handlelength=0.5, 
        handletextpad=0.3, framealpha=0.7
        )
    ax_VMR.set(
        xlabel='VMR', xscale='log', xlim=xlim, 
        ylabel=ylabel, yscale='log', yticks=yticks, 
        ylim=(pressure.min(), pressure.max()), 
        )
    ax_VMR.invert_yaxis()

    return ax_VMR

def fig_hist_posterior(posterior_i, 
                       param_range_i, 
                       param_quantiles_i, 
                       param_key_i, 
                       posterior_color='C0', 
                       title=None, 
                       bins=20, 
                       prefix=None, 
                       ):

    for _ in range(2):

        title = title + r'$=' + '{:.2f}'.format(param_quantiles_i[1])
        title = title + '^{' + '+{:.2f}'.format(param_quantiles_i[2]-param_quantiles_i[1]) + '}'
        title = title + '_{' + '-{:.2f}'.format(param_quantiles_i[1]-param_quantiles_i[0]) + '}$'
        
        fig, ax_hist = plt.subplots(figsize=(3,3))
        # Plot the posterior of parameter i as a histogram
        ax_hist.hist(
            posterior_i, bins=bins, range=param_range_i, 
            color=posterior_color, histtype='step', 
            )

        # Indicate the 68% confidence interval and median
        ax_hist.axvline(param_quantiles_i[0], c=posterior_color, ls='--')
        ax_hist.axvline(param_quantiles_i[1], c=posterior_color, ls='-')
        ax_hist.axvline(param_quantiles_i[2], c=posterior_color, ls='--')
        
        ax_hist.set(yticks=[], xlim=param_range_i, title=title)

        # Save the histogram
        if prefix is not None:
            # Make the histograms directory
            if not os.path.exists(prefix+'plots/hists'):
                os.makedirs(prefix+'plots/hists')
                
            fig.savefig(prefix+f'plots/hists/{param_key_i}.pdf')
            plt.close(fig)

        if not param_key_i.startswith('log_'):
            break

        else:
            # Plot another histogram with linear values
            param_key_i = param_key_i.replace('log_', '')
            posterior_i = 10**posterior_i

            if param_key_i == 'C_ratio':
                posterior_i = 1/posterior_i
                title = r'$\mathrm{^{12}C/^{13}C}$'                

            param_quantiles_i = af.quantiles(posterior_i, q=[0.16,0.5,0.84])
            param_range_i = (
                4*(param_quantiles_i[0]-param_quantiles_i[1])+param_quantiles_i[1], 
                4*(param_quantiles_i[2]-param_quantiles_i[1])+param_quantiles_i[1]
                )

            # Update the mathtext title
            title = title.replace('\\log\\ ', '')

def fig_residual_ACF(d_spec, 
                     m_spec, 
                     LogLike, 
                     Cov, 
                     rv=np.arange(-500,500+1e-6,1), 
                     bestfit_color='C1', 
                     prefix=None
                     ):

    import copy
    # Create a spectrum residual object
    d_spec_res = copy.deepcopy(d_spec)
    d_spec_res.flux = (d_spec.flux - m_spec.flux*LogLike.f[:,:,None])

    # Cross-correlate the residuals with itself
    rv, ACF, _, _ = af.CCF(
        d_spec=d_spec_res, m_spec=d_spec_res, 
        m_wave_pRT_grid=None, m_flux_pRT_grid=None, 
        rv=rv, apply_high_pass_filter=False, 
        )
    del d_spec_res

    n_orders, n_dets = d_spec.n_orders, d_spec.n_dets

    fig, ax = plt.subplots(
        figsize=(10,3*n_orders), 
        nrows=n_orders, ncols=n_dets,
        sharey=True, 
        gridspec_kw={
            'wspace':0.1, 'hspace':0.25, 
            'left':0.1, 'right':0.95, 
            'top':1-0.03*7/n_orders, 
            'bottom':0.03*7/n_orders, 
            }
    )
    if n_orders == 1:
        ax = np.array([ax])

    for i in range(n_orders):
        for j in range(n_dets):
            
            mask_ij = d_spec.mask_isfinite[i,j]
            wave_ij = d_spec.wave[i,j,mask_ij]
            delta_wave_ij = wave_ij[None,:] - wave_ij[:,None]

            # Plot the auto-correlation of residuals
            ax[i,j].plot(rv, ACF[i,j], lw=0.5, c='k')
            ax[i,j].plot(rv, ACF[i,j]*0, lw=0.1, c='k')
            ax[i,j].set(xlim=(rv.min(), rv.max()))


            # Get the covariance matrix
            cov = LogLike.beta[i,j]**2 * Cov[i,j].cov
            if Cov[i,j].is_matrix:
                if Cov[i,j].cholesky_mode == 'sparse':
                    cov = cov.todense()
            else:
                cov = np.diag(cov)

            # Take the mean along each diagonal
            ks = np.arange(0, len(cov), 1)
            collapsed_cov = np.zeros_like(ks, dtype=float)
            collapsed_delta_wave = np.zeros_like(ks, dtype=float)

            for l, k in enumerate(ks):
                collapsed_cov[l] = np.mean(np.diag(cov, k))
                collapsed_delta_wave[l] = np.mean(np.diag(delta_wave_ij, k))

            collapsed_cov = np.concatenate(
                (collapsed_cov[::-1][:-1], collapsed_cov)
                )
            collapsed_delta_wave = np.concatenate(
                (-collapsed_delta_wave[::-1][:-1], collapsed_delta_wave)
                )

            collapsed_cov *= mask_ij.sum()

            ax_delta_wave_ij = ax[i,j].twiny()
            ax_delta_wave_ij.plot(
                collapsed_delta_wave, 
                collapsed_cov*np.nanmax(ACF[i,j])/np.nanmax(collapsed_cov), 
                c=bestfit_color, lw=1
                )
            ax_delta_wave_ij.set(
                xlim=(wave_ij.mean() * rv.min()/(nc.c*1e-5),
                      wave_ij.mean() * rv.max()/(nc.c*1e-5))
                )

            if (i == 0) and (j == 1):
                ax_delta_wave_ij.set(
                    xlabel=r'$\Delta\lambda\ \mathrm{(nm)}$', 
                    )

    ax[n_orders//2,0].set(ylabel='Auto-correlation')
    ax[-1,1].set(xlabel=r'$v_\mathrm{rad}\ \mathrm{(km\ s^{-1})}$')

    if prefix is not None:
        fig.savefig(prefix+'plots/auto_correlation_residuals.pdf')

    #plt.show()
    plt.close(fig)

def plot_ax_CCF(ax, 
                d_spec, 
                m_spec, 
                pRT_atm, 
                m_spec_wo_species=None, 
                pRT_atm_wo_species=None, 
                LogLike=None, 
                Cov=None, 
                rv=np.arange(-1000,1000+1e-6,10), 
                rv_to_exclude=(-100,100), 
                color='k', 
                label=None
                ):

    if pRT_atm_wo_species is not None:
        pRT_atm_wo_species_flux_pRT_grid = pRT_atm_wo_species.flux_pRT_grid_only.copy()
    else:
        pRT_atm_wo_species_flux_pRT_grid = None

    rv, CCF, d_ACF, m_ACF = af.CCF(
        d_spec=d_spec, 
        m_spec=m_spec, 
        m_wave_pRT_grid=pRT_atm.wave_pRT_grid.copy(), 
        m_flux_pRT_grid=pRT_atm.flux_pRT_grid.copy(), 
        m_spec_wo_species=m_spec_wo_species, 
        m_flux_wo_species_pRT_grid=pRT_atm_wo_species_flux_pRT_grid, 
        LogLike=LogLike, 
        Cov=Cov, 
        rv=rv, 
        )

    # Convert to signal-to-noise functions
    CCF_SNR, m_ACF_SNR, excluded_CCF_SNR = af.CCF_to_SNR(
        rv, CCF.sum(axis=(0,1)), ACF=m_ACF.sum(axis=(0,1)), rv_to_exclude=rv_to_exclude
        )

    ax.axvline(0, lw=1, c='k', alpha=0.2)
    ax.axhline(0, lw=0.1, c='k')
    ax.axvspan(
        rv_to_exclude[0], rv_to_exclude[1], 
        color='k', alpha=0.1, ec='none'
        )

    # Plot the cross- and auto-correlation signal-to-noises
    ax.plot(rv, m_ACF_SNR, c=color, lw=1, ls='--', alpha=0.5)
    ax.plot(rv, CCF_SNR, c=color, lw=1, label=label)
    #ax.plot(rv, excluded_CCF_SNR, c=color, lw=0.5, ls='-', alpha=0.5)
    #ax.fill_between(rv, y1=CCF_SNR, y2=excluded_CCF_SNR, color=color, alpha=0.5)

    SNR_rv0 = CCF_SNR[rv==0][0]
    ax.annotate(
        r'S/N$=$'+'{:.1f}'.format(SNR_rv0), xy=(0, SNR_rv0), 
        xytext=(0.55, 0.7), textcoords=ax.transAxes, color=color, 
        arrowprops={'arrowstyle':'-', 'lw':0.5, 
                    'color':color, 'alpha':0.5
                    }
        )

    ax.legend(
        loc='upper right', handlelength=1, framealpha=0.7, 
        handletextpad=0.5, columnspacing=0.8
        )
    ax.set(ylim=(min([1.2*CCF_SNR.min(), 1.2*m_ACF_SNR.min(), -5]), 
                 max([1.2*CCF_SNR.max(), 1.2*m_ACF_SNR.max(), 7])
                 )
           )
    return ax

def fig_species_contribution(d_spec, 
                             m_spec, 
                             m_spec_species, 
                             pRT_atm, 
                             pRT_atm_species, 
                             Chem, 
                             LogLike, 
                             Cov, 
                             species_to_plot, 
                             rv_CCF=np.arange(-1000,1000+1e-6,5), 
                             rv_to_exclude=(-100,100), 
                             bin_size=25, 
                             prefix=None
                             ):

    if not os.path.exists(prefix+'plots/species'):
        os.makedirs(prefix+'plots/species')

    fig_CCF, ax_CCF = plt.subplots(
        figsize=(5,2*(len(species_to_plot)+1)), 
        nrows=len(species_to_plot)+1, 
        sharex=True, sharey=False, 
        gridspec_kw={
            'hspace':0.05, 'left':0.13, 'right':0.95, 
            'top':0.97, 'bottom':0.05
            }
        )
    h = 0

    # Plot the cross-correlation of the complete model
    plot_ax_CCF(
        ax_CCF[0], 
        d_spec, 
        m_spec, 
        pRT_atm, 
        LogLike=LogLike, 
        Cov=Cov, 
        rv=rv_CCF, 
        rv_to_exclude=rv_to_exclude, 
        color='k', 
        label='Complete'
        )
        
    for species_h in Chem.species_info.keys():
        
        if species_h not in species_to_plot:
            continue

        # Check if the line species was included in the model
        line_species_h = Chem.read_species_info(species_h, info_key='pRT_name')
        if line_species_h in Chem.line_species:

            h += 1

            # Read the ModelSpectrum class for this species
            m_spec_h  = m_spec_species[species_h]
            pRT_atm_h = pRT_atm_species[species_h]
            
            # Residual between data and model w/o species_i
            d_res = d_spec.flux/LogLike.f[:,:,None] - m_spec_h.flux

            low_pass_d_res = np.nan * np.ones_like(d_res)
            low_pass_d_res[d_spec.mask_isfinite] = gaussian_filter1d(d_res[d_spec.mask_isfinite], sigma=300)
            d_res -= low_pass_d_res

            # Residual between complete model and model w/o species_i
            #m_res = m_spec.flux - m_spec_h.flux
            m_res = m_spec_h.flux_only

            low_pass_m_res = np.nan * np.ones_like(m_res)
            low_pass_m_res[d_spec.mask_isfinite] = gaussian_filter1d(m_res[d_spec.mask_isfinite], sigma=300)
            m_res -= low_pass_m_res

            # Read the color and label
            color_h = Chem.read_species_info(species_h, info_key='color')
            label_h = Chem.read_species_info(species_h, info_key='label')

            # Plot the cross-correlation of species_h
            plot_ax_CCF(
                ax_CCF[h], 
                d_spec, 
                m_spec, 
                pRT_atm, 
                m_spec_wo_species=m_spec_h, 
                pRT_atm_wo_species=pRT_atm_h, 
                LogLike=LogLike, 
                Cov=Cov, 
                rv=rv_CCF, 
                rv_to_exclude=rv_to_exclude, 
                color=color_h, 
                label=label_h
                )

            # Use a common ylim for all orders
            ylim = (np.nanmean(d_res) - 5*np.nanstd(d_res), 
                    np.nanmean(d_res) + 4*np.nanstd(d_res)
                    )

            fig, ax = fig_order_subplots(
                d_spec.n_orders, 
                ylabel='Residuals\n'+r'$\mathrm{(erg\ s^{-1}\ cm^{-2}\ nm^{-1})}$'
                )

            for i in range(d_spec.n_orders):
                
                ax[i].axhline(0, lw=0.1, c='k')
                
                for j in range(d_spec.n_dets):

                    label = r'$d-m_\mathrm{w/o\ ' + label_h.replace('$', '') + r'}$'
                    if species_h in ['13CO', 'NH3']:
                        alpha = 0.3
                        label_1 = None
                        label_2 = label + f' (binned to {bin_size} pixels)'
                    else:
                        alpha = 1.0
                        label_1 = label

                    ax[i].plot(
                        d_spec.wave[i,j], d_res[i,j], 
                        c='k', lw=0.5, alpha=alpha, label=label_1
                        )
                    
                    if species_h in ['13CO', 'NH3']:
                        mask_ij = d_spec.mask_isfinite[i,j]
                        binned_d_res_ij = np.nan * np.ones_like(d_res[i,j])
                        #binned_d_res_ij[mask_ij] = gaussian_filter1d(d_res[i,j,mask_ij], sigma=bin_size)
                        binned_d_res_ij = generic_filter(d_res[i,j], np.nanmedian, size=bin_size)

                        ax[i].plot(
                            d_spec.wave[i,j], binned_d_res_ij, c='k', lw=0.5, label=label_2
                            )
                    else:
                        pass
                        '''
                        mask_ij = d_spec.mask_isfinite[i,j]
                        low_pass_d_res_ij = np.nan * np.ones_like(d_res[i,j])
                        low_pass_d_res_ij[mask_ij] = gaussian_filter1d(d_res[i,j,mask_ij], sigma=300)
                        ax[i].plot(
                            d_spec.wave[i,j], low_pass_d_res_ij, c='0.5', lw=1
                            )
                        '''

                    ax[i].plot(
                        d_spec.wave[i,j], m_res[i,j], c=color_h, lw=1, 
                        label=r'$m_\mathrm{only\ '+label_h.replace('$', '')+r'}$'
                        )

                    if (i == 0) and (j == 0):
                        ax[i].legend(
                            loc='upper right', ncol=2, fontsize=10, handlelength=1, 
                            framealpha=0.7, handletextpad=0.3, columnspacing=0.8
                            )
                ax[i].set(ylim=ylim)

            if prefix is not None:
                fig.savefig(prefix+f'plots/species/{species_h}_spec.pdf')
            plt.close(fig)

    ax_CCF[-1].set(
        xlabel=r'$v_\mathrm{rad}\ \mathrm{(km\ s^{-1})}$', 
        xlim=(rv_CCF.min(), rv_CCF.max()), 
        #yticks=np.arange(-6,30,3), 
        #ylim=ax_CCF[-1].get_ylim()
        )
    ax_CCF[len(species_to_plot)//2].set(ylabel='S/N')

    if prefix is not None:
        fig_CCF.savefig(prefix+f'plots/species/CCF.pdf')
    plt.close(fig_CCF)