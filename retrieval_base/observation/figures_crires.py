import numpy as np
import matplotlib.pyplot as plt
from .. import utils

def plot_telluric_correction(plots_dir, d_spec):
    """
    Plot the telluric correction.

    Args:
        plots_dir (str): Directory to save the plots.
        d_spec (object): Spectrum object containing data.
    """
    # Plot per order
    fig, subfig = utils.get_subfigures_per_chip(d_spec.n_orders)
    for i, subfig_i in enumerate(subfig):

        xlabel, ylabel = None, (None, None)
        if i == 0:
            xlabel = 'Wavelength (nm)'
            ylabel = (r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 'Transm.')

        # Add some padding
        xlim = (
            d_spec.wave_ranges_orders_dets[i].min()-0.5, 
            d_spec.wave_ranges_orders_dets[i].max()+0.5
            )
        
        gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
        ax_flux   = subfig_i.add_subplot(gs[0])
        ax_transm = subfig_i.add_subplot(gs[1])

        for idx in range(i*d_spec.n_dets, (i+1)*d_spec.n_dets):
            if idx == d_spec.n_chips:
                break
            ax_flux.plot(d_spec.wave[idx], d_spec.uncorrected_flux[idx], 'k-', lw=0.5, alpha=0.4)
            ax_flux.plot(d_spec.wave[idx], d_spec.flux[idx], 'k-', lw=0.7)
            
            ax_transm.plot(d_spec.wave[idx], d_spec.transm_mf[idx], 'k-', lw=0.5)
        
        ax_transm.axhline(d_spec.telluric_threshold, c='r', ls='--')
        
        ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])
        ax_transm.set(xlim=xlim, xlabel=xlabel, ylim=(0,1.1), ylabel=ylabel[1])

    fig.savefig(plots_dir / f'telluric_correction_per_order_{d_spec.m_set}.pdf')
    plt.close(fig)

    # Plot for full spectrum
    fig = plt.figure(figsize=(10,4))
    gs = fig.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
    ax_flux   = fig.add_subplot(gs[0])
    ax_transm = fig.add_subplot(gs[1])

    xlim = (
        d_spec.wave_ranges_orders_dets.min()-15, 
        d_spec.wave_ranges_orders_dets.max()+15
        )

    for idx in range(d_spec.n_chips):
        ax_flux.plot(d_spec.wave[idx], d_spec.uncorrected_flux[idx], 'k-', lw=0.5, alpha=0.4)
        ax_flux.plot(d_spec.wave[idx], d_spec.flux[idx], 'k-', lw=0.7)

        ax_transm.plot(d_spec.wave[idx], d_spec.transm_mf[idx], 'k-', lw=0.5)
    
    ax_transm.axhline(d_spec.telluric_threshold, c='r', ls='--')
    
    ax_flux.set(xlim=xlim, ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$')
    ax_transm.set(xlim=xlim, xlabel='Wavelength (nm)', ylim=(0,1.1), ylabel='Transm.')

    fig.savefig(plots_dir / f'telluric_correction_{d_spec.m_set}.pdf')
    plt.close(fig)

def plot_sigma_clip(plots_dir, d_spec):
    """
    Plot the sigma clipping results for each order.

    Args:
        plots_dir (str): Directory to save the plots.
        d_spec (object): Spectrum object containing data.
    """
    valid_residuals = d_spec.residuals.copy()
    valid_residuals[d_spec.mask_sigma_clipped] = np.nan

    # Plot per order
    fig, subfig = utils.get_subfigures_per_chip(d_spec.n_orders)
    for i, subfig_i in enumerate(subfig):
        
        xlabel, ylabel = None, (None, None)
        if i == 0:
            xlabel = 'Wavelength (nm)'
            ylabel = (r'$F\ (\mathrm{arb.\ units})$', 'Res.')

        # Add some padding
        xlim = (
            d_spec.wave_ranges_orders_dets[i].min()-0.5, 
            d_spec.wave_ranges_orders_dets[i].max()+0.5
            )

        gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
        ax_flux = subfig_i.add_subplot(gs[0])
        ax_res  = subfig_i.add_subplot(gs[1])

        for idx in range(i*d_spec.n_dets, (i+1)*d_spec.n_dets):
            if idx == d_spec.n_chips:
                break
            #ax_flux.plot(d_spec.wave[idx], d_spec.flux[idx], 'k-', lw=0.5)
            ax_flux.plot(d_spec.wave[idx], d_spec.residuals[idx]+d_spec.running_median_flux[idx], 'k-', lw=0.5)
            ax_flux.plot(d_spec.wave[idx], d_spec.running_median_flux[idx], 'r-', lw=0.7)

            ax_res.plot(d_spec.wave[idx], d_spec.residuals[idx], 'r-', lw=0.5)
            ax_res.plot(d_spec.wave[idx], valid_residuals[idx], 'k-', lw=0.7)

            sigma = d_spec.sigma_clip_sigma*np.nanstd(d_spec.residuals[idx])
            ax_res.plot(d_spec.wave[idx], -sigma*np.ones_like(d_spec.wave[idx]), 'r--', lw=0.5)
            ax_res.plot(d_spec.wave[idx], +sigma*np.ones_like(d_spec.wave[idx]), 'r--', lw=0.5)

        ax_res.axhline(0, c='r', ls='-', lw=0.5)

        ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])
        
        ylim = ax_res.get_ylim()
        ylim_max = np.max(np.abs(ylim))
        ylim = (-ylim_max, +ylim_max)
        ax_res.set(xlim=xlim, xlabel=xlabel, ylim=ylim, ylabel=ylabel[1])

    fig.savefig(plots_dir / f'sigma_clipping_{d_spec.m_set}.pdf')
    plt.close(fig)

def plot_spectrum_to_fit(plots_dir, d_spec):
    """
    Plot the spectrum to be fitted for each order.

    Args:
        plots_dir (str): Directory to save the plots.
        d_spec (object): Spectrum object containing data.
    """
    # Plot per order
    fig, subfig = utils.get_subfigures_per_chip(d_spec.n_orders)
    for i, subfig_i in enumerate(subfig):

        xlabel, ylabel = None, None
        if i == 0:
            xlabel = 'Wavelength (nm)'
            ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'

        # Add some padding
        xlim = (
            d_spec.wave_ranges_orders_dets[i].min()-0.5, 
            d_spec.wave_ranges_orders_dets[i].max()+0.5
            )

        gs = subfig_i.add_gridspec(nrows=1)
        ax_flux = subfig_i.add_subplot(gs[0])
        for idx in range(i*d_spec.n_dets, (i+1)*d_spec.n_dets):
            if idx == d_spec.n_chips:
                break
            ax_flux.plot(d_spec.wave[idx], d_spec.flux[idx], 'k-', lw=0.7)

        ax_flux.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel)

    fig.savefig(plots_dir / f'pre_processed_spectrum_{d_spec.m_set}.pdf')
    plt.close(fig)

def plot_bestfit(plots_dir, d_spec, LogLike, **kwargs):
    """
    Plot the best-fit spectrum for each order.

    Args:
        plots_dir (str): Directory to save the plots.
        d_spec (object): Spectrum object containing data.
        LogLike (object): Log-likelihood object containing fit results.
        **kwargs: Additional keyword arguments.
    """
    # Plot per order
    fig, subfig = utils.get_subfigures_per_chip(d_spec.n_orders)
    for i, subfig_i in enumerate(subfig):

        xlabel, ylabel = None, (None, None)
        if i == 0:
            xlabel = 'Wavelength (nm)'
            ylabel = (r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 'Res.')

        # Add some padding
        xlim = (
            d_spec.wave_ranges_orders_dets[i].min()-0.5, 
            d_spec.wave_ranges_orders_dets[i].max()+0.5
            )
        
        gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
        ax_flux = subfig_i.add_subplot(gs[0])
        ax_res  = subfig_i.add_subplot(gs[1])

        for idx in range(i*d_spec.n_dets, (i+1)*d_spec.n_dets):
            if idx == d_spec.n_chips:
                break
            label = None
            if i==0 and idx==i*d_spec.n_dets:
                label = r'$\chi_\mathrm{red}^2=$'+'{:.2f}'.format(LogLike.chi_squared_0_red)

            ax_flux.plot(d_spec.wave[idx], d_spec.flux[idx], 'k-', lw=0.5)

            idx_LogLike = LogLike.indices_per_model_setting[d_spec.m_set][idx]
            ax_flux.plot(d_spec.wave[idx], LogLike.m_flux_phi[idx_LogLike], 'C1-', lw=0.8, label=label)

            ax_res.plot(d_spec.wave[idx], d_spec.flux[idx]-LogLike.m_flux_phi[idx_LogLike], 'k-', lw=0.8)
            ax_res.axhline(0, c='C1', ls='-', lw=0.5)

        ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])

        ylim = ax_res.get_ylim()
        ylim_max = np.max(np.abs(ylim))
        ylim = (-ylim_max, +ylim_max)
        ax_res.set(xlim=xlim, xlabel=xlabel, ylim=ylim, ylabel=ylabel[1])

        if i == 0:
            ax_flux.legend()

    if LogLike.sum_model_settings:
        fig.savefig(plots_dir / f'bestfit_spectrum.pdf')
    else:
        fig.savefig(plots_dir / f'bestfit_spectrum_{d_spec.m_set}.pdf')
    plt.close(fig)