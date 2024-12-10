import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

def get_env_colors(color):

    env_cmap = mpl.colors.LinearSegmentedColormap.from_list('', ['w',color])
    env_colors = env_cmap([0.,1./3,2./3,1.])
    env_colors[:,-1] = 0.5
    env_colors[0,-1] = 0.0

    return env_cmap, env_colors

q = 0.5 + 1/2*np.array([-0.997, -0.95, -0.68, 0., 0.68, 0.95, 0.997])

bestfit_color   = 'C1'
posterior_color = 'C0'
env_cmap, env_colors = get_env_colors(posterior_color)


def plot_corner(fig, posterior, bestfit_parameters, labels=None, plot_datapoints=False):

    # Get the range (4*sigma) to plot between
    posterior_range = np.quantile(posterior, q=q[[2,4]], axis=0)
    median = np.median(posterior, axis=0)
    posterior_range = median + 4*(posterior_range-median)
    posterior_range = posterior_range.T
    
    n_params = posterior.shape[1]

    kwargs = dict(
        quiet=True, 

        bins=20, 
        fill_contours=True, 
        plot_datapoints=plot_datapoints,
        labels=labels, 
        labelpad=0.018*n_params, 
        max_n_ticks=3,
        
        show_titles=True, 
        use_math_text=True,
        title_fmt='.2f', 
        quantiles=[0.16,0.84], 
        linewidths=0.5, 

        color=posterior_color, 
        hist_kwargs={'edgecolor':posterior_color, 'facecolor':env_colors[1], 'fill':True},
        contour_kwargs={'linewidths':0.5},#, 'color':posterior_color},
    )

    import corner
    fig = corner.corner(
        fig=fig, 
        data=posterior, 
        range=posterior_range, 
        **kwargs
    )
    corner.overplot_lines(fig, bestfit_parameters, c=bestfit_color, lw=0.5)
    corner.overplot_lines(fig, median, c=posterior_color, lw=0.5)

    ax = np.array(fig.axes)
    ax = ax.reshape((n_params, n_params))

    for i in range(n_params):
        # Change linestyle of 16/84th percentile in histograms
        ax[i,i].get_lines()[0].set(linewidth=0.5, linestyle=(5,(5,5)))
        ax[i,i].get_lines()[1].set(linewidth=0.5, linestyle=(5,(5,5)))

        ax[i,i].set_title(ax[i,i].get_title().replace('=','=\n'), fontsize=9)

        # Show the best-fit value
        ax[i,i].annotate(
            r'$'+'{:.2f}'.format(bestfit_parameters[i])+'$', 
            xy=(0.95,0.95), xycoords='axes fraction', 
            ha='right', va='top', fontsize=9, color=bestfit_color
            )
        
        for j in range(n_params):
            xlim = posterior_range[j]
            ylim = posterior_range[i]
            if i == j:
                ylim = None # Diagonal (1d histogram)
            ax[i,j].set(xlim=xlim, ylim=ylim)
            ax[i,j].tick_params(top=True, right=True, bottom=True, left=True, direction='inout')

    margin = 0.06
    fig.subplots_adjust(
        wspace=0., hspace=0., left=margin, bottom=margin, top=1-margin, right=1-margin
        )
    #plt.subplots_adjust()
    return fig, ax


def plot_envelopes(ax, y, env_x, indices=[(0,6),(1,5),(2,4)], colors=env_colors[:3], median_kwargs={'c':posterior_color}, **kwargs):
    
    for i, (idx_l, idx_u) in enumerate(indices):
        ax.fill_betweenx(
            y=y, x1=env_x[idx_l], x2=env_x[idx_u], fc=colors[i], 
            ec='none', **kwargs
            )

    if median_kwargs is None:
        return
    ax.plot(env_x[3], y, **median_kwargs)

def plot_PT(PT, ax=None, ls='-'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    
    ax.set(
        yscale='log', ylabel='P (bar)', ylim=(PT.pressure.max(), PT.pressure.min()), 
        xlabel='T (K)', xlim=(0,3900)
        )
    
    temperature_posterior = getattr(PT, 'temperature_posterior', None)
    if temperature_posterior is not None:
        env = np.quantile(temperature_posterior, q=q, axis=0)
        plot_envelopes(
            ax, y=PT.pressure, env_x=env, median_kwargs={'c':posterior_color, 'ls':ls}
            )

    ax.plot(PT.temperature, PT.pressure, c=bestfit_color, ls=ls)
    
    log_P_knots = getattr(PT, 'log_P_knots', [])
    for log_P in log_P_knots:
        ax.axhline(10**log_P, xmax=0.02, c=bestfit_color)
        ax.axhline(10**log_P, xmin=1-0.02, c=bestfit_color)

def plot_gradient(PT, ax=None, ls='-'):
    if ax is None:
        return

    ax.set(
        ylim=(PT.pressure.max(), PT.pressure.min()), yscale='log', yticks=[],
        xlim=(0,0.4), xticks=[0,0.2,0.4], xlabel=r'$\nabla_T$'
        )
        
    ax.set_facecolor('none')
    ax.spines[['left','top','bottom']].set_visible(False)
    ax.spines['right'].set_alpha(0.2)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    temperature_posterior = getattr(PT, 'temperature_posterior', None)
    if temperature_posterior is not None:
        dlnT_dlnP_posterior, y = PT.get_dlnT_dlnP(temperature_posterior, PT.pressure)
        env = np.quantile(dlnT_dlnP_posterior, q=q, axis=0)
        
        gradient_env_colors = env_colors[:3].copy()
        gradient_env_colors[:,-1] = 0.25
        plot_envelopes(
            ax, y=y, env_x=env, median_kwargs={'c':posterior_color, 'ls':ls, 'alpha':0.5}, 
            colors=gradient_env_colors, zorder=-10
            )
        
    dlnT_dlnP, y = PT.get_dlnT_dlnP(PT.temperature, PT.pressure)
    ax.plot(dlnT_dlnP, y, c=bestfit_color, ls=ls, alpha=0.5)

def plot_contribution(m_spec, ax=None, ls='-'):

    ax.set(
        ylim=(m_spec.pressure.max(), m_spec.pressure.min()), 
        yscale='log', yticks=[], xlim=(1,0), xticks=[], 
        )
    ax.set_facecolor('none')
    ax.spines[['left','right','top','bottom']].set_visible(False)

    integrated_contr = np.nansum(m_spec.integrated_contr, axis=0)

    ax.plot(
        integrated_contr/integrated_contr.max(), m_spec.pressure, 
        c=bestfit_color, ls=ls, alpha=0.5
        )


def plot_chemistry(Chem, ax=None, ls='-'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    ax.set(
        yscale='log', ylabel='P (bar)', ylim=(Chem.pressure.max(), Chem.pressure.min()), 
        xscale='log', xlabel='VMR', xlim=(1e-10,1e-2)
        )

    for species_i, VMR_i in Chem.VMRs.items():
        if species_i in ['H2', 'He']:
            continue
        color = Chem.read_species_info(species_i, 'color')
        label = Chem.read_species_info(species_i, 'label')

        VMRs_posterior = getattr(Chem, 'VMRs_posterior', None)
        if VMRs_posterior is not None:
            env = np.quantile(VMRs_posterior[species_i], q=q, axis=0)
            _, VMR_env_colors = get_env_colors(color=color)
            plot_envelopes(
                ax, y=Chem.pressure, env_x=env, indices=[(2,4)], colors=[VMR_env_colors[2]], 
                median_kwargs={'c':color, 'label':label, 'ls':ls}
                )

            continue # Skip the best-fit
    
        ax.plot(VMR_i, Chem.pressure, c=color, ls=ls, label=label)

    ax.legend(
        loc='upper right', bbox_to_anchor=(-0.28,1), frameon=False, 
        handlelength=0.8, labelcolor='linecolor', fontsize=11, markerfirst=False
        )


def plot_clouds(Cloud, ax=None, ls='-'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    ax.set(
        yscale='log', ylabel='P (bar)', ylim=(Cloud.pressure.max(), Cloud.pressure.min()), 
        xscale='log', xlabel=r'$\kappa_\mathrm{cl}\ (\mathrm{cm^2\ g^{-1}})$', 
        xlim=(3e2,1e-4), xticks=[1e2,1e-1,1e-4], 
        )

    total_opacity = getattr(Cloud, 'total_opacity', None)
    if total_opacity is None:
        return
    
    total_opacity_posterior = getattr(Cloud, 'total_opacity_posterior', None)
    if total_opacity_posterior is not None:
        env = np.quantile(total_opacity_posterior, q=q, axis=0)
        plot_envelopes(
            ax, y=Cloud.pressure, env_x=env, median_kwargs={'c':posterior_color, 'ls':ls}
            )
        
    ax.plot(total_opacity, Cloud.pressure, c=bestfit_color, ls=ls)    


def plot_summary(plots_dir, posterior, bestfit_parameters, labels, PT, Chem, Cloud, m_spec, evaluation=False):

    fig = plt.figure(figsize=(13,13))
    fig, ax = plot_corner(
        fig, posterior, bestfit_parameters, labels=labels, plot_datapoints=evaluation
        )

    width, height = 0.5, 0.3
    widths  = np.array([(width-height)*0.7,height,(width-height)*0.3])
    right, top = 0.97, 0.95
    
    ax_cl  = fig.add_axes([right-widths[2],top-height,widths[2],height])
    ax_PT  = fig.add_axes([right-widths[2]-widths[1],top-height,widths[1],height])
    ax_VMR = fig.add_axes([right-width, top-height, widths[0], height])

    ax_gradient = fig.add_axes([right-widths[2]-widths[1],top-height,widths[1]/3,height])
    ax_contr    = fig.add_axes([right-widths[2]-widths[1]/3,top-height,widths[1]/3,height])

    for m_set, ls in zip(PT.keys(), ['-', '--', ':']):
        plot_contribution(m_spec[m_set], ax=ax_contr, ls=ls)
        plot_gradient(PT[m_set], ax=ax_gradient, ls=ls)
        plot_PT(PT[m_set], ax=ax_PT, ls=ls)        

        plot_chemistry(Chem[m_set], ax=ax_VMR, ls=ls)
        plot_clouds(Cloud[m_set], ax=ax_cl, ls=ls)

    for i, ax_i in enumerate([ax_VMR, ax_cl, ax_PT]):
        ax_i.tick_params(right=True, bottom=True, left=True, direction='inout', which='both')
        if i!=0:
            ax_i.set(yticklabels=[])

    for ax_i in [ax_gradient, ax_contr]:
        ax_i.tick_params(left=False, right=False, which='both')

    # Save the figure to a pdf and png
    file_pdf = plots_dir / 'live_summary.pdf'
    if evaluation:
        file_pdf = plots_dir / 'final_summary.pdf'
    fig.savefig(file_pdf)

    file_png = file_pdf.with_suffix('.png')
    fig.savefig(file_png, dpi=100)
    if evaluation:
        plt.close(fig)
        return

    # Make a copy to show the progress over time
    previous_pngs = list(plots_dir.glob('live_summary_*.png'))
    new_file_png = plots_dir / f'live_summary_{len(previous_pngs)}.png'
    new_file_png.write_bytes(file_png.read_bytes())

    plt.close(fig)