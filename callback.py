import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import time
import json
import corner

import auxiliary_functions as af

class CallBack:

    def __init__(
        self, 
        d_spec, 
        cb_count=0, 
        evaluation=False, 
        n_samples_to_use=5000, 
        prefix=None, 
        posterior_color='C0', 
        bestfit_color='C1', 
        ):
        
        self.elapsed_times = []

        self.active = False
        self.cb_count = cb_count

        self.evaluation = evaluation
        self.n_samples_to_use = n_samples_to_use

        self.d_spec = d_spec
        self.prefix = prefix

        self.posterior_color = posterior_color
        self.bestfit_color = bestfit_color

        envelope_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            name='envelope_cmap', 
            colors=['w', self.posterior_color], 
            )
        self.envelope_colors = envelope_cmap([0.0,0.2,0.4,0.6,0.8])
        self.envelope_colors[0,-1] = 0.0

    def __call__(self, Param, LogLike, PT, Chem, m_spec, pRT_atm, posterior):

        # Display the mean elapsed time per lnL evaluation
        print('\nElapsed time per evaluation: {:.2f} seconds'.format(np.mean(self.elapsed_times)))
        self.elapsed_times.clear()

        time_A = time.time()
        self.cb_count += 1

        # Make attributes for convenience
        self.Param   = Param
        self.LogLike = LogLike
        self.PT      = PT
        self.Chem    = Chem
        self.m_spec  = m_spec
        self.pRT_atm = pRT_atm

        self.posterior = posterior

        # Compute the 0.16, 0.5, and 0.84 quantiles
        self.param_quantiles = np.array([af.quantiles(self.posterior[:,i], q=[0.16,0.5,0.84]) \
                                         for i in range(self.posterior.shape[1])])
        # Base the axes-limits off of the quantiles
        self.param_range = [(4*(q_i[0]-q_i[1])+q_i[1], 4*(q_i[2]-q_i[1])+q_i[1]) \
                            for q_i in self.param_quantiles]

        # Create the labels
        self.param_labels = list(self.Param.param_mathtext.values())

        self.bestfit_params = [self.Param.params[key_i] \
                               for key_i in self.Param.param_keys]
        self.median_params  = list(self.param_quantiles[:,1])

        # Make a summary figure
        self.fig_summary()

        # Remove attributes from memory
        del self.Param, self.LogLike, self.PT, self.Chem, self.m_spec, self.pRT_atm

        time_B = time.time()
        print('\nPlotting took {:.0f} seconds\n'.format(time_B-time_A))

    def save_bestfit(self):
        
        # Save the bestfit parameters
        dict_to_save = {
            'params': self.Param.params, 
            'f': self.LogLike.f, 
            'beta': self.LogLike.beta, 
            'temperature': self.PT.temperature, 
            'pressure': self.PT.pressure, 
        }

        with open(prefix+'data/bestfit.json', 'w') as fp:
            json.dump(dict_to_save, fp, indent=4)

        # Save the bestfit spectrum
        pickle_save(self.prefix+'data/bestfit_m_spec.pkl', self.m_spec)
        #np.save(prefix+'data/bestfit_flux.npy', self.m_spec.flux)
        
    def fig_summary(self):

        fig = plt.figure(figsize=(15,15))
        fig = corner.corner(
            self.posterior, 
            fig=fig, 
            quiet=True, 

            labels=self.param_labels, 
            show_titles=True, 
            use_math_text=True, 
            title_fmt='.2f', 
            title_kwargs={'fontsize':9}, 
            labelpad=0.25*self.Param.n_params/17, 
            range=self.param_range, 
            bins=20, 
            max_n_ticks=3, 

            quantiles=[0.16,0.84], 
            color=self.posterior_color, 
            linewidths=0.5, 
            hist_kwargs={'color':self.posterior_color}, 

            #levels=(1-np.exp(-0.5),),
            fill_contours=True, 
            #plot_datapoints=False, 

            contourf_kwargs={'colors':self.envelope_colors}, 
            #smooth=True, 

            contour_kwargs={'linewidths':0.5}, 
            )

        # Add the best-fit and median values as lines
        corner.overplot_lines(fig, self.bestfit_params, c=self.bestfit_color, lw=0.5)
        corner.overplot_lines(fig, self.median_params, c=self.posterior_color, lw=0.5)

        # Reshape the axes to a square matrix
        ax = np.array(fig.axes)
        ax = ax.reshape((int(np.sqrt(len(ax))), 
                         int(np.sqrt(len(ax))))
                        )

        for i in range(ax.shape[0]):
            # Change linestyle of 16/84th percentile in histograms
            ax[i,i].get_lines()[0].set(linewidth=0.5, linestyle=(5,(5,5)))
            ax[i,i].get_lines()[1].set(linewidth=0.5, linestyle=(5,(5,5)))

            # Show the best-fitting value in histograms
            ax[i,i].annotate(
                r'$'+'{:.2f}'.format(self.bestfit_params[i])+'$', 
                xy=(1,0.95), xycoords=ax[i,i].transAxes, 
                color=self.bestfit_color, rotation=0, ha='right', va='top', 
                fontsize=9
                )
            # Adjust the axis-limits
            for j in range(i):
                ax[i,j].set(ylim=self.param_range[i])
            for h in range(ax.shape[0]):
                ax[h,i].set(xlim=self.param_range[i])
            
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)

        # Plot the best-fitting spectrum
        ax_spec = fig.add_axes([0.4,0.8,0.55,0.15])
        l, b, w, h = ax_spec.get_position().bounds
        ax_res = fig.add_axes([l,b-h/3,w,h/3])

        ax_spec, ax_res = self.fig_spectrum(ax_spec, ax_res)

        # Plot the VMR per species
        ax_VMR = fig.add_axes([0.63,0.475,0.1,0.22])
        ax_VMR = self.fig_VMR(ax_VMR)

        # Plot the best-fitting PT profile
        l, b, w, h = ax_VMR.get_position().bounds
        ax_PT = fig.add_axes([l+w,b,h,h])
        ax_PT = self.fig_PT(ax_PT, ylabel=None, yticks=[])

        #plt.show()
        plt.savefig(self.prefix+'plots/live_summary.pdf')
        #plt.savefig(self.prefix+'plots/live_summary.png', dpi=100)
        plt.close()
        #corner.overplot_lines(fig, )

    def fig_spectrum(
        self, 
        ax_spec, 
        ax_res, 
        ):

        # Loop over the orders and detectors
        for i in range(self.d_spec.n_orders):
            for j in range(self.d_spec.n_dets):
                
                # Plot the observed and model spectra
                ax_spec.plot(
                    self.d_spec.wave[i,j], self.d_spec.flux[i,j], 
                    c='k', lw=0.5, label='Observation'
                    )

                '''
                label = 'Best-fit model\n' + \
                        r'$(\chi^2_\mathrm{red}=' + '{:.2f}'.format(self.LogLike.chi_squared_red) + \
                        r',\ \chi^2=' + '{:.2f}'.format(self.LogLike.chi_squared) + \
                        r')$'
                '''
                label = 'Best-fit model ' + \
                        r'$(\chi^2_\mathrm{red}$ (w/o $\sigma$-model)$=' + \
                        '{:.2f}'.format(self.LogLike.chi_squared_red) + \
                        r')$'
                ax_spec.plot(
                    self.d_spec.wave[i,j], self.LogLike.f[i,j]*self.m_spec.flux[i,j], 
                    c=self.bestfit_color, lw=1, 
                    label=label
                    )

                res_ij = self.d_spec.flux[i,j] - self.LogLike.f[i,j]*self.m_spec.flux[i,j]
                ax_res.plot(self.d_spec.wave[i,j], res_ij, c='k', lw=0.5)
                ax_res.plot(
                    [self.d_spec.wave[i,j].min(), self.d_spec.wave[i,j].max()], 
                    [0,0], c=self.bestfit_color, lw=1
                    )

                # Show the mean error
                mean_err_ij = np.mean(self.d_spec.err[i,j][self.d_spec.mask_isfinite[i,j]])
                ax_res.errorbar(
                    self.d_spec.wave[i,j].min()-0.2, 0, yerr=1*mean_err_ij, 
                    fmt='none', lw=1, ecolor='k', capsize=2, color='k', label=r'$\langle\sigma_{ij}\rangle$'
                    )

                mean_scaled_err_ij = self.LogLike.beta[i,j]*mean_err_ij
                ax_res.errorbar(
                    self.d_spec.wave[i,j].min()-0.4, 0, yerr=1*mean_scaled_err_ij, 
                    fmt='none', lw=1, ecolor=self.bestfit_color, capsize=2, color=self.bestfit_color, 
                    label=r'$\beta_{ij}\langle\sigma_{ij}\rangle$'
                    )

                if i==0 and j==0:
                    ax_spec.legend(loc='upper right', ncol=2, handlelength=1, framealpha=0.7)
                    ax_res.legend(loc='upper right', ncol=2, handlelength=1, framealpha=0.7)

        
        ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'
        if self.d_spec.high_pass_filtered:
            ylabel = r'$F_\lambda$ (high-pass filtered)'

        ylim = (np.nanmean(self.d_spec.flux)-4*np.nanstd(self.d_spec.flux), 
                np.nanmean(self.d_spec.flux)+4*np.nanstd(self.d_spec.flux)
                )

        ax_spec.set(
            xticklabels=[], 
            xlim=(self.d_spec.wave.min()-2, self.d_spec.wave.max()+2), 
            ylabel=ylabel, 
            ylim=ylim, 
            )


        ylim = (1/3*(ylim[0]-np.nanmean(self.d_spec.flux)), 
                1/3*(ylim[1]-np.nanmean(self.d_spec.flux)))

        ax_res.set(
            xlabel='Wavelength (nm)', 
            xlim=(self.d_spec.wave.min()-2, self.d_spec.wave.max()+2), 
            ylabel='Residuals', 
            ylim=ylim, 
            )

        return ax_spec, ax_res

    def fig_PT(self, ax_PT, ylabel=r'$P\ \mathrm{(bar)}$', yticks=None):

        # Plot the best-fitting PT profile and median
        ax_PT.plot(self.PT.temperature, self.PT.pressure, c=self.bestfit_color, lw=1)
        ax_PT.plot(
            self.PT.T_knots, self.PT.P_knots, 
            c=self.bestfit_color, ls='', marker='o', markersize=3
            )
        
        if yticks is None:
            yticks = np.logspace(-6, 2, 9)
            
        ax_PT.set(
            xlabel=r'$T\ \mathrm{(K)}$', xlim=(1,3500), 
            ylabel=ylabel, ylim=(self.PT.pressure.min(), self.PT.pressure.max()), 
            yscale='log', yticks=yticks
            )
        ax_PT.invert_yaxis()

        return ax_PT

    def fig_VMR(self, ax_VMR, ylabel=r'$P\ \mathrm{(bar)}$', yticks=None):

        MMW = self.Chem.mass_fractions['MMW']

        for species_i in self.Chem.species_info.keys():
            
            if species_i not in ['12CO', 'H2O', '13CO', 'CH4', 'NH3']:
                continue

            # Check if the line species was included in the model
            line_species_i = self.Chem.read_species_info(species_i, info_key='pRT_name')
            if line_species_i in self.Chem.line_species:

                # Read the mass, color and label
                mass_i  = self.Chem.read_species_info(species_i, info_key='mass')
                color_i = self.Chem.read_species_info(species_i, info_key='color')
                label_i = self.Chem.read_species_info(species_i, info_key='label')

                # Convert the mass fraction to a VMR
                mass_fraction_i = self.Chem.mass_fractions[line_species_i]
                VMR_i = mass_fraction_i * MMW/mass_i

                # Plot volume-mixing ratio as function of pressure
                ax_VMR.plot(VMR_i, self.PT.pressure, c=color_i, lw=1, label=label_i)

        if yticks is None:
            yticks = np.logspace(-6, 2, 9)

        ax_VMR.legend(handlelength=0.5)
        ax_VMR.set(
            xlabel='VMR', xscale='log', xlim=(1e-8, 1e-2), 
            ylabel=ylabel, yscale='log', yticks=yticks, 
            ylim=(self.PT.pressure.min(), self.PT.pressure.max()), 
            )
        ax_VMR.invert_yaxis()

        return ax_VMR