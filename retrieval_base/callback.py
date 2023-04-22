import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import os
import time
import json
import corner

import retrieval_base.auxiliary_functions as af
import retrieval_base.figures as figs

class CallBack:

    def __init__(self, 
                 d_spec, 
                 evaluation=False, 
                 n_samples_to_use=2000, 
                 prefix=None, 
                 posterior_color='C0', 
                 bestfit_color='C1', 
                 species_to_plot=['12CO', 'H2O', '13CO', 'CH4', 'NH3'], 
                 ):
        
        self.elapsed_times = []
        self.active = False
        self.return_PT_mf = False

        self.evaluation = evaluation
        self.cb_count = 0
        if self.evaluation:
            self.cb_count = -2

        self.n_samples_to_use = n_samples_to_use

        self.d_spec = d_spec
        self.prefix = prefix

        self.posterior_color = posterior_color
        self.bestfit_color = bestfit_color

        self.envelope_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            name='envelope_cmap', 
            colors=['w', self.posterior_color], 
            )
        self.envelope_colors = self.envelope_cmap([0.0,0.2,0.4,0.6,0.8])
        self.envelope_colors[0,-1] = 0.0

        self.species_to_plot = species_to_plot

    def __call__(self, 
                 Param, 
                 LogLike, 
                 PT, 
                 Chem, 
                 m_spec, 
                 pRT_atm, 
                 posterior, 
                 temperature_envelopes=None, 
                 mass_fractions_envelopes=None, 
                 ):

        time_A = time.time()
        self.cb_count += 1

        # Make attributes for convenience
        self.Param   = Param
        self.LogLike = LogLike
        self.PT      = PT
        self.Chem    = Chem
        self.m_spec  = m_spec
        self.pRT_atm = pRT_atm

        self.temperature_envelopes    = temperature_envelopes
        self.mass_fractions_envelopes = mass_fractions_envelopes

        if not self.evaluation:
            # Use only the last n samples to plot the posterior
            n_samples = min([len(posterior), self.n_samples_to_use])
            self.posterior = posterior[-n_samples:]
        else:
            self.posterior = posterior

        # Display the mean elapsed time per lnL evaluation
        print('\n\nElapsed time per evaluation: {:.2f} seconds'.format(np.mean(self.elapsed_times)))
        self.elapsed_times.clear()

        # Compute the 0.16, 0.5, and 0.84 quantiles
        self.param_quantiles = np.array([af.quantiles(self.posterior[:,i], q=[0.16,0.5,0.84]) \
                                         for i in range(self.posterior.shape[1])])
        # Base the axes-limits off of the quantiles
        self.param_range = np.array(
            [(4*(q_i[0]-q_i[1])+q_i[1], 4*(q_i[2]-q_i[1])+q_i[1]) for q_i in self.param_quantiles]
            )

        # Create the labels
        self.param_labels = np.array(list(self.Param.param_mathtext.values()))

        print('\nReduced chi-squared (w/o uncertainty-model) = {:.2f}\n(chi-squared={:.2f}, n_dof={:.0f})'.format(
            self.LogLike.chi_squared_red, self.LogLike.chi_squared, self.LogLike.n_dof
            ))

        # Read the best-fitting free parameters
        self.bestfit_params = []
        print('\nBest-fitting free parameters:')
        for key_i in self.Param.param_keys:
            print('{} = {:.2f}'.format(key_i, self.Param.params[key_i]))
            self.bestfit_params.append(self.Param.params[key_i])

        if self.LogLike.scale_flux:
            print('\nOptimal flux-scaling parameters:')
            print(self.LogLike.f.round(2))
        if self.LogLike.scale_err:
            print('\nOptimal uncertainty-scaling parameters:')
            print(self.LogLike.beta.round(2))
        
        self.bestfit_params = np.array(self.bestfit_params)
        self.median_params  = np.array(list(self.param_quantiles[:,1]))

        # Plot the covariance matrices
        figs.fig_cov(
            LogLike=self.LogLike, d_spec=self.d_spec, 
            cmap=self.envelope_cmap, prefix=self.prefix
            )

        # Make a summary figure
        self.fig_summary()

        # Save the bestfit parameters in a .json file
        # and the ModelSpectrum instance as .pkl
        self.save_bestfit()

        # Plot the abundances
        if self.Param.chem_mode == 'free':
            included_params = ['log_12CO', 'log_H2O', 'log_CH4', 'log_C_ratio']
            figsize = (8,8)
        elif self.Param.chem_mode == 'eqchem':
            included_params = ['C/O', 'Fe/H', 'log_C_ratio']
            figsize = (7,7)
        fig, ax = self.fig_corner(
            included_params=included_params, 
            fig=plt.figure(figsize=figsize), 
            smooth=False, ann_fs=10
            )
        plt.subplots_adjust(left=0.11, right=0.95, top=0.95, bottom=0.11, wspace=0, hspace=0)
        fig.savefig(self.prefix+'plots/abundances.pdf')
        plt.close(fig)

        # Remove attributes from memory
        del self.Param, self.LogLike, self.PT, self.Chem, self.m_spec, self.pRT_atm, self.posterior

        time_B = time.time()
        print('\nPlotting took {:.0f} seconds\n'.format(time_B-time_A))

    def save_bestfit(self):
        
        # Save the bestfit parameters
        params_to_save = {}
        for key_i, val_i in self.Param.params.items():
            if isinstance(val_i, np.ndarray):
                val_i = val_i.tolist()
            params_to_save[key_i] = val_i

        dict_to_save = {
            'params': params_to_save, 
            'f': self.LogLike.f.tolist(), 
            'beta': self.LogLike.beta.tolist(), 
            'temperature': self.PT.temperature.tolist(), 
            'pressure': self.PT.pressure.tolist(), 
        }

        with open(self.prefix+'data/bestfit.json', 'w') as fp:
            json.dump(dict_to_save, fp, indent=4)

        # Save the bestfit spectrum
        af.pickle_save(self.prefix+'data/bestfit_m_spec.pkl', self.m_spec)
        
    def fig_corner(self, included_params=None, fig=None, smooth=False, ann_fs=9):
        
        if fig is None:
            fig = plt.figure(figsize=(15,15))
        
        # Only select the included parameters
        mask_params = np.ones(len(self.Param.param_keys), dtype=np.bool)
        if included_params is not None:
            mask_params = np.isin(
                self.Param.param_keys, test_elements=included_params
                )
        
        # Number of parameters
        n_params = mask_params.sum()

        fig = corner.corner(
            self.posterior[:,mask_params], 
            fig=fig, 
            quiet=True, 

            labels=self.param_labels[mask_params], 
            show_titles=True, 
            use_math_text=True, 
            title_fmt='.2f', 
            title_kwargs={'fontsize':9}, 
            labelpad=0.25*n_params/17, 
            range=self.param_range[mask_params], 
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
            smooth=smooth, 

            contour_kwargs={'linewidths':0.5}, 
            )

        # Add the best-fit and median values as lines
        corner.overplot_lines(fig, self.bestfit_params[mask_params], c=self.bestfit_color, lw=0.5)
        corner.overplot_lines(fig, self.median_params[mask_params], c=self.posterior_color, lw=0.5)

        # Reshape the axes to a square matrix
        ax = np.array(fig.axes)
        for ax_i in ax:
            ax_i.tick_params(axis='both', direction='inout')

        ax = ax.reshape((int(np.sqrt(len(ax))), 
                         int(np.sqrt(len(ax))))
                        )

        for i in range(ax.shape[0]):
            # Change linestyle of 16/84th percentile in histograms
            ax[i,i].get_lines()[0].set(linewidth=0.5, linestyle=(5,(5,5)))
            ax[i,i].get_lines()[1].set(linewidth=0.5, linestyle=(5,(5,5)))

            # Show the best-fitting value in histograms
            ax[i,i].annotate(
                r'$'+'{:.2f}'.format(self.bestfit_params[mask_params][i])+'$', 
                xy=(0.95,0.95), xycoords=ax[i,i].transAxes, 
                color=self.bestfit_color, rotation=0, ha='right', va='top', 
                fontsize=ann_fs
                )
            # Adjust the axis-limits
            for j in range(i):
                ax[i,j].set(ylim=self.param_range[mask_params][i])
            for h in range(ax.shape[0]):
                ax[h,i].set(xlim=self.param_range[mask_params][i])
            
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)

        return fig, ax

    def fig_summary(self):

        fig, ax = self.fig_corner()

        # Plot the best-fitting spectrum
        ax_spec = fig.add_axes([0.4,0.8,0.55,0.15])
        l, b, w, h = ax_spec.get_position().bounds
        ax_res = fig.add_axes([l,b-h/3,w,h/3])

        ax_spec, ax_res = figs.fig_bestfit_model(
            d_spec=self.d_spec, 
            m_spec=self.m_spec, 
            LogLike=self.LogLike, 
            bestfit_color=self.bestfit_color, 
            ax_spec=ax_spec, 
            ax_res=ax_res, 
            prefix=self.prefix, 
            )

        ax_VMR = fig.add_axes([0.63,0.475,0.1,0.22])
        l, b, w, h = ax_VMR.get_position().bounds
        ax_PT = fig.add_axes([l+w,b,h,h])
        ax_contr = ax_PT.twiny()

        # Plot the VMR per species
        ax_VMR = self.fig_VMR(ax_VMR)

        # Plot the best-fitting PT profile
        ax_PT = self.fig_PT(ax_PT, ylabel=None, yticks=[])

        # Plot the integrated emission contribution function
        ax_contr = self.fig_contr_em(ax_contr)

        #plt.show()
        if self.evaluation:
            
            # Make the histograms directory
            if not os.path.exists(self.prefix+'plots/hists'):
                os.makedirs(self.prefix+'plots/hists')

            # Save the histograms separately
            for i in range(ax.shape[0]):

                self.fig_hist(
                    posterior_i=self.posterior[:,i], 
                    param_range_i=self.param_range[i], 
                    param_quantiles_i=self.param_quantiles[i], 
                    param_key_i=self.Param.param_keys[i], 
                    title=ax[i,i].get_title(), 
                    bins=20
                    )

                if self.Param.param_keys[i].startswith('log_'):

                    # Plot the logarithmic parameters as linear as well
                    title = self.param_labels[i].replace('\\log', '')
                    title = title.replace('\\ ', '')
                    title = title + r'$=' + '{:.3f}'.format(10**self.param_quantiles[i][1])
                    title = title + '^{' + '+{:.3f}'.format(10**self.param_quantiles[i][2]-10**self.param_quantiles[i][1]) + '}'
                    title = title + '_{' + '{:.3f}'.format(10**self.param_quantiles[i][1]-10**self.param_quantiles[i][0]) + '}$'

                    self.fig_hist(
                        posterior_i=10**self.posterior[:,i], 
                        param_range_i=(10**self.param_range[i][0], 10**self.param_range[i][1]), 
                        param_quantiles_i=10**self.param_quantiles[i], 
                        param_key_i=self.Param.param_keys[i].replace('log_', ''), 
                        title=title, bins=20
                        )
                                
            fig.savefig(self.prefix+'plots/final_summary.pdf')
            fig.savefig(self.prefix+f'plots/final_summary.png', dpi=100)

        else:
            fig.savefig(self.prefix+'plots/live_summary.pdf')
            fig.savefig(self.prefix+f'plots/live_summary_{self.cb_count}.png', dpi=100)
        plt.close(fig)

        # Plot the best-fit spectrum in subplots
        figs.fig_bestfit_model(
            d_spec=self.d_spec, 
            m_spec=self.m_spec, 
            LogLike=self.LogLike, 
            bestfit_color=self.bestfit_color, 
            ax_spec=None, 
            ax_res=None, 
            prefix=self.prefix, 
            )

    def fig_hist(self, posterior_i, param_range_i, param_quantiles_i, param_key_i, title=None, bins=20):

        fig_hist, ax_hist = plt.subplots(figsize=(3,3))
        ax_hist.hist(
            posterior_i, bins=bins, range=param_range_i, 
            color=self.posterior_color, histtype='step', 
            )
        ax_hist.axvline(param_quantiles_i[0], c=self.posterior_color, ls='--')
        ax_hist.axvline(param_quantiles_i[1], c=self.posterior_color, ls='-')
        ax_hist.axvline(param_quantiles_i[2], c=self.posterior_color, ls='--')

        ax_hist.set(yticks=[], xlim=param_range_i, title=title)
        fig_hist.savefig(self.prefix+f'plots/hists/{param_key_i}.pdf')
        plt.close(fig_hist)

    def fig_PT(self, ax_PT, ylabel=r'$P\ \mathrm{(bar)}$', yticks=None):
            
        if self.evaluation:
            # Plot the PT envelope as well
            for i in range(3):
                ax_PT.fill_betweenx(
                    y=self.PT.pressure, x1=self.temperature_envelopes[i], 
                    x2=self.temperature_envelopes[-i-1], 
                    color=self.envelope_colors[i+1], ec='none', 
                    )
            # Plot the median PT
            ax_PT.plot(
                self.temperature_envelopes[3], self.PT.pressure, 
                c=self.posterior_color, lw=1
            )
        
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
            
            if species_i not in self.species_to_plot:
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

                if self.evaluation:
                    # Plot the VMR envelope as well
                    cmap_i = mpl.colors.LinearSegmentedColormap.from_list(
                        'cmap_i', colors=['w', color_i]
                        )
                    VMR_envelope_colors_i = cmap_i([0.0,0.2,0.4,0.6,0.8])

                    for j in range(3):
                        ax_VMR.fill_betweenx(
                            y=self.PT.pressure, x1=MMW/mass_i * self.mass_fractions_envelopes[line_species_i][j], 
                            x2=MMW/mass_i * self.mass_fractions_envelopes[line_species_i][-j-1], 
                            color=VMR_envelope_colors_i[j+1], ec='none', 
                            )


        if yticks is None:
            yticks = np.logspace(-6, 2, 9)

        ax_VMR.legend(handlelength=0.5, handletextpad=0.3, framealpha=0.7)
        ax_VMR.set(
            xlabel='VMR', xscale='log', xlim=(1e-8, 1e-2), 
            ylabel=ylabel, yscale='log', yticks=yticks, 
            ylim=(self.PT.pressure.min(), self.PT.pressure.max()), 
            )
        ax_VMR.invert_yaxis()

        return ax_VMR

    def fig_contr_em(self, ax_contr):

        ax_contr.plot(
            self.pRT_atm.int_contr_em, self.PT.pressure, 
            c=self.bestfit_color, ls='--', alpha=0.7
            )
        ax_contr.tick_params(axis='x', which='both', top=False, labeltop=False)
        ax_contr.set(xlim=(0, 1.1*np.nanmax(self.pRT_atm.int_contr_em)))

        return ax_contr