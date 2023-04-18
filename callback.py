import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import time
import json
import corner

import auxiliary_functions as af
import figures as figs

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

        time_A = time.time()
        self.cb_count += 1

        # Make attributes for convenience
        self.Param   = Param
        self.LogLike = LogLike
        self.PT      = PT
        self.Chem    = Chem
        self.m_spec  = m_spec
        self.pRT_atm = pRT_atm

        # Use only the last n samples to plot the posterior
        n_samples = min([len(posterior), self.n_samples_to_use])
        self.posterior = posterior[:n_samples]

        # Display the mean elapsed time per lnL evaluation
        print('\n\nElapsed time per evaluation: {:.2f} seconds'.format(np.mean(self.elapsed_times)))
        self.elapsed_times.clear()

        # Compute the 0.16, 0.5, and 0.84 quantiles
        self.param_quantiles = np.array([af.quantiles(self.posterior[:,i], q=[0.16,0.5,0.84]) \
                                         for i in range(self.posterior.shape[1])])
        # Base the axes-limits off of the quantiles
        self.param_range = [(4*(q_i[0]-q_i[1])+q_i[1], 4*(q_i[2]-q_i[1])+q_i[1]) \
                            for q_i in self.param_quantiles]

        # Create the labels
        self.param_labels = list(self.Param.param_mathtext.values())

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

        ax_spec, ax_res = figs.fig_bestfit_model(
            d_spec=self.d_spec, 
            m_spec=self.m_spec, 
            LogLike=self.LogLike, 
            bestfit_color=self.bestfit_color, 
            ax_spec=ax_spec, 
            ax_res=ax_res
            )

        # Plot the VMR per species
        ax_VMR = fig.add_axes([0.63,0.475,0.1,0.22])
        ax_VMR = self.fig_VMR(ax_VMR)

        # Plot the best-fitting PT profile
        l, b, w, h = ax_VMR.get_position().bounds
        ax_PT = fig.add_axes([l+w,b,h,h])
        ax_PT = self.fig_PT(ax_PT, ylabel=None, yticks=[])

        #plt.show()
        plt.savefig(self.prefix+'plots/live_summary.pdf')
        plt.savefig(self.prefix+f'plots/live_summary_{self.cb_count}.png', dpi=100)
        plt.close()

        # Plot the best-fit spectrum in subplots
        figs.fig_bestfit_model(
            d_spec=self.d_spec, 
            m_spec=self.m_spec, 
            LogLike=self.LogLike, 
            bestfit_color=self.bestfit_color, 
            ax_spec=None, 
            ax_res=None
            )

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