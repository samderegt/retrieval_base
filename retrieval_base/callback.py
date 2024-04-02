import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import os
import time
import json
import copy
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
                 species_to_plot_VMR=['12CO', 'H2O', '13CO', 'CH4', 'NH3', 'C18O'], 
                 species_to_plot_CCF=['12CO', 'H2O', '13CO', 'CH4'], 
                 model_settings={}, 
                 ):
        
        self.elapsed_times = []
        self.active = False

        self.return_PT_mf = {m_set: False for m_set in model_settings.keys()}

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
            name='envelope_cmap', colors=['w', self.posterior_color], 
            )
        self.envelope_colors = self.envelope_cmap([0.0,0.2,0.4,0.6,0.8])
        self.envelope_colors[0,-1] = 0.0

        self.species_to_plot_VMR = species_to_plot_VMR
        self.species_to_plot_CCF = species_to_plot_CCF

    def __call__(self, 
                 Param, 
                 LogLike, 
                 Cov, 
                 PT, 
                 Chem, 
                 m_spec, 
                 pRT_atm, 
                 posterior, 
                 m_spec_species=None, 
                 pRT_atm_species=None, 
                 ):

        time_A = time.time()
        self.cb_count += 1

        # Make attributes for convenience
        self.Param   = Param
        self.LogLike = LogLike
        self.Cov     = Cov
        self.PT      = PT
        self.Chem    = Chem
        self.m_spec  = m_spec
        self.pRT_atm = pRT_atm

        self.m_spec_species  = m_spec_species
        self.pRT_atm_species = pRT_atm_species

        if not self.evaluation:
            # Use only the last n samples to plot the posterior
            n_samples = min([len(posterior), self.n_samples_to_use])
            self.posterior = posterior[-n_samples:]
        else:
            self.posterior = posterior

        # Display the mean elapsed time per lnL evaluation
        print('\n\nElapsed time per evaluation: {:.2f} seconds'.format(np.mean(self.elapsed_times)))
        self.elapsed_times.clear()

        # Create the labels
        self.param_labels = self.Param.param_mathtext

        chi_squared_tot, n_dof = 0, 0
        if not isinstance(self.LogLike, dict):
            chi_squared_tot += self.LogLike.chi2_0
            n_dof += self.LogLike.N_d

        else:
            for m_set, LogLike_i in self.LogLike.items():
                print(f'\n--- {m_set} -------------------------')
                print('Reduced chi-squared (w/o uncertainty-model) = {:.2f}\n(chi-squared={:.2f}, n_dof={:.0f})'.format(
                    LogLike_i.chi2_0_red, LogLike_i.chi2_0, LogLike_i.N_d
                    ))
                
                chi_squared_tot += LogLike_i.chi2_0
                n_dof += self.d_spec[m_set].mask_isfinite.sum()

        print(f'\n--- Total -------------------------')
        print('Reduced chi-squared (w/o uncertainty-model) = {:.2f}\n(chi-squared={:.2f}, n_dof={:.0f})'.format(
            chi_squared_tot/n_dof, chi_squared_tot, n_dof
            ))

        # Read the best-fitting free parameters
        self.bestfit_params = []
        print('\nBest-fitting free parameters:')
        for i, key_i in enumerate(self.Param.param_keys):
            print('{} = {:.2f}'.format(key_i, self.Param.cube[i]))
            self.bestfit_params.append(self.Param.cube[i])

        if not isinstance(self.LogLike, dict):
            print(f'\n-------------------------------------')
            if self.LogLike.scale_flux:
                print('\nOptimal flux-scaling parameters:')
                print(self.LogLike.phi[:,:,0].round(2))
                print('R_p (R_Jup):')
                print(np.sqrt(self.LogLike.phi[:,:,0]).round(2))
            if self.LogLike.scale_err:
                print('\nOptimal uncertainty-scaling parameters:')
                print(self.LogLike.s2.round(2))
        else:
            for m_set, LogLike_i in self.LogLike.items():
                print(f'\n--- {m_set} -------------------------')
                if LogLike_i.scale_flux:
                    print('\nOptimal flux-scaling parameters:')
                    print(LogLike_i.phi[:,:,0].round(2))
                    print('R_p (R_Jup):')
                    print(np.sqrt(LogLike_i.phi[:,:,0]).round(2))
                if LogLike_i.scale_err:
                    print('\nOptimal uncertainty-scaling parameters:')
                    print(LogLike_i.s2.round(2))
        
        self.bestfit_params = np.array(self.bestfit_params)
        
        # Save the bestfit parameters in a .json file
        # and the ModelSpectrum instance as .pkl
        self.save_bestfit()
            
        # Save a separate figure of the PT profile
        figs.fig_PT(
            PT=self.PT, 
            pRT_atm=self.pRT_atm, 
            ax_PT=None, 
            envelope_colors=self.envelope_colors, 
            posterior_color=self.posterior_color, 
            bestfit_color=self.bestfit_color, 
            prefix=self.prefix
        )

        # Make a summary figure
        self.fig_summary()

        if not self.evaluation:
            # Remove attributes from memory
            del self.Param, self.LogLike, self.PT, self.Chem, self.m_spec, self.pRT_atm, self.posterior

            time_B = time.time()
            print('\nPlotting took {:.0f} seconds\n'.format(time_B-time_A))

            return
            
        for m_set in self.d_spec.keys():

            if isinstance(self.LogLike, dict):
                LogLike_i = self.LogLike[m_set]
                Cov_i     = self.Cov[m_set]
            else:
                LogLike_i = self.LogLike
                Cov_i     = self.Cov

            # Plot the CCFs + spectra of species' contributions
            figs.fig_species_contribution(
                d_spec=self.d_spec[m_set], 
                m_spec=self.m_spec[m_set], 
                m_spec_species=self.m_spec_species[m_set], 
                pRT_atm=self.pRT_atm[m_set], 
                pRT_atm_species=self.pRT_atm_species[m_set], 
                Chem=self.Chem[m_set], 
                LogLike=LogLike_i, 
                Cov=Cov_i, 
                species_to_plot=self.species_to_plot_CCF, 
                prefix=self.prefix, 
                m_set=m_set, 
                )
        
            # Plot the auto-correlation of the residuals
            figs.fig_residual_ACF(
                d_spec=self.d_spec[m_set], 
                m_spec=self.m_spec[m_set], 
                LogLike=LogLike_i, 
                Cov=Cov_i, 
                bestfit_color=self.bestfit_color, 
                prefix=self.prefix, 
                m_set=m_set, 
                )

            # Plot the covariance matrices
            all_cov = figs.fig_cov(
                LogLike=LogLike_i, 
                Cov=Cov_i, 
                d_spec=self.d_spec[m_set], 
                cmap=self.envelope_cmap, 
                prefix=self.prefix, 
                m_set=m_set, 
                )

        # Plot the abundances in a corner-plot
        #self.fig_abundances_corner()

        # Remove attributes from memory
        del self.Param, self.LogLike, self.PT, self.Chem, self.m_spec, self.pRT_atm, self.posterior

        time_B = time.time()
        print('\nPlotting took {:.0f} seconds\n'.format(time_B-time_A))

    def save_bestfit(self):
        
        # Save the best-fitting parameters
        params_to_save = {}
        for m_set, Param_i in self.Param.Params_m_set.items():
            params_to_save[m_set] = {}
            for key_i, val_i in Param_i.params.items():
                if isinstance(val_i, np.ndarray):
                    val_i = val_i.tolist()
                params_to_save[m_set][key_i] = val_i

        dict_to_save = {
            'params': params_to_save, 
        }
        for m_set, PT_i in self.PT.items():
            dict_to_save[m_set] = {
                'temperature': PT_i.temperature.tolist(), 
                'pressure': PT_i.pressure.tolist(), 
                }
        if isinstance(self.LogLike, dict):
            for m_set, LogLike_i in self.LogLike.items():
                dict_to_save[m_set]['phi'] = LogLike_i.phi.tolist()
                dict_to_save[m_set]['s2']  = LogLike_i.s2.tolist()
        else:
            dict_to_save['phi'] = self.LogLike.phi.tolist()
            dict_to_save['s2']  = self.LogLike.s2.tolist()

        with open(self.prefix+'data/bestfit.json', 'w') as fp:
            json.dump(dict_to_save, fp, indent=4)

        # Save some of the objects
        for m_set in self.PT.keys():
            af.pickle_save(self.prefix+f'data/bestfit_PT_{m_set}.pkl', self.PT[m_set])

            Chem_to_save = copy.copy(self.Chem[m_set])
            if hasattr(Chem_to_save, 'fastchem'):
                del Chem_to_save.fastchem
            if hasattr(Chem_to_save, 'output'):
                del Chem_to_save.output
            if hasattr(Chem_to_save, 'input'):
                del Chem_to_save.input
            af.pickle_save(self.prefix+f'data/bestfit_Chem_{m_set}.pkl', Chem_to_save)

            af.pickle_save(self.prefix+f'data/bestfit_m_spec_{m_set}.pkl', self.m_spec[m_set])

            # Save the contribution functions and cloud opacities
            np.save(
                self.prefix+f'data/bestfit_int_contr_em_{m_set}.npy', 
                self.pRT_atm[m_set].int_contr_em
                )
            np.save(
                self.prefix+f'data/bestfit_int_contr_em_per_order_{m_set}.npy', 
                self.pRT_atm[m_set].int_contr_em_per_order
                )
            np.save(
                self.prefix+f'data/bestfit_int_opa_cloud_{m_set}.npy', 
                self.pRT_atm[m_set].int_opa_cloud
                )

            if isinstance(self.LogLike, dict):
                # Save the best-fitting log-likelihood
                LogLike_to_save = copy.deepcopy(self.LogLike[m_set])
                del LogLike_to_save.d_spec
                af.pickle_save(self.prefix+f'data/bestfit_LogLike_{m_set}.pkl', LogLike_to_save)

                # Save the best-fitting covariance matrix
                af.pickle_save(self.prefix+f'data/bestfit_Cov_{m_set}.pkl', self.Cov[m_set])

    def fig_abundances_corner(self):

        included_params = []

        # Plot the abundances
        if self.Param.chem_mode == 'free':

            for species_i in self.Chem.species_info.index:

                line_species_i = self.Chem.read_species_info(species_i, 'pRT_name')
                
                # Add to the parameters to be plotted
                if (line_species_i in self.Chem.line_species) and \
                    (f'log_{species_i}' in self.Param.param_keys):
                    included_params.append(f'log_{species_i}')
                for j in range(3):
                    if (line_species_i in self.Chem.line_species) and \
                        (f'log_{species_i}_{j}' in self.Param.param_keys):
                        included_params.append(f'log_{species_i}_{j}')

            if 'log_C_ratio' in self.Param.param_keys:
                included_params.append('log_C_ratio')

            # Add C/O and Fe/H to the parameters to be plotted
            if self.evaluation:
                
                # Add to the posterior
                self.posterior = np.concatenate(
                    (self.posterior, self.Chem.CO_posterior[:,None], 
                     self.Chem.FeH_posterior[:,None]), axis=1
                    )
                # Add to the parameter keys
                self.Param.param_keys = np.concatenate(
                    (self.Param.param_keys, ['C/O', 'C/H'])
                )
                self.param_labels = np.concatenate(
                    (self.param_labels, ['C/O', '[C/H]'])
                    )
                
                self.bestfit_params = np.concatenate(
                    (self.bestfit_params, [self.Chem.CO, self.Chem.CH])
                )
                included_params.extend(['C/O', 'C/H'])

        elif self.Param.chem_mode in ['eqchem', 'fastchem', 'SONORAchem']:
            
            for key in self.Param.param_keys:
                if key.startswith('log_C_ratio'):
                    included_params.append(key)
                
                elif key.startswith('log_C13_12_ratio'):
                    included_params.append(key)
                elif key.startswith('log_O18_16_ratio'):
                    included_params.append(key)
                elif key.startswith('log_O17_16_ratio'):
                    included_params.append(key)

                elif key.startswith('log_P_quench'):
                    included_params.append(key)

            included_params.extend(['C/O', 'Fe/H'])

        figsize = (
            4/3*len(included_params), 4/3*len(included_params)
            )
        fig, ax = self.fig_corner(
            included_params=included_params, 
            fig=plt.figure(figsize=figsize), 
            smooth=False, ann_fs=10
            )

        # Plot the VMR per species
        ax_VMR = fig.add_axes([0.6,0.6,0.32,0.32])
        ax_VMR = figs.fig_VMR(
            ax_VMR=ax_VMR, 
            Chem=self.Chem, 
            species_to_plot=self.species_to_plot_VMR, 
            pressure=self.PT.pressure, 
            )

        plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, wspace=0, hspace=0)
        fig.savefig(self.prefix+'plots/abundances.pdf')
        plt.close(fig)

        if self.evaluation and self.Param.chem_mode == 'free':
            # Remove the changes
            self.posterior        = self.posterior[:,:-2]
            self.Param.param_keys = self.Param.param_keys[:-2]
            self.param_labels     = self.param_labels[:-2]
            self.bestfit_params   = self.bestfit_params[:-2]
        
    def fig_corner(self, included_params=None, fig=None, smooth=False, ann_fs=9):
        
        # Compute the 0.16, 0.5, and 0.84 quantiles
        self.param_quantiles = np.array(
            [af.quantiles(self.posterior[:,i], q=[0.16,0.5,0.84]) \
             for i in range(self.posterior.shape[1])]
             )
        # Base the axes-limits off of the quantiles
        self.param_range = np.array(
            [(4*(q_i[0]-q_i[1])+q_i[1], 4*(q_i[2]-q_i[1])+q_i[1]) \
             for q_i in self.param_quantiles]
            )

        self.median_params = np.array(list(self.param_quantiles[:,1]))

        if fig is None:
            fig = plt.figure(figsize=(15,15))
        
        # Only select the included parameters
        mask_params = np.ones(len(self.Param.param_keys), dtype=bool)
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
            plot_datapoints=self.evaluation, 

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

        if isinstance(self.LogLike, dict):
            # Different wavelength-settings, show multiple panels
            n_m_set = len(self.d_spec)
        else:
            # Same wavelength-setting
            n_m_set = 1
        l, b, w, h = [0.4,0.70,0.57,0.25]

        ax_spec, ax_res = [], []
        for i in range(n_m_set):
            ax_res_dim_i  = [l, b+i*(h+0.03)/(n_m_set), w, 0.97*h/(5*n_m_set)]
            ax_spec_dim_i = [l, ax_res_dim_i[1]+ax_res_dim_i[3], w, 4*0.97*h/(5*n_m_set)]

            ax_spec.append(fig.add_axes(ax_spec_dim_i))
            ax_res.append(fig.add_axes(ax_res_dim_i))

        ax_spec = np.array(ax_spec)[::-1]
        ax_res  = np.array(ax_res)[::-1]
        for i, (m_set, d_spec_i) in enumerate(self.d_spec.items()):
            if isinstance(self.LogLike, dict):
                LogLike_i = self.LogLike[m_set]
                Cov_i = self.Cov[m_set]
            else:
                LogLike_i = self.LogLike
                Cov_i = self.Cov

            # Plot the best-fitting spectrum
            ax_spec[i], ax_res[i] = figs.fig_bestfit_model(
                d_spec=d_spec_i, 
                m_spec=self.m_spec[m_set], 
                LogLike=LogLike_i, 
                Cov=Cov_i, 
                bestfit_color=self.bestfit_color, 
                ax_spec=ax_spec[i], 
                ax_res=ax_res[i], 
                prefix=self.prefix, 
                xlabel=['Wavelength (nm)', None][i]
                )
            
            if n_m_set == 1:
                del LogLike_i, Cov_i
                break

        ax_VMR = fig.add_axes([0.65,0.43,0.1,0.22])
        l, b, w, h = ax_VMR.get_position().bounds
        ax_PT = fig.add_axes([l+w,b,h,h])
        #ax_contr = ax_PT.twiny()

        # Plot the VMR per species
        for i, m_set in enumerate(self.Chem.keys()):
            ax_VMR = figs.fig_VMR(
                ax_VMR=ax_VMR, 
                Chem=self.Chem[m_set], 
                species_to_plot=self.species_to_plot_VMR, 
                pressure=self.PT[m_set].pressure, 
                ls=['-', '--'][i]
                )

        # Plot the best-fitting PT profile
        ax_PT = figs.fig_PT(
            PT=self.PT, 
            pRT_atm=self.pRT_atm, 
            ax_PT=ax_PT, 
            envelope_colors=self.envelope_colors, 
            posterior_color=self.posterior_color, 
            bestfit_color=self.bestfit_color, 
            ylabel=None, 
            yticks=[]
            )

        # Plot the integrated emission contribution function
        #ax_contr = self.fig_contr_em(ax_contr)

        #plt.show()
        if self.evaluation:

            for i in range(ax.shape[0]):
                # Plot the histograms separately
                figs.fig_hist_posterior(
                    posterior_i=self.posterior[:,i], 
                    param_range_i=self.param_range[i], 
                    param_quantiles_i=self.param_quantiles[i], 
                    param_key_i=self.Param.param_keys[i], 
                    posterior_color=self.posterior_color, 
                    title=self.param_labels[i], 
                    bins=20, 
                    prefix=self.prefix
                    )
            
            fig.savefig(self.prefix+'plots/final_summary.pdf')
            fig.savefig(self.prefix+f'plots/final_summary.png', dpi=100)

        else:
            fig.savefig(self.prefix+'plots/live_summary.pdf')
            fig.savefig(self.prefix+f'plots/live_summary_{self.cb_count}.png', dpi=100)
        plt.close(fig)

        for m_set in self.d_spec.keys():
            if isinstance(self.LogLike, dict):
                LogLike_i = self.LogLike[m_set]
                Cov_i = self.Cov[m_set]
            else:
                LogLike_i = self.LogLike
                Cov_i = self.Cov                

            # Plot the best-fit spectrum in subplots
            figs.fig_bestfit_model(
                d_spec=self.d_spec[m_set], 
                m_spec=self.m_spec[m_set], 
                LogLike=LogLike_i, 
                Cov=Cov_i, 
                bestfit_color=self.bestfit_color, 
                ax_spec=None, 
                ax_res=None, 
                prefix=self.prefix, 
                m_set=m_set
                )

            if n_m_set == 1:
                break