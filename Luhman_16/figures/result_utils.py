import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path

from retrieval_base.retrieval import RetrievalRun, Retrieval

q = 0.5 + np.array([-0.997, -0.95, -0.68, 0.0, +0.68, +0.95, +0.997])/2

def profile_quantiles(y, q=q, axis=0):
    """Compute quantiles of a profile."""
    return np.quantile(y, q=q, axis=axis)

def convert_CCF_to_SNR(rv, CCF, rv_sep=100):
    """Convert the cross-correlation function (CCF) to a signal-to-noise ratio (SNR) function."""
    rv_mask = np.abs(rv) > rv_sep

    mean_CCF = np.mean(CCF[rv_mask])
    std_CCF  = np.std(CCF[rv_mask])

    return (CCF - mean_CCF) / std_CCF

class RetrievalResults(RetrievalRun, Retrieval):

    def __init__(self, prefix):

        self.evaluation = True

        # Load the config file
        config = self._load_config(prefix)

        # Set up the load_components() method
        Retrieval.__init__(self, config, create_output_dir=False)

        # Load the components
        self.load_components(['ParamTable'])

        # Load the posterior and best-fit parameters
        self.posterior, self.bestfit_parameters = self._load_posterior_and_bestfit()

    def get_model_spectrum(self, line_species_to_exclude=None):
        """Get the model spectrum."""

        # Load the components
        self.load_components(['m_spec_broad', 'Chem', 'PT', 'Cloud', 'Rotation', 'LineOpacity_broad'])

        m_set = self.model_settings[0]
        self.ParamTable.set_queried_m_set(['all',m_set])

        # Update the mass fractions
        for key_i in self.Chem[m_set].mass_fractions:
            if key_i in ['MMW','H2','He']:
                continue

            if line_species_to_exclude is not None:
                if key_i in line_species_to_exclude:
                    # Set the mass fraction to zero
                    self.Chem[m_set].mass_fractions[key_i] *= 0.0
                    continue

        if not self._skip_update(m_set, self.LineOpacity_broad):
            for LineOpacity_i in self.LineOpacity_broad[m_set]:
                LineOpacity_i(self.ParamTable, PT=self.PT[m_set], Chem=self.Chem[m_set])

        # Update the broadened model spectrum
        self.m_spec_broad[m_set].evaluation = False
        self.m_spec_broad[m_set](
            self.ParamTable, 
            Chem=self.Chem[m_set], 
            PT=self.PT[m_set],
            Cloud=self.Cloud[m_set],
            Rotation=self.Rotation[m_set],
            LineOpacity=self.LineOpacity_broad[m_set],
            )

        self.ParamTable.set_queried_m_set('all')
        del self.Chem, self.PT, self.Cloud, self.Rotation, self.LineOpacity_broad

    def get_CCF(self, m_spec_template, m_spec_to_subtract=None, rv=None, high_pass_filter={}, plot=False):

        if rv is None:
            rv = np.arange(-1000, 1000+1e-6, 1)

        # Load the components
        self.load_components(['d_spec', 'LogLike', 'Cov'])
        m_set = self.model_settings[0]

        CCF = np.nan * np.ones((len(rv), self.d_spec[m_set].n_chips))

        d_wave = np.copy(self.d_spec[m_set].wave)
        d_res  = np.copy(self.d_spec[m_set].flux)
        
        # Compute the cross-correlation using residuals
        if m_spec_to_subtract in ['m_flux_phi','complete']:
            # Residual with respect to the complete model
            d_res -= np.array(self.LogLike.m_flux_phi)
        elif m_spec_to_subtract is not None:
            # Residual with respect to a specific model
            d_res -= m_spec_to_subtract[m_set].flux_binned * self.LogLike.phi[:,None]

        from tqdm import tqdm
        for i, rv_i in enumerate(tqdm(rv)):

            # Loop over chips
            for j in range(d_res.shape[0]):

                d_res_j = np.copy(d_res[j])  
                if high_pass_filter.get('d_res') is not None:
                    # Apply a high-pass filter
                    d_res_j = high_pass_filter['d_res'](d_res_j)

                mask_j = np.isfinite(d_res_j)

                # Shift the model template
                m_flux_template_j = np.copy(m_spec_template[m_set].flux[j])
                m_wave_template_j = np.copy(m_spec_template[m_set].wave[j]) * (1 + rv_i/3e5)

                m_flux_binned_template_j = np.interp(
                    d_wave[j], m_wave_template_j, m_flux_template_j
                    )
                m_flux_binned_template_j *= self.LogLike.phi[j] # Optimal scaling
                m_flux_binned_template_j[~mask_j] = np.nan

                if high_pass_filter.get('m_res') is not None:
                    # Apply a high-pass filter
                    m_flux_binned_template_j = high_pass_filter['m_res'](m_flux_binned_template_j)

                # Compute the cross-correlation
                CCF[i,j] = np.dot(
                    m_flux_binned_template_j[mask_j], 1/self.LogLike.s_squared[j] * self.Cov[j].solve(d_res_j[mask_j])
                    )

                if not plot:
                    continue
                if not rv_i==0.:
                    continue
            
                if j%3 == 0:
                    plt.figure(figsize=(10,3))
                plt.plot(d_wave[j], d_res_j, c='k', lw=0.8)
                plt.plot(d_wave[j], m_flux_binned_template_j, c='r', lw=1.2)
                
                if j%3 == 2:
                    plt.show()
                    plt.close()

        return rv, CCF

    def get_mean_scaled_uncertainty(self, mode='order', n_dets_per_order=3):
        """Get the mean scaled uncertainty for each order or chip."""

        if mode not in ['order', 'chip']:
            raise ValueError('mode must be either "order" or "chip"')

        # Load the components
        self.load_components(['LogLike', 'Cov'])

        sigma = []

        if mode == 'chip':
            for i in range(self.LogLike.n_chips):
                # Compute per chip
                sigma_i = np.nanmean(np.sqrt(self.Cov[i].cov[0]*self.LogLike.s_squared[i]))
                sigma.append(sigma_i)

        elif mode == 'order':
            # Compute per order
            idx_l = np.arange(0, self.LogLike.n_chips, n_dets_per_order)
            idx_h = idx_l + n_dets_per_order
            for i_l, i_h in zip(idx_l, idx_h):

                # Combine detector diagonals
                diag_i = []
                for i in range(i_l, i_h):
                    try:
                        diag_i.append(self.Cov[i].cov[0]*self.LogLike.s_squared[i])
                    except IndexError:
                        continue
                
                sigma_i = np.nanmean(np.sqrt(np.concatenate(diag_i)))
                sigma.append(sigma_i)
        
        del self.LogLike, self.Cov
        
        return np.array(sigma)

    @staticmethod
    def _load_config(prefix):
        """Load the config file from the data directory."""

        import importlib

        # Find the file with .py suffix
        data_dir = Path(f'{prefix}data')
        print(data_dir)
        config_file = list(data_dir.glob('*.py'))
        if len(config_file) != 1:
            raise ValueError('There should be exactly one config file in the data directory')

        # Temporarily copy the config file to the current directory
        destination = Path('./tmp_config.py')
        destination.write_bytes(config_file[0].read_bytes())

        # Import the config file
        config = importlib.import_module('tmp_config')
        setattr(config, 'config_file', 'tmp_config.py')

        # Update to the full path
        config.prefix = prefix
        return config

    def _load_posterior_and_bestfit(self, posterior=None):
        """Load the posterior and best-fit parameters from the output files."""

        import pymultinest

        # Read the equally-weighted posterior
        analyzer = pymultinest.Analyzer(
            n_params=self.ParamTable.n_free_params, 
            outputfiles_basename=self.config.prefix
            )
        posterior = analyzer.get_equal_weighted_posterior()
        posterior = posterior[:,:-1]
        
        # Read best-fit parameters
        stats = analyzer.get_stats()
        bestfit_parameters = np.array(stats['modes'][0]['maximum a posterior'])
        
        return posterior, bestfit_parameters

class HighPassFilter:

    def __init__(self, filter_type, **kwargs):

        self.filter_type = filter_type
        self.kwargs = kwargs

        if self.filter_type not in ['savgol', 'nanmedian']:
            raise ValueError('filter_type must be either "savgol" or "nanmedian"')
    
    def __call__(self, flux):

        if self.filter_type == 'savgol':
            return self.savitzky_golay(flux)
        elif self.filter_type == 'nanmedian':
            return self.nanmedian(flux)

    def savitzky_golay(self, flux):

        mask_isfinite = np.isfinite(flux)
        lp_flux = np.nan * flux.copy()

        window_length_copy = np.copy(self.kwargs['window_length'])
        self.kwargs['window_length'] = min(window_length_copy, mask_isfinite.sum())
        
        # Apply Savitzky-Golay filter to remove broad structure
        from scipy.signal import savgol_filter
        lp_flux[mask_isfinite] = savgol_filter(flux[mask_isfinite], **self.kwargs)

        self.kwargs['window_length'] = window_length_copy
        
        return flux - lp_flux

    def nanmedian(self, flux):

        # Apply median filter to remove broad structure
        #from scipy.ndimage import generic_filter
        #lp_flux = generic_filter(flux, np.nanmedian, **self.kwargs)
        from scipy.ndimage import median_filter
        lp_flux = median_filter(flux, **self.kwargs)
        
        return flux - lp_flux