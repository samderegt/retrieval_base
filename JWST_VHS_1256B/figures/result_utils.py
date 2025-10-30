import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
from pathlib import Path

from retrieval_base.retrieval import RetrievalRun, Retrieval

q = 0.5 + np.array([-0.997, -0.95, -0.68, 0.0, +0.68, +0.95, +0.997])/2

def latex_format(*posteriors, q=q[[4,2]], decimals=2, as_p10=False):
    """Format the parameters for LaTeX output."""
    q = np.atleast_1d(q)
    full_str = []

    if len(q) == 1:
        if q[0] > 0.5:
            prefix = '<' # Upper limit
        elif q[0] < 0.5:
            prefix = '>' # Lower limit

        for p in posteriors:
            str_i = '{}'.format(np.round(np.quantile(p, q=q), decimals)[0])
            full_str.append('$'+prefix+str_i+'$')

        print(' & '.join(full_str)+r' \\')
        return

    style = f'{{:.{decimals}f}}^+{{:.{decimals}f}}/{{:.{decimals}f}}'
    if as_p10:
        style = f'{{:.{decimals}e}}^+{{:.{decimals}e}}/{{:.{decimals}e}}'

    for p in posteriors:
        str_i = style.format(np.median(p), *np.quantile(p, q=q)-np.median(p))
        str_i = str_i.replace('+', '{+')
        str_i = str_i.replace('/', '}_{')
        str_i += '}'

        full_str.append('$'+str_i+'$')

    print(' & '.join(full_str)+r' \\')

def profile_quantiles(y, q=q, axis=0):
    """Compute quantiles of a profile."""
    return np.quantile(y, q=q, axis=axis)

def convert_CCF_to_SNR(rv, CCF, rv_sep=100, rv_0=0):
    """Convert the cross-correlation function (CCF) to a signal-to-noise ratio (SNR) function."""
    rv_mask = np.abs(rv-rv_0) > rv_sep

    mean_CCF = np.mean(CCF[rv_mask])
    std_CCF  = np.std(CCF[rv_mask])

    return (CCF - mean_CCF) / std_CCF

def bootstrap_CCF_SNR(rv, CCF):

    from astropy.modeling.models import Gaussian1D, Const1D
    from astropy.modeling import fitting
    
    amplitude = CCF[rv==0]
    m_init = (
        Gaussian1D(amplitude=amplitude, stddev=10, fixed={'mean':True}) + Const1D(amplitude=0)
    )
    fit = fitting.LevMarLSQFitter()
    m_fit = fit(m_init, rv, CCF)
    m_fit.amplitude_1 = 0. # Remove vertical offset

    CCF_residuals = CCF - m_fit(rv)
    m_fit_shifted = m_fit.copy()

    SNRs = np.zeros_like(rv, dtype=float)
    for rv_i in rv:
        # Inject peak at another velocity
        m_fit_shifted.mean_0 = m_fit.mean_0 + rv_i
        CCF_shifted = CCF_residuals + m_fit_shifted(rv)

        # Calculate SNR
        SNR_i = convert_CCF_to_SNR(rv, CCF_shifted, rv_sep=300, rv_0=rv_i)
        SNRs[rv==rv_i] = SNR_i[rv==rv_i]

    return SNRs

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

    def get_contribution_function(self):
        """Get the emission contribution functions."""
        
        # Load the components
        self.load_components(['m_spec'])

        integrated_contr = {
            m_set: np.nansum(self.m_spec[m_set].integrated_contr, axis=0)
            for m_set in self.model_settings
        }
        del self.m_spec

        return integrated_contr

    def get_model_spectrum(self, line_species_to_exclude=None, apply_rot_broad=True, m_set=None, reload_m_spec=True):
        """Get the model spectrum."""

        # Load the components
        components = ['Chem', 'PT', 'Cloud', 'Rotation', 'LineOpacity_broad']
        if reload_m_spec or not hasattr(self, 'm_spec_broad'):
            components += ['m_spec_broad']
        self.load_components(components)

        if m_set is None:
            m_set = self.model_settings[0]
        self.ParamTable.set_queried_m_set('all',m_set)

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

        if not apply_rot_broad:
            # Set rotational velocity to 0
            vsini_copy = np.copy(self.ParamTable.get('vsini'))
            self.ParamTable._add_param(name='vsini', m_set='all', val=0.)
            self.ParamTable.set_queried_m_set('all',m_set) # Update the queried table
            self._update_rotation(m_set)

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

        if not apply_rot_broad:
            # Reset the rotational velocity
            self.ParamTable._add_param(name='vsini', m_set='all', val=vsini_copy)
            self.ParamTable.set_queried_m_set('all')
            self._update_rotation(m_set)

        self.ParamTable.set_queried_m_set('all')
        del self.Chem, self.PT, self.Cloud, self.Rotation, self.LineOpacity_broad

    def get_CCF(self, m_spec_template, m_spec_to_subtract=None, rv=None, high_pass_filter={}, plot=False):

        if rv is None:
            rv = np.arange(-1000, 1000+1e-6, 1)

        # Load the components
        self.load_components(['d_spec', 'LogLike', 'Cov'])
        m_set = self.model_settings[0]

        CCF = np.nan * np.ones((len(rv), self.d_spec[m_set].n_chips))
        ACF = np.nan * np.ones((len(rv), self.d_spec[m_set].n_chips))

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
        for i, rv_i in enumerate(tqdm(rv, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')):

            # Loop over chips
            for j in range(d_res.shape[0]):

                d_res_j = np.copy(d_res[j])  
                if high_pass_filter.get('d_res') is not None:
                    # Apply a high-pass filter
                    d_res_j = high_pass_filter['d_res'](d_res_j)

                mask_j = np.isfinite(d_res_j)

                # Shift the model template
                m_flux_template_j = np.copy(m_spec_template[m_set].flux[j])
                m_wave_template_j = np.copy(m_spec_template[m_set].wave[j])

                m_flux_binned_static_j = np.interp(d_wave[j], m_wave_template_j, m_flux_template_j)
                m_flux_binned_static_j *= self.LogLike.phi[j] # Optimal scaling
                m_flux_binned_static_j[~mask_j] = np.nan

                m_flux_binned_template_j = np.interp(
                    d_wave[j], m_wave_template_j*(1+rv_i/3e5), m_flux_template_j
                    )
                m_flux_binned_template_j *= self.LogLike.phi[j] # Optimal scaling
                m_flux_binned_template_j[~mask_j] = np.nan

                if high_pass_filter.get('m_res') is not None:
                    # Apply a high-pass filter
                    m_flux_binned_static_j   = high_pass_filter['m_res'](m_flux_binned_static_j)
                    m_flux_binned_template_j = high_pass_filter['m_res'](m_flux_binned_template_j)

                # Compute the cross-correlation
                CCF[i,j] = np.dot(
                    # m_flux_binned_template_j[mask_j], 1/self.LogLike.s_squared[j] * self.Cov[j].solve(d_res_j[mask_j])
                    d_res_j[mask_j], 1/self.LogLike.s_squared[j] * self.Cov[j].solve(m_flux_binned_template_j[mask_j])
                    )
                ACF[i,j] = np.dot(
                    m_flux_binned_static_j[mask_j], 1/self.LogLike.s_squared[j] * self.Cov[j].solve(m_flux_binned_template_j[mask_j])
                    )

                if not plot:
                    continue
                if not rv_i==0.:
                    continue
            
                if j%3 == 0:
                    plt.figure(figsize=(10,3))
                plt.plot(d_wave[j], d_res_j, c='k', lw=0.8)
                plt.plot(d_wave[j], m_flux_binned_template_j, c='r', lw=1.2, alpha=0.5)
                
                if j%3 == 2:
                    plt.show()
                    plt.close()

        return rv, CCF, ACF

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

    def compare_evidence(self, other):

        from scipy.special import lambertw, erfcinv
        ln_B = self.ln_Z - other.ln_Z
        B = np.exp(ln_B)
        p = np.real(np.exp(lambertw((-1.0/(B*np.exp(1))),-1)))
        sigma = np.sqrt(2)*erfcinv(p)

        _ln_B = other.ln_Z - self.ln_Z
        _B = np.exp(_ln_B)
        _p = np.real(np.exp(lambertw((-1.0/(_B*np.exp(1))),-1)))
        _sigma = np.sqrt(2)*erfcinv(_p)

        print('Current vs. given: ln(B)={:.2f} | sigma={:.2f}'.format(ln_B, sigma))
        print('Given vs. current: ln(B)={:.2f} | sigma={:.2f}'.format(_ln_B, _sigma))
        return B, sigma

    def compare_AIC_BIC(self, other):

        self.load_components(['LogLike', 'ParamTable'])
        other.load_components(['LogLike', 'ParamTable'])

        k_self  = self.ParamTable.n_free_params
        k_other = other.ParamTable.n_free_params

        n_self  = self.LogLike.N_d
        n_other = other.LogLike.N_d

        AIC_self  = 2*k_self - 2*self.LogLike.ln_L
        AIC_other = 2*k_other - 2*other.LogLike.ln_L

        BIC_self  = k_self*np.log(n_self) - 2*self.LogLike.ln_L
        BIC_other = k_other*np.log(n_other) - 2*other.LogLike.ln_L

        print('Current vs. given: AIC={:.2f}'.format(AIC_self-AIC_other))
        print('Current vs. given: BIC={:.2f}'.format(BIC_self-BIC_other))
        
    @staticmethod
    def _load_config(prefix):
        """Load the config file from the data directory."""

        import importlib

        # Remove any existing tmp_config.py and related cached files/modules
        tmp_file = Path('./tmp_config.py')
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception as e:
                print(f'Warning: could not remove existing tmp_config.py: {e}')

        tmp_pyc = Path('./tmp_config.pyc')
        if tmp_pyc.exists():
            try:
                tmp_pyc.unlink()
            except Exception:
                pass

        cache_dir = Path('./__pycache__')
        if cache_dir.exists():
            for f in cache_dir.glob('tmp_config*.pyc'):
                try:
                    f.unlink()
                except Exception:
                    pass

        if 'tmp_config' in sys.modules:
            try:
                del sys.modules['tmp_config']
            except Exception:
                pass

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
        
        self.ln_Z = stats['nested importance sampling global log-evidence']
        # self.ln_Z = stats['nested sampling global log-evidence']

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