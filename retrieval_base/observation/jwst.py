import numpy as np
import matplotlib.pyplot as plt

import broadpy

from .spectrum import Spectrum
from .. import utils

def get_class(m_set, config_data_m_set):
    """
    Get the SpectrumJWST class instance.

    Args:
        m_set (str): Model set identifier.
        config_data_m_set (dict): Configuration data for the model set.

    Returns:
        SpectrumJWST: Instance of SpectrumJWST class.
    """
    return SpectrumJWST(m_set=m_set, **config_data_m_set['kwargs'])

class SpectrumJWST(Spectrum):
    """Class for handling JWST spectrum data."""

    def instrumental_broadening(self, wave, flux, **kwargs):
        """
        Apply instrumental broadening to the spectrum.

        Args:
            wave (numpy.ndarray): Wavelength array.
            flux (numpy.ndarray): Flux array.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Broadened flux array.
        """
        IB = broadpy.InstrumentalBroadening(wave, flux)
        
        if isinstance(self.resolution, np.ndarray):
            # Variable resolution profile
            variable_resolution = np.interp(wave, self.resolution_wave_grid, self.resolution)
            flux_LSF = IB(fwhm=2.998e5/variable_resolution, kernel='gaussian_variable')
            return flux_LSF
        
        # Constant resolution
        flux_LSF = IB(res=self.resolution, kernel='gaussian')
        return flux_LSF
    
    def __init__(self, file, wave_range, m_set, resolution=None, grating='G395H', n_chunks=1, **kwargs):
        """
        Initialize the SpectrumJWST instance.

        Args:
            file (str): File path to the spectrum data.
            wave_range (tuple): Wavelength range to consider.
            m_set (str): Model set identifier.
            resolution (int, optional): Spectral resolution. Defaults to None.
            grating (str, optional): Grating used. Defaults to 'G395H'.
            n_chunks (int, optional): Number of chunks to load data in. Defaults to 1.
            **kwargs: Additional keyword arguments.
        """
        # Read info from wavelength settings
        self.m_set = m_set
        self.load_spectrum(file)
        
        # Load a constant resolution
        self.resolution = resolution
        if self.resolution is None:
            # Load variable resolution profile
            from broadpy.utils import load_nirspec_resolution_profile
            self.resolution_wave_grid, self.resolution \
                = load_nirspec_resolution_profile(grating=grating)

        # Mask user-specified wavelength ranges
        self.mask_wavelength_ranges(kwargs.get('wave_to_mask'), pad=0.)
        
        # Reshape and crop the spectrum
        self.crop_spectrum(wave_range)
        self.reshape_spectrum(n_chunks=n_chunks)

        # Update the number of chips
        self.wave_ranges_chips = np.array([
            [wave.min(), wave.max()] for wave in self.wave
            ])
        self.n_chips = len(self.wave_ranges_chips)
        self.n_pixels = len(self.wave[0])

    def reshape_spectrum(self, n_chunks=1):
        """Reshape the spectrum to a 2D array."""
        
        # Split the spectrum into chunks
        split_wave = np.array_split(self.wave, n_chunks)
        split_flux = np.array_split(self.flux, n_chunks)
        split_err  = np.array_split(self.err, n_chunks)

        # Pad unequal length arrays
        max_len = max([len(wave) for wave in split_wave])

        for i in range(n_chunks):
            pad_len = max_len - len(split_wave[i])
            split_wave[i] = np.pad(
                split_wave[i], (0,pad_len), constant_values=(split_wave[i][0],split_wave[i][-1])
                )
            split_flux[i] = np.pad(split_flux[i], (0,pad_len), constant_values=np.nan)
            split_err[i]  = np.pad(split_err[i], (0,pad_len), constant_values=np.nan)

        # Reshape to 2D array
        self.wave = np.array(split_wave)
        self.flux = np.array(split_flux)
        self.err  = np.array(split_err)
    
    def load_spectrum(self, file):
        """
        Load the spectrum data from a file.

        Args:
            file (str): File path to the spectrum data.
        """
        # Load in the data of the target
        self.wave, self.flux, self.err = np.genfromtxt(file, delimiter=',').T

        mask_isnan = np.isnan(self.flux) | np.isnan(self.err)
        self.wave = self.wave[~mask_isnan]
        self.flux = self.flux[~mask_isnan]
        self.err  = self.err[~mask_isnan]

        # Convert from [um] to [nm]
        self.wave *= 1e3

        # Make a copy of the wavelengths before rv-shifts
        self.wave_initial = self.wave.copy()

    def plot_pre_processing(self, plots_dir):
        """
        Plot the pre-processed spectrum.

        Args:
            plots_dir (str): Directory to save the plots.
        """
        # Plot per chunk
        fig, subfig = utils.get_subfigures_per_chip(self.n_chips)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, None
            if i == 0:
                xlabel = 'Wavelength (nm)'
                ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'

            # Add some padding
            xlim = (self.wave_ranges_chips[i].min()-2, self.wave_ranges_chips[i].max()+2)

            gs = subfig_i.add_gridspec(nrows=1)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_flux.plot(self.wave[i], self.flux[i], 'k-', lw=0.7)

            ax_flux.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel)

        fig.savefig(plots_dir / f'pre_processed_spectrum_per_chunk_{self.m_set}.pdf')
        plt.close(fig)

        # Plot for full spectrum
        fig, subfig = utils.get_subfigures_per_chip(1)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, None
            if i == 0:
                xlabel = 'Wavelength (nm)'
                ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'

            # Add some padding
            xlim = (self.wave_ranges_chips[:].min()-2, self.wave_ranges_chips[:].max()+2)

            gs = subfig_i.add_gridspec(nrows=1)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_flux.plot(self.wave[:].flatten(), self.flux[:].flatten(), 'k-', lw=0.7)

            ax_flux.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel)

        fig.savefig(plots_dir / f'pre_processed_spectrum_{self.m_set}.pdf')
        plt.close(fig)

    def plot_bestfit(self, plots_dir, LogLike, Cov):
        """
        Plot the best-fit spectrum.

        Args:
            plots_dir (str): Directory to save the plots.
            LogLike (object): Log-likelihood object containing fit results.
        """
        # Plot per chunk
        fig, subfig = utils.get_subfigures_per_chip(self.n_chips)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, (None, None)
            label = None
            if i == 0:
                xlabel = 'Wavelength (nm)'
                ylabel = (r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 'Res.')
                label  = r'$\chi_\mathrm{red}^2=$'+'{:.2f}'.format(LogLike.chi_squared_0_red)

            # Add some padding
            xlim = (self.wave_ranges_chips[i].min()-2, self.wave_ranges_chips[i].max()+2)
            
            gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_res  = subfig_i.add_subplot(gs[1])

            ax_flux.plot(self.wave[i], self.flux[i], 'k-', lw=0.5)

            idx_LogLike = LogLike.indices_per_model_setting[self.m_set][i]
            ax_flux.plot(self.wave[i], LogLike.m_flux_phi[idx_LogLike], 'C1-', lw=0.8, label=label)

            ax_res.plot(self.wave[i], self.flux[i]-LogLike.m_flux_phi[idx_LogLike], 'k-', lw=0.8)
            ax_res.axhline(0, c='C1', ls='-', lw=0.5)

            ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])

            # First column of banded covariance matrix is the diagonal
            sigma_scaled = np.nanmean(np.sqrt(Cov[i].cov[idx_LogLike]*LogLike.s_squared[idx_LogLike]))
            sigma_data   = np.nanmean(self.err[i])
            err_kwargs = dict(clip_on=False, capsize=1.7, lw=1., capthick=1.)

            # Plot an errorbar in the axes
            ylim = ax_flux.get_ylim()
            y = ylim[0] + 0.1*np.diff(ylim)[0]
            for x, c, sigma in zip([1.01,1.02], ['C1','k'], [sigma_scaled,sigma_data]):
                ax_flux.errorbar(x, y, yerr=sigma, c=c, transform=ax_flux.get_yaxis_transform(), **err_kwargs)
                ax_res.errorbar(x, 0., yerr=sigma, c=c, transform=ax_res.get_yaxis_transform(), **err_kwargs)

            ylim = ax_res.get_ylim()
            ylim_max = np.max(np.abs(ylim))
            ylim = (-ylim_max, +ylim_max)
            ax_res.set(xlim=xlim, xlabel=xlabel, ylim=ylim, ylabel=ylabel[1])

            if i == 0:
                ax_flux.legend()

        if LogLike.sum_model_settings:
            fig.savefig(plots_dir / f'bestfit_spectrum_per_chunk.pdf')
        else:
            fig.savefig(plots_dir / f'bestfit_spectrum_per_chunk_{self.m_set}.pdf')
        plt.close(fig)

        # Plot for full spectrum
        fig, subfig = utils.get_subfigures_per_chip(1)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, (None, None)
            label = None
            if i == 0:
                xlabel = 'Wavelength (nm)'
                ylabel = (r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 'Res.')
                label  = r'$\chi_\mathrm{red}^2=$'+'{:.2f}'.format(LogLike.chi_squared_0_red)

            # Add some padding
            xlim = (self.wave_ranges_chips[:].min()-2, self.wave_ranges_chips[:].max()+2)
            
            gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_res  = subfig_i.add_subplot(gs[1])

            ax_flux.plot(self.wave[:].flatten(), self.flux[:].flatten(), 'k-', lw=0.5)

            for j in range(self.n_chips):
                if j != 0:
                    label = None

                idx_LogLike = LogLike.indices_per_model_setting[self.m_set][j]
                ax_flux.plot(self.wave[j], LogLike.m_flux_phi[idx_LogLike], 'C1-', lw=0.8, label=label)
                ax_res.plot(self.wave[j], self.flux[j]-LogLike.m_flux_phi[idx_LogLike], 'k-', lw=0.8)
                
            ax_res.axhline(0, c='C1', ls='-', lw=0.5)

            ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])
            ax_res.set(xlim=xlim, xlabel=xlabel, ylim=ylim, ylabel=ylabel[1])

            if i == 0:
                ax_flux.legend()

        if LogLike.sum_model_settings:
            fig.savefig(plots_dir / f'bestfit_spectrum.pdf')
        else:
            fig.savefig(plots_dir / f'bestfit_spectrum_{self.m_set}.pdf')
        plt.close(fig)