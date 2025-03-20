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
    
    def __init__(self, file, m_set, wave_range=None, resolution=None, grating='G395H', n_chunks=1, **kwargs):
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
        self.load_spectrum(file, min_SNR=kwargs.get('min_SNR', 0.))
        
        # Load a constant resolution
        self.resolution = resolution
        self.grating = grating
        if self.resolution is None:
            if self.grating.lower() not in ['g140h', 'g235h', 'g395h']:
                raise ValueError(f'Grating {self.grating} requires resolution to be specified.')

            # Load variable resolution profile
            from broadpy.utils import load_nirspec_resolution_profile
            self.resolution_wave_grid, self.resolution \
                = load_nirspec_resolution_profile(grating=self.grating)

        # Mask user-specified wavelength ranges
        self.mask_wavelength_ranges(kwargs.get('wave_to_mask'), pad=0.)
        
        # Reshape and crop the spectrum
        if wave_range is not None:
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

        # Remove the nan values
        mask_isnan = np.isnan(self.flux) | np.isnan(self.err)
        self.wave = self.wave[~mask_isnan]
        self.flux = self.flux[~mask_isnan]
        self.err  = self.err[~mask_isnan]
        
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
    
    def load_spectrum(self, file, min_SNR=0.):
        """
        Load the spectrum data from a file.

        Args:
            file (str): File path to the spectrum data.
            min_SNR (float): Minimum signal-to-noise ratio to keep.
        """
        # Load in the data of the target
        self.wave, self.flux, self.err = np.genfromtxt(file, delimiter=',').T

        mask_isnan = np.isnan(self.flux) | np.isnan(self.err)
        self.wave = self.wave[~mask_isnan]
        self.flux = self.flux[~mask_isnan]
        self.err  = self.err[~mask_isnan]

        SNR = self.flux / self.err
        mask_SNR = SNR > min_SNR
        self.flux[~mask_SNR] = np.nan
        self.err[~mask_SNR]  = np.nan

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

        # Plot for full spectrum
        fig, subfig = utils.get_subfigures_per_chip(1)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, None
            if i == 0:
                xlabel = r'Wavelength ($\mathrm{\mu}$m)'
                ylabel = r'$F_\lambda\ (\mathrm{W\ m^{-2}\ \mu m^{-1}})$'

            # Add some padding
            xlim = ((self.wave_ranges_chips[:].min()-2)*1e-3, (self.wave_ranges_chips[:].max()+2)*1e-3)

            gs = subfig_i.add_gridspec(nrows=1)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_flux.fill_between(
                self.wave[:].flatten()*1e-3, self.flux[:].flatten()-self.err[:].flatten(), 
                self.flux[:].flatten()+self.err[:].flatten(), fc='k', alpha=0.2, ec='none'
                )
            ax_flux.plot(self.wave[:].flatten()*1e-3, self.flux[:].flatten(), 'k-', lw=0.7)

            ax_flux.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel)

        fig.savefig(plots_dir / f'pre_processed_spectrum_{self.m_set}.pdf')
        plt.close(fig)

        # Plot per chunk
        n_chunks = self.n_chips
        if (self.n_chips == 1) and (self.grating.lower() in ['g140h', 'g235h', 'g395h']):
            n_chunks = 4
        if n_chunks == 1:
            return

        fig, subfig = utils.get_subfigures_per_chip(n_chunks)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, None
            if i == 0:
                xlabel = r'Wavelength ($\mathrm{\mu}$m)'
                ylabel = r'$F_\lambda\ (\mathrm{W\ m^{-2}\ \mu m^{-1}})$'

            # Add some padding
            if self.n_chips == 1:
                xlim = np.linspace((self.wave_ranges_chips.min())*1e-3, (self.wave_ranges_chips.max())*1e-3, n_chunks+1)[i:i+2]
                i = 0
            else:
                xlim = ((self.wave_ranges_chips[i].min())*1e-3, (self.wave_ranges_chips[i].max())*1e-3)

            gs = subfig_i.add_gridspec(nrows=1)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_flux.fill_between(
                self.wave[i]*1e-3, self.flux[i]-self.err[i], self.flux[i]+self.err[i], 
                fc='k', alpha=0.2, ec='none'
                )
            ax_flux.plot(self.wave[i]*1e-3, self.flux[i], 'k-', lw=0.7)

            ax_flux.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel)

        fig.savefig(plots_dir / f'pre_processed_spectrum_per_chunk_{self.m_set}.pdf')
        plt.close(fig)

    def plot_bestfit(self, plots_dir, LogLike, Cov):
        """
        Plot the best-fit spectrum.

        Args:
            plots_dir (str): Directory to save the plots.
            LogLike (object): Log-likelihood object containing fit results.
        """

        # Plot for full spectrum
        fig, subfig = utils.get_subfigures_per_chip(1)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, (None, None)
            label = None
            if i == 0:
                xlabel = r'Wavelength ($\mathrm{\mu}$m)'
                ylabel = (r'$F_\lambda\ (\mathrm{W\ m^{-2}\ \mu m^{-1}})$', 'Res.')
                label  = r'$\chi_\mathrm{red}^2=$'+'{:.2f}'.format(LogLike.chi_squared_0_red)

            # Add some padding
            xlim = ((self.wave_ranges_chips[:].min()-2)*1e-3, (self.wave_ranges_chips[:].max()+2)*1e-3)
            
            gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_res  = subfig_i.add_subplot(gs[1])

            ax_flux.fill_between(
                self.wave[:].flatten()*1e-3, self.flux[:].flatten()-self.err[:].flatten(), 
                self.flux[:].flatten()+self.err[:].flatten(), fc='k', alpha=0.2, ec='none', zorder=-1
                )
            ax_flux.plot(self.wave[:].flatten()*1e-3, self.flux[:].flatten(), 'k-', lw=0.5)

            for j in range(self.n_chips):
                if j != 0:
                    label = None
                # Plot the spectrum and residuals
                idx_LogLike = LogLike.indices_per_model_setting[self.m_set][j]
                ax_flux.plot(self.wave[j]*1e-3, LogLike.m_flux_phi[idx_LogLike], 'C1-', lw=0.8, label=label)
                ax_res.plot(self.wave[j]*1e-3, self.flux[j]-LogLike.m_flux_phi[idx_LogLike], 'k-', lw=0.8)
                
            ax_res.axhline(0, c='C1', ls='-', lw=0.5)

            ylim = ax_res.get_ylim()
            ylim_max = np.max(np.abs(ylim))
            ylim = (-ylim_max, +ylim_max)
            ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])
            ax_res.set(xlim=xlim, xlabel=xlabel, ylim=ylim, ylabel=ylabel[1])

            for j in range(self.n_chips):
                # Plot the error envelope in the axes
                idx_LogLike = LogLike.indices_per_model_setting[self.m_set][j]
                sigma_scaled = np.sqrt(Cov[idx_LogLike].cov[0]*LogLike.s_squared[idx_LogLike])
                ax_res.fill_between(self.wave[j]*1e-3, -self.err[j], +self.err[j], fc='k', alpha=0.15, ec='none', zorder=-1)
                ax_res.fill_between(self.wave[j]*1e-3, -sigma_scaled, +sigma_scaled, fc='C1', alpha=0.15, ec='none', zorder=-1)

            if i == 0:
                ax_flux.legend()

        if LogLike.sum_model_settings:
            fig.savefig(plots_dir / f'bestfit_spectrum.pdf')
        else:
            fig.savefig(plots_dir / f'bestfit_spectrum_{self.m_set}.pdf')
        plt.close(fig)

        # Plot per chunk
        n_chunks = self.n_chips
        if (self.n_chips == 1) and (self.grating.lower() in ['g140h', 'g235h', 'g395h']):
            n_chunks = 4
        if n_chunks == 1:
            return

        fig, subfig = utils.get_subfigures_per_chip(n_chunks)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, (None, None)
            label = None
            if i == 0:
                xlabel = r'Wavelength ($\mathrm{\mu}$m)'
                ylabel = (r'$F_\lambda\ (\mathrm{W\ m^{-2}\ \mu m^{-1}})$', 'Res.')
                label  = r'$\chi_\mathrm{red}^2=$'+'{:.2f}'.format(LogLike.chi_squared_0_red)

            # Add some padding
            if self.n_chips == 1:
                xlim = np.linspace((self.wave_ranges_chips.min())*1e-3, (self.wave_ranges_chips.max())*1e-3, n_chunks+1)[i:i+2]
                i = 0
            else:
                xlim = ((self.wave_ranges_chips[i].min())*1e-3, (self.wave_ranges_chips[i].max())*1e-3)
            
            gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_res  = subfig_i.add_subplot(gs[1])

            ax_flux.fill_between(
                self.wave[i]*1e-3, self.flux[i]-self.err[i], self.flux[i]+self.err[i], 
                fc='k', alpha=0.2, ec='none'
                )
            ax_flux.plot(self.wave[i]*1e-3, self.flux[i], 'k-', lw=0.5)

            idx_LogLike = LogLike.indices_per_model_setting[self.m_set][i]
            ax_flux.plot(self.wave[i]*1e-3, LogLike.m_flux_phi[idx_LogLike], 'C1-', lw=0.8, label=label)

            ax_res.plot(self.wave[i]*1e-3, self.flux[i]-LogLike.m_flux_phi[idx_LogLike], 'k-', lw=0.8)
            ax_res.axhline(0, c='C1', ls='-', lw=0.5)

            ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])

            ylim = ax_res.get_ylim()
            ylim_max = np.max(np.abs(ylim))
            ylim = (-ylim_max, +ylim_max)
            ax_res.set(xlim=xlim, xlabel=xlabel, ylim=ylim, ylabel=ylabel[1])

            # Plot the error envelope in the axes
            sigma_scaled = np.sqrt(Cov[idx_LogLike].cov[0]*LogLike.s_squared[idx_LogLike])
            ax_res.fill_between(self.wave[i]*1e-3, -self.err[i], +self.err[i], fc='k', alpha=0.15, ec='none', zorder=-1)
            ax_res.fill_between(self.wave[i]*1e-3, -sigma_scaled, +sigma_scaled, fc='C1', alpha=0.15, ec='none', zorder=-1)

            if (i == 0) and (ax_flux.get_legend_handles_labels() != ([], [])):
                ax_flux.legend()

        if LogLike.sum_model_settings:
            fig.savefig(plots_dir / f'bestfit_spectrum_per_chunk.pdf')
        else:
            fig.savefig(plots_dir / f'bestfit_spectrum_per_chunk_{self.m_set}.pdf')
        plt.close(fig)
