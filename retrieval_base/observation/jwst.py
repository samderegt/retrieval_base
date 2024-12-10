import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

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

    @staticmethod
    def instrumental_broadening(wave, flux, resolution, initial_resolution):
        """
        Apply instrumental broadening to the spectrum.

        Args:
            wave (numpy.ndarray): Wavelength array.
            flux (numpy.ndarray): Flux array.
            resolution (float): Target resolution.
            initial_resolution (float): Initial resolution.

        Returns:
            numpy.ndarray: Broadened flux array.
        """
        # Delta lambda of resolution element is FWHM of the LSF's standard deviation
        sigma_LSF = np.sqrt(1/resolution**2 - 1/initial_resolution**2) / \
                    (2*np.sqrt(2*np.log(2)))

        spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))

        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF / spacing
        
        # Apply gaussian filter to broaden with the spectral resolution
        flux_LSF = gaussian_filter(
            flux, sigma=sigma_LSF_gauss_filter, mode='nearest'
            )
        return flux_LSF
    
    def __init__(self, file, wave_range, m_set, resolution=2700, **kwargs):
        """
        Initialize the SpectrumJWST instance.

        Args:
            file (str): File path to the spectrum data.
            wave_range (tuple): Wavelength range to consider.
            m_set (str): Model set identifier.
            resolution (int, optional): Spectral resolution. Defaults to 2700.
            **kwargs: Additional keyword arguments.
        """
        # Read info from wavelength settings
        self.m_set = m_set
        self.load_spectrum(file)
        
        self.resolution = resolution

        # Mask user-specified wavelength ranges
        self.mask_wavelength_ranges(kwargs.get('wave_to_mask'), pad=0.)

        self.n_chips  = 1
        self.n_pixels = len(self.wave)

        # Reshape and crop the spectrum
        self.reshape_spectrum()
        self.crop_spectrum(wave_range)
        self.remove_empty_chips(remove_empty_pixels=True)

        # Update the number of chips
        self.wave_ranges_chips = np.array([
            [wave.min(), wave.max()] for wave in self.wave
            ])
        self.n_chips = len(self.wave_ranges_chips)
    
    def load_spectrum(self, file):
        """
        Load the spectrum data from a file.

        Args:
            file (str): File path to the spectrum data.
        """
        # Load in the data of the target
        #self.wave, self.flux, self.err = np.loadtxt(file).T
        self.wave, self.flux, self.err = np.genfromtxt(file, delimiter=',').T

        mask_isnan = np.isnan(self.flux) | np.isnan(self.err)
        self.flux[mask_isnan] = np.nan
        self.err[mask_isnan]  = np.nan

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
        # Plot per chip
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

        fig.savefig(plots_dir / f'pre_processed_spectrum_{self.m_set}.pdf')
        plt.close(fig)

    def plot_bestfit(self, plots_dir, LogLike):
        """
        Plot the best-fit spectrum.

        Args:
            plots_dir (str): Directory to save the plots.
            LogLike (object): Log-likelihood object containing fit results.
        """
        # Plot per chip
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

            ylim = ax_res.get_ylim()
            ylim_max = np.max(np.abs(ylim))
            ylim = (-ylim_max, +ylim_max)
            ax_res.set(xlim=xlim, xlabel=xlabel, ylim=ylim, ylabel=ylabel[1])

            if i == 0:
                ax_flux.legend()

        if LogLike.sum_model_settings:
            fig.savefig(plots_dir / f'bestfit_spectrum.pdf')
        else:
            fig.savefig(plots_dir / f'bestfit_spectrum_{self.m_set}.pdf')
        plt.close(fig)