import numpy as np
from scipy.ndimage import gaussian_filter

from .spectrum import Spectrum
from .. import utils

import matplotlib.pyplot as plt

def get_class(m_set, config_data_m_set):
    """
    Get the IGRINS class instance.

    Args:
        m_set (str): Model set identifier.
        config_data_m_set (dict): Configuration data for the model set.

    Returns:
        SpectrumIGRINS: Instance of SpectrumIGRINS class.
    """
    # Load the target and standard-star spectra
    d_spec_target = SpectrumIGRINS(m_set=m_set, **config_data_m_set['kwargs'])

    # Pre-process the data
    d_spec_target.flux_calibration(**config_data_m_set['target_kwargs'])

    return d_spec_target

class SpectrumIGRINS(Spectrum):
    """Class for handling IGRINS spectrum data."""

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

    def __init__(self, file, wave_range, m_set, resolution=None, n_chunks=1, **kwargs):
        """
        Initialize the IGRINS spectrum object.

        Args:
            file (str): File path to the spectrum data.
            wave_range (tuple): Wavelength range to crop the spectrum.
            m_set (str): Model setting name.
            resolution (float): Instrumental resolution.
            n_chunks (int): Number of chunks to split the spectrum into.
            **kwargs: Additional keyword arguments.
        """
        # Read info from wavelength settings
        self.m_set = m_set
        self.load_spectrum(file, min_SNR=kwargs.get('min_SNR', 0.))

        # Load a constant resolution
        if resolution is None:
            self.resolution = 45000

        # Mask user-specified wavelength ranges
        self.mask_wavelength_ranges(kwargs.get('wave_to_mask'), pad=0.)

        # Crop the spectrum
        self.crop_spectrum(wave_range)
        self.reshape_spectrum(n_chunks=n_chunks)

        self.remove_empty_chips(remove_empty_pixels=True)

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

    def load_spectrum(self, file, min_SNR=0.):
        """
        Load the spectrum data from a file.

        Args:
            file (str): File path to the spectrum data.
            min_SNR (float): Minimum signal-to-noise ratio to keep.
        """
        # Load in the data of the target
        self.wave, self.flux, self.err, SNR = np.loadtxt(file, delimiter=',').T

        mask_isnan = np.isnan(self.flux) | np.isnan(self.err)
        self.wave = self.wave[~mask_isnan]
        self.flux = self.flux[~mask_isnan]
        self.err  = self.err[~mask_isnan]

        mask_SNR = SNR[~mask_isnan] > min_SNR
        self.flux[~mask_SNR] = np.nan
        self.err[~mask_SNR]  = np.nan

        # Convert from [um] to [nm]
        self.wave *= 1e3

    def flux_calibration(self, filter_name, magnitude, **kwargs):
        """
        Calibrate the flux using a filter and magnitude.

        Args:
            filter_name (str): Name of the filter.
            magnitude (float): Magnitude of the target.
            **kwargs: Additional keyword arguments.
        """
        from species import SpeciesInit
        from species.phot.syn_phot import SyntheticPhotometry
        #from species import SpeciesInit, SyntheticPhotometry

        # Initiate database
        SpeciesInit()

        # Get the filter response curve
        synphot = SyntheticPhotometry(filter_name)

        # Get the flux in [W m^-2 um^-1] from the reported magnitude
        flux, _ = synphot.magnitude_to_flux(magnitude)

        # Mask the spectrum outside of the filter range
        filter_response = synphot.filter_interp(self.wave.flatten()*1e-3)
        mask = ~np.isnan(filter_response*self.flux.flatten())

        # Integrate the telluric-corrected spectrum over the filter curve
        integrated_flux, _ = synphot.spectrum_to_flux(
            self.wave.flatten()[mask]*1e-3, self.flux.flatten()[mask]
            )

        # Match the integrated flux to the reported flux
        scaling_factor = flux / integrated_flux
        #self.uncorrected_flux *= scaling_factor
        self.flux *= scaling_factor
        self.err  *= scaling_factor

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
            xlim = (self.wave_ranges_chips[i].min()-0.5, self.wave_ranges_chips[i].max()+0.5)

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
            xlim = (self.wave_ranges_chips[:].min()-0.5, self.wave_ranges_chips[:].max()+0.5)

            gs = subfig_i.add_gridspec(nrows=1)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_flux.plot(self.wave[:].flatten(), self.flux[:].flatten(), 'k-', lw=0.7)

            ax_flux.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel)

        fig.savefig(plots_dir / f'pre_processed_spectrum_{self.m_set}.pdf')
        plt.close(fig)

    def plot_bestfit(self, plots_dir, LogLike, Cov, **kwargs):
        """
        Plot the best-fit spectrum.

        Args:
            plots_dir (str): Directory to save the plots.
            LogLike (object): Log-likelihood object containing fit results.
            Cov (object): Covariance object containing uncertainties.
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
            xlim = (self.wave_ranges_chips[i].min()-0.5, self.wave_ranges_chips[i].max()+0.5)
            
            gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_res  = subfig_i.add_subplot(gs[1])

            ax_flux.plot(self.wave[i], self.flux[i], 'k-', lw=0.5)

            ax_flux.plot(self.wave[i], LogLike.m_flux_phi[self.m_set][i], 'C1-', lw=0.8, label=label)
            ax_res.plot(self.wave[i], self.flux[i]-LogLike.m_flux_phi[self.m_set][i], 'k-', lw=0.8)
            ax_res.axhline(0, c='C1', ls='-', lw=0.5)

            ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])

            # First column of banded covariance matrix is the diagonal
            sigma_scaled = np.nanmean(np.sqrt(Cov[self.m_set][i].cov[0]*LogLike.s_squared[self.m_set][i]))
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
            xlim = (self.wave_ranges_chips[:].min()-0.5, self.wave_ranges_chips[:].max()+0.5)
            
            gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_res  = subfig_i.add_subplot(gs[1])

            ax_flux.plot(self.wave[:].flatten(), self.flux[:].flatten(), 'k-', lw=0.5)

            for j in range(self.n_chips):
                if j != 0:
                    label = None

                ax_flux.plot(self.wave[j], LogLike.m_flux_phi[self.m_set][j], 'C1-', lw=0.8, label=label)
                ax_res.plot(self.wave[j], self.flux[j]-LogLike.m_flux_phi[self.m_set][j], 'k-', lw=0.8)
                
            ax_res.axhline(0, c='C1', ls='-', lw=0.5)

            ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])
            ax_res.set(xlim=xlim, xlabel=xlabel, ylim=ylim, ylabel=ylabel[1])

            if i == 0:
                ax_flux.legend()

        fig.savefig(plots_dir / f'bestfit_spectrum_{self.m_set}.pdf')
        plt.close(fig)