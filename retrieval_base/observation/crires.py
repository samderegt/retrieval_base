import numpy as np
from scipy.ndimage import gaussian_filter

from .spectrum import Spectrum
from ..utils import sc

def get_class(m_set, config_data_m_set):
    """
    Get the SpectrumCRIRES class instance.

    Args:
        m_set (str): Model set identifier.
        config_data_m_set (dict): Configuration data for the model set.

    Returns:
        SpectrumCRIRES: Instance of SpectrumCRIRES class.
    """
    # Load the target and standard-star spectra
    d_spec_target = SpectrumCRIRES(
        m_set=m_set, **config_data_m_set['target_kwargs'], **config_data_m_set['kwargs']
        )
    d_spec_std = SpectrumCRIRES(
        **config_data_m_set['std_kwargs'], **config_data_m_set['kwargs']
        )
    
    # Pre-process the data
    d_spec_target.telluric_correction(d_spec_std)
    d_spec_target.sigma_clip(**config_data_m_set['kwargs'])

    d_spec_target.flux_calibration(**config_data_m_set['target_kwargs'])
    d_spec_target.savgol_filter()

    return d_spec_target

class SpectrumCRIRES(Spectrum):
    """Class for handling CRIRES spectrum data."""

    wavelength_settings = {
        'J1226': np.array([
            [[1116.376, 1124.028], [1124.586, 1131.879], [1132.404, 1139.333]], 
            [[1139.175, 1146.984], [1147.551, 1154.994], [1155.521, 1162.592]], 
            [[1162.922, 1170.895], [1171.466, 1179.065], [1179.598, 1186.818]], 
            [[1187.667, 1195.821], [1196.391, 1204.153], [1204.700, 1212.078]], 
            [[1213.484, 1221.805], [1222.389, 1230.320], [1230.864, 1238.399]], 
            [[1240.463, 1248.942], [1249.534, 1257.642], [1258.205, 1265.874]], 
            [[1268.607, 1277.307], [1277.901, 1286.194], [1286.754, 1294.634]], 
            [[1298.103, 1306.964], [1307.579, 1316.065], [1316.608, 1324.672]], 
            [[1328.957, 1338.011], [1338.632, 1347.322], [1347.898, 1356.153]], 
            ]), 
        'K2166': np.array([
            [[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
            [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
            [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
            [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
            [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
            [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
            [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]],
            ]), 
        }
    ghosts = {
        'J1226': np.array([
            [1119.44,1120.33], [1142.78,1143.76], [1167.12,1168.08], 
            [1192.52,1193.49], [1219.01,1220.04], [1246.71,1247.76], 
            [1275.70,1276.80], [1306.05,1307.15], [1337.98,1338.94], 
            ])
    }
    n_pixels = 2048

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
    
    @staticmethod
    def clip_detector_edges(array, n_pixels=30):
        """
        Clip the edges of the detector array.

        Args:
            array (numpy.ndarray): Array to be clipped.
            n_pixels (int, optional): Number of pixels to clip. Defaults to 30.

        Returns:
            numpy.ndarray: Clipped array.
        """
        array[:,:n_pixels]  = np.nan
        array[:,-n_pixels:] = np.nan
        return array
    
    def __init__(self, file, file_wave, wave_range, w_set, slit, ra, dec, mjd, resolution=None, m_set=None, **kwargs):
        """
        Initialize the SpectrumCRIRES instance.

        Args:
            file (str): File path to the spectrum data.
            file_wave (str): File path to the wavelength data.
            wave_range (tuple): Wavelength range to consider.
            w_set (str): Wavelength setting identifier.
            slit (str): Slit identifier.
            ra (float): Right ascension.
            dec (float): Declination.
            mjd (float): Modified Julian Date.
            resolution (float, optional): Spectral resolution. Defaults to None.
            m_set (str, optional): Model set identifier. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        # Read info from wavelength settings
        self.m_set = m_set
        self.w_set = w_set
        
        self.wave_ranges_orders_dets = self.wavelength_settings[self.w_set]

        self.n_orders, self.n_dets, _ = self.wave_ranges_orders_dets.shape
        self.n_chips = self.n_orders * self.n_dets

        self.wave_ranges_chips = self.wave_ranges_orders_dets.reshape(self.n_chips, 2)

        self.load_spectrum_excalibuhr(file, file_wave)
        #self.load_spectrum_pycrires(file, file_wave)

        # Set the resolution
        self.set_resolution(slit, resolution)

        # Mask ghosts and user-specified wavelength ranges
        self.mask_wavelength_ranges(self.ghosts.get(self.w_set), pad=0.1)
        self.mask_wavelength_ranges(kwargs.get('wave_to_mask'), pad=0.)

        # Reshape and crop the spectrum
        self.reshape_spectrum()
        self.crop_spectrum(wave_range)
        self.flux = self.clip_detector_edges(self.flux)
        self.err  = self.clip_detector_edges(self.err)

        self.remove_empty_chips()

        # Barycentric velocity correction
        paranal_coords = {
            'obs_long': -70.403, 'obs_lat': -24.625, 'obs_alt': 2635
            }
        self.barycentric_correction(ra, dec, mjd, paranal_coords)

        if kwargs.get('file_molecfit_transm') is not None:
            self.load_molecfit_transmission(**kwargs)

        # Update the number of orders and chips
        self.wave_ranges_chips = self.wave_ranges_chips[self.non_empty_chips]
        self.n_chips = self.wave_ranges_chips.shape[0]
        
        mask = self.non_empty_chips.reshape(self.n_orders, self.n_dets)
        self.wave_ranges_orders_dets = self.wave_ranges_orders_dets[mask.any(axis=1),:,:]
        self.n_orders = self.wave_ranges_orders_dets.shape[0]

    def set_resolution(self, slit, resolution):
        """
        Set the spectral resolution.

        Args:
            slit (str): Slit identifier.
            resolution (float, optional): Spectral resolution. Defaults to None.

        Raises:
            ValueError: If slit-width is not recognized.
        """
        self.slit = slit
        self.resolution = resolution

        if self.resolution is not None:
            return

        # Set the resolution based on the slit-width        
        if self.slit == 'w_0.4':
            self.resolution = 5e4
        elif self.slit == 'w_0.2':
            self.resolution = 1e5
        else:
            raise ValueError('Slit-width not recognized')
        
    def load_spectrum_excalibuhr(self, file, file_wave):
        """
        Load the spectrum data reduced with excalibuhr.

        Args:
            file (str): File path to the spectrum data.
            file_wave (str): File path to the wavelength data.
        """
        # Load in the data of the target
        _, self.flux, self.err = np.loadtxt(file).T
        # Load in (different) wavelength solution
        self.wave, *_ = np.loadtxt(file_wave).T

        # Make a copy of the wavelengths before rv-shifts
        self.wave_initial = self.wave.copy()

        # Convert from [photons] to [erg nm^-1]
        self.flux /= self.wave
        self.err  /= self.wave

    def load_spectrum_pycrires(self, file):
        """
        Load the spectrum data reduced with PyCRIRES.

        Args:
            file (str): File path to the spectrum data.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError
    
    def load_molecfit_transmission(self, file_molecfit_transm, file_molecfit_continuum=None, T_BB=None, telluric_threshold=0.8, **kwargs):
        """
        Load the Molecfit transmission data.

        Args:
            file_molecfit_transm (str): File path to the Molecfit transmission data.
            file_molecfit_continuum (str, optional): File path to the Molecfit continuum data. Defaults to None.
            T_BB (float, optional): Blackbody temperature. Defaults to None.
            telluric_threshold (float, optional): Telluric threshold. Defaults to 0.8.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the wavelength solution of Molecfit and observation do not match.
        """
        # Load the pre-computed molecfit transmission
        self.wave_mf, self.transm_mf = np.loadtxt(file_molecfit_transm).T
        self.transm_mf = self.transm_mf.reshape(self.n_chips, self.n_pixels)
        self.transm_mf = self.clip_detector_edges(self.transm_mf)
        
        # Remove empty chips
        self.transm_mf = self.transm_mf[self.non_empty_chips]

        # Set the telluric threshold
        self.telluric_threshold = telluric_threshold

        # Confirm that we are using the same wavelength solution
        if not np.allclose(self.wave_initial, self.wave_mf):
            raise ValueError('Wavelength solution of molecfit and observation do not match')
        del self.wave_initial

        # Mask pixels at detector edge

        if (T_BB is None) or (file_molecfit_continuum is None):
            return
        
        # Load the pre-computed molecfit continuum
        _, continuum_mf = np.loadtxt(file_molecfit_continuum).T

        # Calculate the blackbody spectrum
        h = sc.h * 1e7  # erg * s (from J * s)
        c = sc.c * 1e2  # cm/s (from m/s)
        k = sc.k * 1e7  # erg/K (from J/K)
        bb = 2*h*c**2/(self.wave_mf*1e-7)**5 / (np.exp(h*c/((self.wave_mf*1e-7)*k*T_BB))-1)

        # Remove the blackbody spectrum from the continuum
        self.throughput_mf = (continuum_mf/self.wave_mf) / bb

        self.wave_mf = self.wave_mf.reshape(self.n_chips, self.n_pixels)
        self.wave_mf = self.wave_mf[self.non_empty_chips]

        self.throughput_mf = self.throughput_mf.reshape(self.n_chips, self.n_pixels)
        self.throughput_mf = self.clip_detector_edges(self.throughput_mf)
        self.throughput_mf = self.throughput_mf[self.non_empty_chips]

    def telluric_correction(self, std_spectrum):
        """
        Apply telluric correction to the spectrum.

        Args:
            std_spectrum (object): Standard star spectrum object.
        """
        # Correct for the instrumental response
        self.throughput_mf = std_spectrum.throughput_mf
        self.flux /= self.throughput_mf
        self.err  /= self.throughput_mf

        # Used in a figure
        self.uncorrected_flux = self.flux.copy()

        # Correct the telluric lines
        self.flux /= self.transm_mf
        self.err  /= self.transm_mf

        # Mask deepest telluric lines
        mask = self.transm_mf < self.telluric_threshold
        self.flux[mask] = np.nan
        self.err[mask]  = np.nan

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
        self.uncorrected_flux *= scaling_factor
        self.flux *= scaling_factor
        self.err  *= scaling_factor

    def sigma_clip(self, sigma_clip_sigma=3, sigma_clip_width=5, **kwargs):
        """
        Apply sigma clipping to the spectrum.

        Args:
            sigma_clip_sigma (int, optional): Sigma threshold for clipping. Defaults to 3.
            sigma_clip_width (int, optional): Width of the running median filter. Defaults to 5.
            **kwargs: Additional keyword arguments.
        """
        from scipy.ndimage import generic_filter

        # Apply a running median filter to the flux
        self.running_median_flux = np.array([
            generic_filter(flux_i, np.nanmedian, size=sigma_clip_width) \
            for flux_i in self.flux
            ])
        
        # Calculate the standard deviation of the residuals
        self.residuals = self.flux - self.running_median_flux
        std_residuals = np.nanstd(self.residuals, axis=-1, keepdims=True)

        # Mask where residuals > sigma * standard deviation
        self.sigma_clip_sigma = sigma_clip_sigma
        self.mask_sigma_clipped = np.abs(self.residuals) > sigma_clip_sigma*std_residuals
        self.flux[self.mask_sigma_clipped] = np.nan
        self.err[self.mask_sigma_clipped]  = np.nan

    def savgol_filter(self, **kwargs):
        """
        Apply Savitzky-Golay filter to the spectrum.

        Args:
            **kwargs: Additional keyword arguments.
        """
        #raise NotImplementedError
        return
    
    def plot_pre_processing(self, plots_dir):
        """
        Plot the pre-processed spectrum.

        Args:
            plots_dir (str): Directory to save the plots.
        """
        # Make some summary figures
        from . import figures_crires
        figures_crires.plot_telluric_correction(plots_dir, self)
        figures_crires.plot_sigma_clip(plots_dir, self)
        figures_crires.plot_spectrum_to_fit(plots_dir, self)

        # Only needed for figures
        del (
            self.running_median_flux,
            self.mask_sigma_clipped, 
            self.residuals, 
            self.sigma_clip_sigma
        )

    def plot_bestfit(self, plots_dir, LogLike):
        """
        Plot the best-fit spectrum.

        Args:
            plots_dir (str): Directory to save the plots.
            LogLike (object): Log-likelihood object containing fit results.
        """
        from . import figures_crires
        figures_crires.plot_bestfit(plots_dir, self, LogLike=LogLike)
