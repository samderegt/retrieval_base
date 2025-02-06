import numpy as np
from scipy.ndimage import gaussian_filter

from .spectrum import Spectrum
from ..utils import sc
from .. import utils

import matplotlib.pyplot as plt

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
        'H1567': np.array([
            [[1431.243, 1441.052], [1441.763, 1451.128], [1451.778, 1460.671]], 
            [[1468.933, 1478.995], [1479.723, 1489.331], [1489.999, 1499.122]], 
            [[1508.653, 1518.982], [1519.727, 1529.590], [1530.277, 1539.643]], 
            [[1550.568, 1561.178], [1561.942, 1572.073], [1572.779, 1582.399]], 
            [[1594.864, 1605.770], [1606.553, 1616.967], [1617.693, 1627.582]], 
            [[1641.751, 1652.969], [1653.772, 1664.484], [1665.232, 1675.405]], 
            [[1691.460, 1703.010], [1703.833, 1714.861], [1715.631, 1726.104]], 
            [[1744.249, 1756.148], [1756.992, 1768.353], [1769.149, 1779.939]], 
            ]), 
        'K2217': np.array([
            [[1968.506, 1980.625], [2038.848, 2051.392], [2114.377, 2127.377]], 
            [[2195.687, 2209.178], [2283.474, 2297.492], [2378.532, 2393.120]], 
            [[1981.503, 1993.009], [2052.295, 2064.205], [2128.310, 2140.654]], 
            [[2210.143, 2222.954], [2298.491, 2311.803], [2394.150, 2408.003]], 
            [[1993.799, 2004.659], [2065.026, 2076.268], [2141.507, 2153.159]], 
            [[2223.843, 2235.935], [2312.734, 2325.300], [2408.983, 2422.059]], 
            ]), 
        'K2192': np.array([
            [[1945.903, 1958.593], [2015.429, 2028.567], [2090.110, 2103.732]], 
            [[2170.514, 2184.625], [2257.283, 2271.974], [2351.270, 2366.529]], 
            [[2453.362, 2469.300], [1959.513, 1971.602], [2029.515, 2042.023]], 
            [[2104.689, 2117.655], [2185.627, 2199.084], [2273.005, 2286.985]], 
            [[2367.643, 2382.180], [2470.049, 2485.351], [1972.427, 1983.862]], 
            [[2042.891, 2054.735], [2118.558, 2130.829], [2200.025, 2212.757]], 
            [[2287.961, 2301.199], [2383.225, 2396.977], [2486.629, 2501.029]], 
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
        'K2148': np.array([
            [[1972.256, 1986.385], [2045.340, 2059.983], [2124.020, 2139.214]], 
            [[2208.960, 2224.749], [2300.945, 2317.377], [2400.875, 2418.003]], 
            [[1987.410, 2000.926], [2061.032, 2075.043], [2140.304, 2154.844]], 
            [[2225.886, 2240.997], [2318.557, 2334.281], [2419.210, 2435.596]], 
            [[2001.873, 2014.734], [2076.014, 2089.352], [2155.863, 2169.707]], 
            [[2242.063, 2256.446], [2335.367, 2350.327], [2436.646, 2452.223]], 
            ]), 
        'L3262': np.array([
            [[2885.632, 2906.906], [2908.446, 2928.829], [2930.266, 2949.685]], 
            [[3045.964, 3068.406], [3070.035, 3091.545], [3093.044, 3113.551]], 
            [[3225.143, 3248.895], [3250.607, 3273.370], [3274.954, 3296.657]], 
            [[3426.699, 3451.899], [3453.711, 3477.883], [3479.581, 3502.630]], 
            [[3655.022, 3681.910], [3683.850, 3709.611], [3711.404, 3735.980]], 
            [[3915.904, 3944.689], [3946.699, 3974.300], [3976.391, 4003.132]], 
            ]), 
        'L3340': np.array([
            [[2958.407, 2978.046], [2979.463, 2998.191], [2999.490, 3017.240]], 
            [[3122.778, 3143.499], [3144.993, 3164.757], [3166.122, 3184.854]], 
            [[3306.456, 3328.382], [3329.967, 3350.842], [3352.330, 3372.157]], 
            [[3513.062, 3536.345], [3538.014, 3560.226], [3561.772, 3582.824]], 
            [[3747.144, 3771.966], [3773.759, 3797.451], [3799.056, 3821.502]], 
            [[4014.595, 4041.163], [4043.047, 4068.385], [4070.157, 4094.191]], 
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
        self.plot_telluric_correction(plots_dir)
        self.plot_sigma_clip(plots_dir)
        self.plot_spectrum_to_fit(plots_dir)

        # Only needed for figures
        del self.running_median_flux, self.mask_sigma_clipped, self.residuals, self.sigma_clip_sigma

    def plot_telluric_correction(self, plots_dir):
        """
        Plot the telluric correction.

        Args:
            plots_dir (str): Directory to save the plots.
        """
        # Plot per order
        fig, subfig = utils.get_subfigures_per_chip(self.n_orders)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, (None, None)
            if i == 0:
                xlabel = 'Wavelength (nm)'
                ylabel = (r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 'Transm.')

            # Add some padding
            xlim = (self.wave_ranges_orders_dets[i].min()-0.5, self.wave_ranges_orders_dets[i].max()+0.5)
            
            gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
            ax_flux   = subfig_i.add_subplot(gs[0])
            ax_transm = subfig_i.add_subplot(gs[1])

            for idx in range(i*self.n_dets, (i+1)*self.n_dets):
                if idx == self.n_chips:
                    break
                ax_flux.plot(self.wave[idx], self.uncorrected_flux[idx], 'k-', lw=0.5, alpha=0.4)
                ax_flux.plot(self.wave[idx], self.flux[idx], 'k-', lw=0.7)
                
                ax_transm.plot(self.wave[idx], self.transm_mf[idx], 'k-', lw=0.5)
            
            ax_transm.axhline(self.telluric_threshold, c='r', ls='--')
            
            ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])
            ax_transm.set(xlim=xlim, xlabel=xlabel, ylim=(0,1.1), ylabel=ylabel[1])

        fig.savefig(plots_dir / f'telluric_correction_per_order_{self.m_set}.pdf')
        plt.close(fig)

        # Plot for full spectrum
        fig = plt.figure(figsize=(10,4))
        gs = fig.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
        ax_flux   = fig.add_subplot(gs[0])
        ax_transm = fig.add_subplot(gs[1])

        xlim = (self.wave_ranges_orders_dets.min()-15, self.wave_ranges_orders_dets.max()+15)

        for idx in range(self.n_chips):
            ax_flux.plot(self.wave[idx], self.uncorrected_flux[idx], 'k-', lw=0.5, alpha=0.4)
            ax_flux.plot(self.wave[idx], self.flux[idx], 'k-', lw=0.7)

            ax_transm.plot(self.wave[idx], self.transm_mf[idx], 'k-', lw=0.5)
        
        ax_transm.axhline(self.telluric_threshold, c='r', ls='--')
        
        ax_flux.set(xlim=xlim, ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$')
        ax_transm.set(xlim=xlim, xlabel='Wavelength (nm)', ylim=(0,1.1), ylabel='Transm.')

        fig.savefig(plots_dir / f'telluric_correction_{self.m_set}.pdf')
        plt.close(fig)

    def plot_sigma_clip(self, plots_dir):
        """
        Plot the sigma clipping results for each order.

        Args:
            plots_dir (str): Directory to save the plots.
        """
        valid_residuals = self.residuals.copy()
        valid_residuals[self.mask_sigma_clipped] = np.nan

        # Plot per order
        fig, subfig = utils.get_subfigures_per_chip(self.n_orders)
        for i, subfig_i in enumerate(subfig):
            
            xlabel, ylabel = None, (None, None)
            if i == 0:
                xlabel = 'Wavelength (nm)'
                ylabel = (r'$F\ (\mathrm{arb.\ units})$', 'Res.')

            # Add some padding
            xlim = (self.wave_ranges_orders_dets[i].min()-0.5, self.wave_ranges_orders_dets[i].max()+0.5)

            gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_res  = subfig_i.add_subplot(gs[1])

            for idx in range(i*self.n_dets, (i+1)*self.n_dets):
                if idx == self.n_chips:
                    break
                ax_flux.plot(self.wave[idx], self.residuals[idx]+self.running_median_flux[idx], 'k-', lw=0.5)
                ax_flux.plot(self.wave[idx], self.running_median_flux[idx], 'r-', lw=0.7)

                ax_res.plot(self.wave[idx], self.residuals[idx], 'r-', lw=0.5)
                ax_res.plot(self.wave[idx], valid_residuals[idx], 'k-', lw=0.7)

                sigma = self.sigma_clip_sigma*np.nanstd(self.residuals[idx])
                ax_res.plot(self.wave[idx], -sigma*np.ones_like(self.wave[idx]), 'r--', lw=0.5)
                ax_res.plot(self.wave[idx], +sigma*np.ones_like(self.wave[idx]), 'r--', lw=0.5)

            ax_res.axhline(0, c='r', ls='-', lw=0.5)

            ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])
            
            ylim = ax_res.get_ylim()
            ylim_max = np.max(np.abs(ylim))
            ylim = (-ylim_max, +ylim_max)
            ax_res.set(xlim=xlim, xlabel=xlabel, ylim=ylim, ylabel=ylabel[1])

        fig.savefig(plots_dir / f'sigma_clipping_{self.m_set}.pdf')
        plt.close(fig)
    
    def plot_spectrum_to_fit(self, plots_dir):
        """
        Plot the spectrum to be fitted for each order.

        Args:
            plots_dir (str): Directory to save the plots.
            d_spec (object): Spectrum object containing data.
        """
        # Plot per order
        fig, subfig = utils.get_subfigures_per_chip(self.n_orders)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, None
            if i == 0:
                xlabel = 'Wavelength (nm)'
                ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'

            # Add some padding
            xlim = (self.wave_ranges_orders_dets[i].min()-0.5, self.wave_ranges_orders_dets[i].max()+0.5)

            gs = subfig_i.add_gridspec(nrows=1)
            ax_flux = subfig_i.add_subplot(gs[0])
            for idx in range(i*self.n_dets, (i+1)*self.n_dets):
                if idx == self.n_chips:
                    break
                ax_flux.plot(self.wave[idx], self.flux[idx], 'k-', lw=0.7)

            ax_flux.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel)

        fig.savefig(plots_dir / f'pre_processed_spectrum_{self.m_set}.pdf')
        plt.close(fig)

    def plot_bestfit(self, plots_dir, LogLike, Cov):
        """
        Plot the best-fit spectrum for each order.

        Args:
            plots_dir (str): Directory to save the plots.
            LogLike (object): Log-likelihood object containing fit results.
            **kwargs: Additional keyword arguments.
        """
        # Plot per order
        fig, subfig = utils.get_subfigures_per_chip(self.n_orders)
        for i, subfig_i in enumerate(subfig):

            xlabel, ylabel = None, (None, None)
            if i == 0:
                xlabel = 'Wavelength (nm)'
                ylabel = (r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 'Res.')

            # Add some padding
            xlim = (self.wave_ranges_orders_dets[i].min()-0.5, self.wave_ranges_orders_dets[i].max()+0.5)
            
            gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.7,0.3], hspace=0.)
            ax_flux = subfig_i.add_subplot(gs[0])
            ax_res  = subfig_i.add_subplot(gs[1])

            diagonal_i, err_i = [], []
            for idx in range(i*self.n_dets, (i+1)*self.n_dets):
                if idx == self.n_chips:
                    break

                label = None
                if i==0 and idx==i*self.n_dets:
                    label = r'$\chi_\mathrm{red}^2=$'+'{:.2f}'.format(LogLike.chi_squared_0_red)

                ax_flux.plot(self.wave[idx], self.flux[idx], 'k-', lw=0.5)

                idx_LogLike = LogLike.indices_per_model_setting[self.m_set][idx]
                ax_flux.plot(self.wave[idx], LogLike.m_flux_phi[idx_LogLike], 'C1-', lw=0.8, label=label)

                ax_res.plot(self.wave[idx], self.flux[idx]-LogLike.m_flux_phi[idx_LogLike], 'k-', lw=0.8)
                ax_res.axhline(0, c='C1', ls='-', lw=0.5)

                diagonal_i.append(Cov[idx_LogLike].cov[0]*LogLike.s_squared[idx_LogLike])
                err_i.append(self.err[idx])

            ax_flux.set(xlim=xlim, xticks=[], ylabel=ylabel[0])

            sigma_scaled = np.nanmean(np.sqrt(np.concatenate(diagonal_i)))
            sigma_data   = np.nanmean(err_i)
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
            fig.savefig(plots_dir / f'bestfit_spectrum.pdf')
        else:
            fig.savefig(plots_dir / f'bestfit_spectrum_{self.m_set}.pdf')
        plt.close(fig)