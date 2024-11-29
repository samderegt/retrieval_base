import numpy as np
import scipy.constants as sc

from .spectrum_new import Spectrum

class DataSpectrumCRIRES(Spectrum):

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

    def __init__(self, file, file_wave, wave_range, w_set, slit, ra, dec, mjd, resolution=None, **kwargs):

        # Read info from wavelength settings
        self.w_set = w_set
        self.wave_ranges_orders_dets = self.wavelength_settings[self.w_set]

        self.n_orders, self.n_dets, _ = self.wave_ranges_orders_dets.shape
        self.n_chips = self.n_orders * self.n_dets

        self.wave_ranges_chips = self.wave_ranges_orders_dets.reshape(self.n_chips, 2)

        self.load_spectrum_excalibuhr(file, file_wave)
        #self.load_spectrum_pycrires(file, file_wave)

        # Mask ghosts and user-specified wavlength ranges
        self.mask_wavelength_ranges(self.ghosts.get(self.w_set), pad=0.1)
        self.mask_wavelength_ranges(kwargs.get('wave_to_mask'), pad=0.)

        # Reshape and crop the spectrum
        self.reshape_spectrum()
        self.crop_spectrum(wave_range)
        self.clip_detector_edges()
        self.remove_empty_chips()

        # Barycentric velocity correction
        paranal_coords = {
            'obs_long': -70.403, 'obs_lat': -24.625, 'obs_alt': 2635
            }
        self.barycentric_correction(ra, dec, mjd, paranal_coords)

        if kwargs.get('file_molecfit_transm') is not None:
            self.load_molecfit_transmission(**kwargs)

        # Set the resolution
        self.set_resolution(slit, resolution)

    def set_resolution(self, slit, resolution):
        
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
        raise NotImplementedError
    
    def load_molecfit_transmission(self, file_molecfit_transm, file_molecfit_continuum=None, T_BB=None, telluric_threshold=0.8, **kwargs):

        # Load the pre-computed molecfit transmission
        self.wave_mf, self.transm_mf = np.loadtxt(file_molecfit_transm).T
        self.transm_mf = self.transm_mf.reshape(self.n_chips, self.n_pixels)

        # Set the telluric threshold
        self.telluric_threshold = telluric_threshold

        # Confirm that we are using the same wavelength solution
        if not np.allclose(self.wave_initial, self.wave_mf):
            raise ValueError('Wavelength solution of molecfit and observation do not match')
        del self.wave_initial

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

        self.wave_mf       = self.wave_mf.reshape(self.n_chips, self.n_pixels)
        self.throughput_mf = self.throughput_mf.reshape(self.n_chips, self.n_pixels)

    def telluric_correction(self, std_spectrum):

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

        from species import SpeciesInit
        from species.phot.syn_phot import SyntheticPhotometry
        #from species import SpeciesInit, SyntheticPhotometry

        # Initiate database
        SpeciesInit()

        # Get the filter response curve
        synphot = SyntheticPhotometry(filter_name)

        # Get the flux in [W m^-2 um^-1] from the reported magnitude
        flux, _ = synphot.magnitude_to_flux(magnitude)

        # Integrate the telluric-corrected spectrum over the filter curve
        filter_response = synphot.filter_interp(self.wave.flatten()*1e-3)
        integrand1 = filter_response*self.flux.flatten()
        integrand2 = filter_response

        mask = ~np.isnan(integrand1) & ~np.isnan(integrand2)
        integral1 = np.trapz(integrand1[mask], x=self.wave.flatten()[mask])
        integral2 = np.trapz(integrand2[mask], x=self.wave.flatten()[mask])

        integrated_flux = integral1 / integral2

        # Match the integrated flux to the reported flux
        scaling_factor = flux / integrated_flux
        self.uncorrected_flux *= scaling_factor
        self.flux *= scaling_factor
        self.err  *= scaling_factor

    def sigma_clip(self, sigma_clip_sigma=3, sigma_clip_width=5, **kwargs):
        
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
        self.mask_sigma_clipped = \
            np.abs(self.residuals) > sigma_clip_sigma*std_residuals
        self.flux[self.mask_sigma_clipped] = np.nan
        self.err[self.mask_sigma_clipped]  = np.nan

    def savgol_filter(self, **kwargs):
        #raise NotImplementedError
        return
    
    def make_figures(self, plots_dir, **kwargs):
        # Make some summary figures
        plot_telluric_correction(plots_dir, self)
        plot_sigma_clip(plots_dir, self)
        plot_spectrum_to_fit(plots_dir, self)

        del self.mask_sigma_clipped, self.residuals, self.running_median_flux
        del self.sigma_clip_sigma

import matplotlib.pyplot as plt

def get_subfigures_per_chip(N):

    fig = plt.figure(figsize=(10,3*N))
    gs = fig.add_gridspec(nrows=N)
    subfig = np.array([fig.add_subfigure(gs[i]) for i in range(N)])

    return fig, subfig

def plot_telluric_correction(plots_dir, d_spec):

    # Plot per order
    fig, subfig = get_subfigures_per_chip(d_spec.n_orders)
    for i, subfig_i in enumerate(subfig):

        xlabel, ylabel = None, (None, None)
        if i == 0:
            xlabel = 'Wavelength (nm)'
            ylabel = (r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 'Transm.')

        # Add some padding
        xlim = (
            d_spec.wave_ranges_orders_dets[i].min()-0.5, 
            d_spec.wave_ranges_orders_dets[i].max()+0.5
            )
        
        gs = subfig_i.add_gridspec(nrows=2, height_ratios=[1,0.5], hspace=0.)
        ax_flux   = subfig_i.add_subplot(gs[0])
        ax_transm = subfig_i.add_subplot(gs[1])

        for idx in range(i*d_spec.n_dets, (i+1)*d_spec.n_dets):
            ax_flux.plot(d_spec.wave[idx], d_spec.uncorrected_flux[idx], 'k-', lw=0.5, alpha=0.4)
            ax_flux.plot(d_spec.wave[idx], d_spec.flux[idx], 'k-', lw=0.7)
            
            ax_transm.plot(d_spec.wave[idx], d_spec.transm_mf[idx], 'k-', lw=0.5)
        
        ax_transm.axhline(d_spec.telluric_threshold, c='r', ls='--')
        
        ax_flux.set(xlim=xlim, ylabel=ylabel[0])
        ax_transm.set(xlim=xlim, xlabel=xlabel, ylim=(0,1.1), ylabel=ylabel[1])

    fig.savefig(plots_dir / f'telluric_correction_per_order_{d_spec.w_set}.pdf')
    plt.close(fig)

    # Plot for full spectrum
    fig = plt.figure(figsize=(10,4))
    gs = fig.add_gridspec(
        nrows=2, height_ratios=[1,0.5], hspace=0., 
        left=0.1, right=0.95, top=0.93, bottom=0.15,
        )
    ax_flux   = fig.add_subplot(gs[0])
    ax_transm = fig.add_subplot(gs[1])

    xlim = (
        d_spec.wave_ranges_orders_dets.min()-15, 
        d_spec.wave_ranges_orders_dets.max()+15
        )

    for idx in range(d_spec.n_chips):
        ax_flux.plot(d_spec.wave[idx], d_spec.uncorrected_flux[idx], 'k-', lw=0.5, alpha=0.4)
        ax_flux.plot(d_spec.wave[idx], d_spec.flux[idx], 'k-', lw=0.7)

        ax_transm.plot(d_spec.wave[idx], d_spec.transm_mf[idx], 'k-', lw=0.5)
    
    ax_transm.axhline(d_spec.telluric_threshold, c='r', ls='--')
    
    ax_flux.set(xlim=xlim, ylabel=r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$')
    ax_transm.set(xlim=xlim, xlabel='Wavelength (nm)', ylim=(0,1.1), ylabel='Transm.')

    fig.savefig(plots_dir / f'telluric_correction_{d_spec.w_set}.pdf')
    plt.close(fig)

def plot_sigma_clip(plots_dir, d_spec):

    valid_residuals = d_spec.residuals.copy()
    valid_residuals[d_spec.mask_sigma_clipped] = np.nan

    # Plot per order
    fig, subfig = get_subfigures_per_chip(d_spec.n_orders)
    for i, subfig_i in enumerate(subfig):
        
        xlabel, ylabel = None, (None, None)
        if i == 0:
            xlabel = 'Wavelength (nm)'
            ylabel = (r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$', 'Res.')

        # Add some padding
        xlim = (
            d_spec.wave_ranges_orders_dets[i].min()-0.5, 
            d_spec.wave_ranges_orders_dets[i].max()+0.5
            )

        gs = subfig_i.add_gridspec(nrows=2, height_ratios=[0.5,0.5], hspace=0.)
        ax_flux = subfig_i.add_subplot(gs[0])
        ax_res  = subfig_i.add_subplot(gs[1])

        for idx in range(i*d_spec.n_dets, (i+1)*d_spec.n_dets):
            ax_flux.plot(d_spec.wave[idx], d_spec.flux[idx], 'k-', lw=0.5)
            ax_flux.plot(d_spec.wave[idx], d_spec.running_median_flux[idx], 'r-', lw=0.7)

            ax_res.plot(d_spec.wave[idx], d_spec.residuals[idx], 'r-', lw=0.5)
            ax_res.plot(d_spec.wave[idx], valid_residuals[idx], 'k-', lw=0.7)

            sigma = d_spec.sigma_clip_sigma*np.nanstd(d_spec.residuals[idx])
            ax_res.plot(d_spec.wave[idx], -sigma*np.ones_like(d_spec.wave[idx]), 'r--', lw=0.5)
            ax_res.plot(d_spec.wave[idx], +sigma*np.ones_like(d_spec.wave[idx]), 'r--', lw=0.5)

        ax_res.axhline(0, c='r', ls='-', lw=0.5)

        ax_flux.set(xlim=xlim, ylabel=ylabel[0])
        ax_res.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel[1])

    fig.savefig(plots_dir / f'sigma_clipping_{d_spec.w_set}.pdf')
    plt.close(fig)

def plot_spectrum_to_fit(plots_dir, d_spec):

    # Plot per order
    fig, subfig = get_subfigures_per_chip(d_spec.n_orders)
    for i, subfig_i in enumerate(subfig):

        xlabel, ylabel = None, None
        if i == 0:
            xlabel = 'Wavelength (nm)'
            ylabel = r'$F_\lambda\ (\mathrm{erg\ s^{-1}\ cm^{-2}\ nm^{-1}})$'

        # Add some padding
        xlim = (
            d_spec.wave_ranges_orders_dets[i].min()-0.5, 
            d_spec.wave_ranges_orders_dets[i].max()+0.5
            )

        gs = subfig_i.add_gridspec(nrows=1)
        ax_flux = subfig_i.add_subplot(gs[0])
        for idx in range(i*d_spec.n_dets, (i+1)*d_spec.n_dets):
            ax_flux.plot(d_spec.wave[idx], d_spec.flux[idx], 'k-', lw=0.7)

        ax_flux.set(xlim=xlim, xlabel=xlabel, ylabel=ylabel)
    
    fig.savefig(plots_dir / f'pre_processed_spectrum_{d_spec.w_set}.pdf')
    plt.close(fig)