import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter, generic_filter
from scipy.interpolate import interp1d
from scipy.sparse import triu

import pickle
import os

from PyAstronomy import pyasl
import petitRADTRANS.nat_cst as nc
from petitRADTRANS.retrieval import rebin_give_width as rgw

import retrieval_base.auxiliary_functions as af
import retrieval_base.figures as figs

class Spectrum:

    # The wavelength ranges of each detector and order
    settings = {
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

    def __init__(self, wave, flux, err=None, w_set='K2166'):

        self.wave = wave
        self.flux = flux
        self.err  = err

        self.w_set = w_set
        self.order_wlen_ranges = self.settings[self.w_set]
        self.n_orders, self.n_dets, _ = self.order_wlen_ranges.shape

        # Make the isfinite mask
        self.update_isfinite_mask()

        self.high_pass_filtered = False

    def update_isfinite_mask(self, array=None):

        if array is None:
            self.mask_isfinite = np.isfinite(self.flux)
        else:
            self.mask_isfinite = np.isfinite(array)
        self.n_data_points = self.mask_isfinite.sum()

    def rv_shift(self, rv, wave=None, replace_wave=False):

        # Use the supplied wavelengths
        if wave is None:
            wave = np.copy(self.wave)

        # Apply a Doppler shift to the model spectrum
        wave_shifted = wave * (1 + rv/(nc.c*1e-5))
        if replace_wave:
            self.wave = wave_shifted
        
        return wave_shifted
    
    def high_pass_filter(self, removal_mode='divide', filter_mode='gaussian', sigma=300, replace_flux_err=False):

        # Prepare an array of low-frequency structure
        low_pass_flux = np.ones_like(self.flux) * np.nan

        for i in range(self.n_orders):
            for j in range(self.n_dets):

                # Apply high-pass filter to each detector separately
                mask_ij = self.mask_isfinite[i,j,:]
                if mask_ij.sum() != 0:
                    flux_ij = self.flux[i,j,:][mask_ij]

                    if filter_mode == 'gaussian':
                        # Find low-frequency structure
                        low_pass_flux[i,j,mask_ij] = gaussian_filter1d(flux_ij, sigma=sigma, mode='reflect')

                    elif filter_mode == 'savgol':
                        # TODO: savgol filter
                        pass

        if removal_mode == 'divide':
            # Divide out the low-frequency structure
            high_pass_flux = self.flux / low_pass_flux
            if self.err is not None:
                high_pass_err = self.err / low_pass_flux
            else:
                high_pass_err = None

        elif removal_mode == 'subtract':
            # Subtract away the low-frequency structure
            high_pass_flux = self.flux - low_pass_flux
            if self.err is not None:
                high_pass_err = self.err
            else:
                high_pass_err = None

        if replace_flux_err:
            self.flux = high_pass_flux
            self.err = high_pass_err

            self.high_pass = True

        return high_pass_flux, high_pass_err

    def sigma_clip_poly(self, sigma=5, poly_deg=1, replace_flux=True, prefix=None):

        flux_copy = self.flux.copy()

        sigma_clip_bounds = np.ones((3, self.n_orders, 3*self.n_pixels)) * np.nan

        # Loop over the orders
        for i in range(self.n_orders):
                
            # Select only pixels within the order, should be 3*2048
            idx_low  = self.n_pixels * (i*self.n_dets)
            idx_high = self.n_pixels * ((i+1)*self.n_dets)
            
            mask_wave = np.zeros_like(self.wave, dtype=bool)
            mask_wave[idx_low:idx_high] = True

            mask_order = (mask_wave & self.mask_isfinite)

            if mask_order.any():

                flux_i = flux_copy[mask_order]
                
                # Fit an n-th degree polynomial to this order
                p = np.polyfit(self.wave[mask_order], flux_i, 
                               w=1/self.err[mask_order], deg=poly_deg)

                # Polynomial representation of order                
                poly_model = np.poly1d(p)(self.wave[mask_order])

                # Subtract the polynomial approximation
                residuals = flux_i - poly_model

                # Sigma-clip the residuals
                mask_clipped = (np.abs(residuals) > sigma*np.std(residuals))

                sigma_clip_bounds[1,i] = np.poly1d(p)(self.wave[mask_wave])
                sigma_clip_bounds[0,i] = sigma_clip_bounds[1,i] - sigma*np.std(residuals)
                sigma_clip_bounds[2,i] = sigma_clip_bounds[1,i] + sigma*np.std(residuals)

                # Set clipped values to NaNs
                flux_i[mask_clipped]  = np.nan
                flux_copy[mask_order] = flux_i

        # Plot the sigma-clipping procedure
        figs.fig_sigma_clip(wave=self.wave, 
                            flux=flux_copy, 
                            flux_wo_clip=self.flux, 
                            sigma_clip_bounds=sigma_clip_bounds,
                            order_wlen_ranges=self.order_wlen_ranges, 
                            sigma=sigma, 
                            prefix=prefix, 
                            )

        if replace_flux:
            self.flux = flux_copy

            # Update the isfinite mask
            self.update_isfinite_mask()

        return flux_copy

    def sigma_clip_median_filter(self, sigma=3, filter_width=3, replace_flux=True, prefix=None):

        flux_copy = self.flux.copy()
        sigma_clip_bounds = np.ones((3, self.n_orders, 3*self.n_pixels)) * np.nan

        # Loop over the orders
        for i in range(self.n_orders):

            # Select only pixels within the order, should be 3*2048
            idx_low  = self.n_pixels * (i*self.n_dets)
            idx_high = self.n_pixels * ((i+1)*self.n_dets)
            
            mask_wave = np.zeros_like(self.wave, dtype=bool)
            mask_wave[idx_low:idx_high] = True
            
            mask_order = (mask_wave & self.mask_isfinite)

            if mask_order.any():

                flux_i = flux_copy[mask_order]

                # Apply a median filter to this order
                filtered_flux_i = generic_filter(flux_i, np.nanmedian, size=filter_width)
                
                # Subtract the filtered flux
                residuals = flux_i - filtered_flux_i

                # Sigma-clip the residuals
                mask_clipped = (np.abs(residuals) > sigma*np.std(residuals))

                sigma_clip_bounds[1,i,self.mask_isfinite[mask_wave]] = filtered_flux_i
                sigma_clip_bounds[0,i] = sigma_clip_bounds[1,i] - sigma*np.std(residuals)
                sigma_clip_bounds[2,i] = sigma_clip_bounds[1,i] + sigma*np.std(residuals)

                # Set clipped values to NaNs
                flux_i[mask_clipped]  = np.nan
                flux_copy[mask_order] = flux_i

        # Plot the sigma-clipping procedure
        figs.fig_sigma_clip(wave=self.wave, 
                            flux=flux_copy, 
                            flux_wo_clip=self.flux, 
                            sigma_clip_bounds=sigma_clip_bounds,
                            order_wlen_ranges=self.order_wlen_ranges, 
                            sigma=sigma, 
                            prefix=prefix, 
                            w_set=self.w_set, 
                            )

        if replace_flux:
            self.flux = flux_copy

            # Update the isfinite mask
            self.update_isfinite_mask()

        return flux_copy
    
    def rot_broadening(self, vsini, epsilon_limb=0, wave=None, flux=None, replace_wave_flux=False):

        if wave is None:
            wave = self.wave
        if flux is None:
            flux = self.flux

        # Evenly space the wavelength grid
        wave_even = np.linspace(wave.min(), wave.max(), wave.size)
        flux_even = np.interp(wave_even, xp=wave, fp=flux)
        
        # Rotational broadening of the model spectrum
        flux_rot_broad = pyasl.fastRotBroad(wave_even, flux_even, 
                                            epsilon=epsilon_limb, 
                                            vsini=vsini
                                            )
        if replace_wave_flux:
            self.wave = wave_even
            self.flux = flux_rot_broad
        
            return flux_rot_broad
        
        else:
            return wave_even, flux_rot_broad

    @classmethod
    def instr_broadening(cls, wave, flux, out_res=1e6, in_res=1e6):

        # Delta lambda of resolution element is FWHM of the LSF's standard deviation
        sigma_LSF = np.sqrt(1/out_res**2 - 1/in_res**2) / \
                    (2*np.sqrt(2*np.log(2)))

        spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))

        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF / spacing
        
        # Apply gaussian filter to broaden with the spectral resolution
        flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter, 
                                   mode='nearest'
                                   )
        return flux_LSF
    
    @classmethod
    def spectrally_weighted_integration(cls, wave, flux, array):

        # Integrate and weigh the array by the spectrum
        integral1 = np.trapz(wave*flux*array, wave)
        integral2 = np.trapz(wave*flux, wave)

        return integral1/integral2

    
class DataSpectrum(Spectrum):

    def __init__(self, 
                 wave, 
                 flux, 
                 err, 
                 ra, 
                 dec, 
                 mjd, 
                 pwv, 
                 file_target=None, 
                 file_wave=None, 
                 slit='w_0.2', 
                 wave_range=(1900,2500), 
                 w_set='K2166', 
                 ):

        if file_target is not None:
            wave, flux, err = self.load_spectrum_excalibuhr(file_target, file_wave)
        
        super().__init__(wave, flux, err, w_set)

        # Reshape the orders and detectors
        #self.reshape_orders_dets()

        self.ra, self.dec, self.mjd, self.pwv = ra, dec, mjd, pwv

        # Set to None initially
        self.transm, self.transm_err = None, None

        # Get the spectral resolution
        self.slit = slit
        if self.slit == 'w_0.2':
            self.resolution = 1e5
        elif self.slit == 'w_0.4':
            self.resolution = 5e4

        self.wave_range = wave_range

        # Set to None by default
        self.separation = None
        self.err_eff, self.flux_eff = None, None

    def load_spectrum_excalibuhr(self, file_target, file_wave=None):

        # Load in the data of the target
        if not isinstance(file_target, (list, np.ndarray)):
            wave, flux, err = np.loadtxt(file_target).T

        else:
            # Combine multiple runs
            wave, flux, err = [], [], []
            for file_i in file_target:
                wave_i, flux_i, err_i = np.loadtxt(file_i).T
            
                wave.append(wave_i)
                flux.append(flux_i)
                err.append(err_i)

                # TODO: apply wavelength correction for each run and then combine
                # (or fit for the nights separately?)
            
            wave = np.nanmean(np.array(wave), axis=0)
            flux = np.nansum(np.array(flux), axis=0)
            err  = np.nansum(np.array(err)**2, axis=0)**(1/2)

        # Load in (other) corrected wavelengths
        if file_wave is not None:
            wave, _, _ = np.loadtxt(file_wave).T

        
        wave_bins = np.zeros_like(wave.flatten())
        wave_bins[:-1] = np.diff(wave.flatten())
        wave_bins[-1]  = wave_bins[-2]

        # Convert from [photons] to [erg nm^-1]
        flux /= wave
        #flux /= wave_bins
        #flux /= wave

        err /= wave
        #err /= wave_bins
        #err /= wave

        return wave, flux, err

    def crop_spectrum(self):

        # Crop the spectrum to within a given wavelength range
        mask_wave = (self.wave >= self.wave_range[0]) & \
                    (self.wave <= self.wave_range[1])

        self.flux[~mask_wave] = np.nan

    def mask_ghosts(self, wave_to_mask=None):
        
        # Mask user-specified lines
        if wave_to_mask is not None:
            for (wave_min, wave_max) in wave_to_mask:

                mask_wave = (self.wave >= wave_min-0.1) & \
                    (self.wave <= wave_max+0.1)
                self.flux[mask_wave] = np.nan

        if self.ghosts.get(self.w_set) is None:
            return
            
        # Loop over all segments of the ghost signature
        for (wave_min, wave_max) in self.ghosts.get(self.w_set):

            mask_wave = (self.wave >= wave_min-0.1) & \
                (self.wave <= wave_max+0.1)
            self.flux[mask_wave] = np.nan

    def bary_corr(self, replace_wave=True, return_v_bary=False):

        # Barycentric velocity (using Paranal coordinates)
        self.v_bary, _ = pyasl.helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, 
                                       ra2000=self.ra, dec2000=self.dec, 
                                       jd=self.mjd+2400000.5
                                       )
        print('Barycentric velocity: {:.2f} km/s'.format(self.v_bary))
        if return_v_bary:
            return self.v_bary

        # Apply barycentric correction
        wave_shifted = self.rv_shift(self.v_bary, replace_wave=replace_wave)
        return wave_shifted

    def reshape_orders_dets(self):

        # Ordered arrays of shape (n_orders, n_dets, n_pixels)
        wave_ordered = np.ones((self.n_orders, self.n_dets, self.n_pixels)) * np.nan
        flux_ordered = np.copy(wave_ordered)
        err_ordered  = np.copy(wave_ordered)
        transm_ordered = np.copy(wave_ordered)
        flux_uncorr_ordered = np.copy(wave_ordered)

        # Loop over the orders and detectors
        for i in range(self.n_orders):
            for j in range(self.n_dets):

                # Select only pixels within the detector, should be 2048
                mask_wave = np.arange(
                    self.n_pixels * (i*self.n_dets + j), 
                    self.n_pixels * (i*self.n_dets + j + 1), 
                    dtype=int
                    )

                if mask_wave.any():
                    wave_ordered[i,j] = self.wave[mask_wave]
                    flux_ordered[i,j] = self.flux[mask_wave]
                    err_ordered[i,j]  = self.err[mask_wave]

                    if self.transm is not None:
                        transm_ordered[i,j] = self.transm[mask_wave]
                    if self.flux_uncorr is not None:
                        flux_uncorr_ordered[i,j] = self.flux_uncorr[mask_wave]

        self.wave = wave_ordered
        self.flux = flux_ordered
        self.err  = err_ordered
        self.transm = transm_ordered
        self.flux_uncorr = flux_uncorr_ordered

        # Remove empty orders / detectors
        self.clear_empty_orders_dets()

        # Update the isfinite mask
        self.update_isfinite_mask()

    def clear_empty_orders_dets(self):

        # If all pixels are NaNs within an order...
        mask_empty = (~np.isfinite(self.flux)).all(axis=(1,2))
        
        # ... remove that order
        self.wave = self.wave[~mask_empty,:,:]
        self.flux = self.flux[~mask_empty,:,:]
        self.err  = self.err[~mask_empty,:,:]
        self.transm = self.transm[~mask_empty,:,:]
        self.flux_uncorr = self.flux_uncorr[~mask_empty,:,:]

        # Update the wavelength ranges for this instance
        self.order_wlen_ranges = self.order_wlen_ranges[~mask_empty]

        # Update the number of orders, detectors, and pixels 
        # for this instance
        self.n_orders, self.n_dets, self.n_pixels = self.flux.shape

    def prepare_for_covariance(self, prepare_err_eff=False):

        # Make a nested array of ndarray objects with different shapes
        self.separation = np.empty((self.n_orders, self.n_dets), dtype=object)

        if prepare_err_eff:
            self.err_eff = np.empty((self.n_orders, self.n_dets), dtype=object)
            self.flux_eff = np.empty((self.n_orders, self.n_dets), dtype=object)
        
        # Loop over the orders and detectors
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                
                # Mask the arrays, on-the-spot is slower
                mask_ij = self.mask_isfinite[i,j]
                wave_ij = self.wave[i,j,mask_ij]

                # Wavelength separation between all pixels within order/detector
                separation_ij = np.abs(wave_ij[None,:] - wave_ij[:,None])
                # Velocity separation in km/s
                #separation_ij = 2 * nc.c*1e-5 * np.abs(
                #    (wave_ij[None,:]-wave_ij[:,None]) / (wave_ij[None,:]+wave_ij[:,None])
                #    )
                self.separation[i,j] = separation_ij

                if prepare_err_eff:
                    # Arithmetic mean of the squared flux-errors
                    err_ij  = self.err[i,j,mask_ij]
                    #self.err_eff[i,j] = np.sqrt(1/2*(err_ij[None,:]**2 + err_ij[:,None]**2))
                    self.err_eff[i,j] = np.mean(err_ij)

                    #flux_ij  = self.flux[i,j,mask_ij]
                    #self.err_eff[i,j] = np.std(flux_ij)
                    #self.flux_eff[i,j] = np.sqrt(1/2*(flux_ij[None,:]**2 + flux_ij[:,None]**2))
                    

    def clip_det_edges(self, n_edge_pixels=30):
        
        # Loop over the orders and detectors
        for i in range(self.n_orders):
            for j in range(self.n_dets):

                idx_low  = self.n_pixels * (i*self.n_dets + j)
                idx_high = self.n_pixels * (i*self.n_dets + j + 1)

                self.flux[idx_low : idx_low + n_edge_pixels]   = np.nan
                self.flux[idx_high : idx_high - n_edge_pixels] = np.nan

        # Update the isfinite mask
        self.update_isfinite_mask()

    def get_transmission(self, T=10000, log_g=3.5, ref_rv=0, ref_vsini=1, mode='bb'):

        lines_to_mask = [1282.0, 1945.09,2166.12]
        mask_width = [7, 10,10]
        #mask_width = [7, 10,30]
        #mask_width = [7, 10,5]

        # Get the barycentric velocity during the standard observation
        v_bary = self.bary_corr(return_v_bary=True)

        if mode == 'bb':
            
            # Retrieve a Planck spectrum for the given temperature
            ref_flux = 2*nc.h*nc.c**2/(self.wave.flatten()**5) * \
                       1/(np.exp(nc.h*nc.c/(self.wave*nc.kB*T)) - 1)

        elif mode == 'PHOENIX':

            T_to_use = T
            if T > 12000:
                T_to_use = 12000

            # Download or read a PHOENIX model spectrum
            ref_wave, ref_flux = af.get_PHOENIX_model(
                T_to_use, log_g, FeH=0, 
                wave_range=(self.wave.min()-100,self.wave.max()+100)
                )

            # Apply RV + barycentric shifts
            ref_wave = self.rv_shift(rv=ref_rv-v_bary, wave=ref_wave)

            # Apply rotational and instrumental broadening
            ref_wave, ref_flux = self.rot_broadening(
                ref_vsini, epsilon_limb=0.3, 
                #ref_vsini, epsilon_limb=0.6, 
                wave=ref_wave, flux=ref_flux, 
                replace_wave_flux=False
                )
            ref_flux = self.instr_broadening(
                ref_wave, ref_flux, out_res=self.resolution, in_res=500000
                )

            # Interpolate onto the data's wavelength grid
            ref_flux = np.interp(self.wave.flatten(), ref_wave, ref_flux)

            # Change the slope with a blackbody spectrum of a different temperature
            ref_flux *= (np.exp(nc.h*nc.c/(self.wave*1e-7*nc.kB*T_to_use)) - 1) / \
                (np.exp(nc.h*nc.c/(self.wave*1e-7*nc.kB*T)) - 1)

        # Mask the standard star's hydrogen lines
        for line_i, mask_width_i in zip(lines_to_mask, mask_width):
            mask = (
                (self.wave.flatten() > line_i - mask_width_i/2) & \
                (self.wave.flatten() < line_i + mask_width_i/2)
                )
            ref_flux[mask] = np.nan
        
        # Retrieve and normalize the transmissivity
        self.transm = self.flux / ref_flux
        self.transm /= np.nanmax(self.transm)

        self.transm_err = self.err / ref_flux 
        self.transm_err /= np.nanmax(self.flux / ref_flux)

    def add_transmission(self, transm, transm_err):
        
        # Add transmission (e.g. from another instance) as attributes
        self.transm     = transm
        self.transm_err = transm_err

        # Update the isfinite mask
        self.update_isfinite_mask(transm)

    def flux_calib_2MASS(self, photom_2MASS, filter_2MASS, tell_threshold=0.2, replace_flux_err=True, prefix=None, file_skycalc_transm=None):

        # Retrieve an approximate telluric transmission spectrum
        wave_skycalc, transm_skycalc = self.get_skycalc_transm(
            file_skycalc_transm=file_skycalc_transm
            )

        # Interpolate onto the data wavelength grid
        transm_skycalc = np.interp(self.wave, xp=wave_skycalc, fp=transm_skycalc)

        #mask_high_transm = (transm_skycalc > 0.99)
        #mask_high_transm = (transm_skycalc > 0.98)
        mask_high_transm = (transm_skycalc > 0.95)

        poly_model = np.nan * np.ones_like(self.wave)
        for j in range(self.n_dets):

            # Select the spectrum within this order
            mask_det = np.zeros_like(self.wave, dtype=bool)

            for i in range(self.n_orders):
                wave_min, wave_max = self.order_wlen_ranges[i,j]
                mask_det[(self.wave.flatten() >= wave_min-0.5) & \
                    (self.wave.flatten() <= wave_max+0.5)] = True

            # Linear fit to the continuum, where telluric absorption is minimal
            p = np.polyfit(
                self.wave[mask_det & mask_high_transm & self.mask_isfinite].flatten(), 
                self.transm[mask_det & mask_high_transm & self.mask_isfinite].flatten(), deg=1
                )
            poly_model[mask_det] = np.poly1d(p)(self.wave[mask_det])

        # Apply correction for telluric transmission
        tell_corr_flux = self.flux / self.transm
        # Replace the deepest tellurics with NaNs
        tell_corr_flux[(self.transm/poly_model / np.nanmax(self.transm/poly_model)) < tell_threshold] = np.nan
        # Update the NaN mask
        self.update_isfinite_mask(tell_corr_flux)

        tell_corr_err = np.sqrt((self.err/self.transm)**2 + \
                                (tell_corr_flux*self.transm_err/self.transm)**2
                                )

        # Read in the transmission curve of the broadband instrument
        wave_2MASS, transm_2MASS = photom_2MASS.transm_curves[filter_2MASS].T
        # Interpolate onto the CRIRES wavelength grid
        interp_func = interp1d(wave_2MASS, transm_2MASS, kind='linear', 
                               bounds_error=False, fill_value=0.0)
        transm_2MASS = interp_func(self.wave)

        # Apply broadband transmission to the CRIRES spectrum
        integrand1 = (tell_corr_flux*transm_2MASS)[self.mask_isfinite]
        integral1  = np.trapz(integrand1, self.wave[self.mask_isfinite])
            
        integrand2 = transm_2MASS[self.mask_isfinite]
        integral2  = np.trapz(integrand2, self.wave[self.mask_isfinite])

        # Broadband flux if spectrum was observed with broadband instrument
        broadband_flux_CRIRES = integral1 / integral2

        # Conversion factor turning [counts] -> [erg s^-1 cm^-2 nm^-1]
        calib_factor = photom_2MASS.fluxes[filter_2MASS][0] / broadband_flux_CRIRES

        # Apply the flux calibration
        calib_flux = tell_corr_flux * calib_factor
        calib_err  = tell_corr_err * calib_factor

        # Plot the flux calibration
        figs.fig_flux_calib_2MASS(
            wave=self.wave, 
            calib_flux=calib_flux, 
            #calib_flux_wo_tell_corr=self.flux*poly_model*calib_flux, 
            calib_flux_wo_tell_corr=self.flux*calib_factor, 
            #transm=self.transm/poly_model, 
            transm=self.transm, 
            poly_model=poly_model, 
            wave_2MASS=wave_2MASS, 
            transm_2MASS=photom_2MASS.transm_curves[filter_2MASS].T[1], 
            tell_threshold=tell_threshold, 
            order_wlen_ranges=self.order_wlen_ranges, 
            prefix=prefix, 
            w_set=self.w_set, 
            )

        self.transm /= poly_model
        #self.transm /= np.nanmax(self.transm)
        
        self.flux_uncorr = None
        if replace_flux_err:
            self.flux_uncorr = (self.flux/poly_model)*calib_factor

            self.flux = calib_flux
            self.err  = calib_err

        return calib_flux, calib_err

    def get_skycalc_transm(self, resolution_skycalc=2e5, file_skycalc_transm=None):

        if file_skycalc_transm is not None:
            if os.path.exists(file_skycalc_transm):
                # Load the skycalc transmissivity
                wave_skycalc, transm_skycalc = np.loadtxt(file_skycalc_transm).T
            
                return wave_skycalc, transm_skycalc
            
        # Download the skycalc transmissivity
        import skycalc_ipy

        sky_calc = skycalc_ipy.SkyCalc()
        sky_calc.get_almanac_data(ra=self.ra, dec=self.dec, date=None, mjd=self.mjd, 
                                  observatory='paranal', update_values=True
                                  )
        
        # See https://skycalc-ipy.readthedocs.io/en/latest/GettingStarted.html
        sky_calc['msolflux'] = 130

        # K-band
        sky_calc['wmin'] = self.wave.min()-100  # (nm)
        sky_calc['wmax'] = self.wave.max()+100  # (nm)

        sky_calc['wgrid_mode'] = 'fixed_spectral_resolution'
        sky_calc['wres'] = resolution_skycalc
        sky_calc['pwv']  = self.pwv

        # Get the telluric spectrum from skycalc
        wave_skycalc, transm_skycalc, _ = sky_calc.get_sky_spectrum(return_type='arrays')
        wave_skycalc   = wave_skycalc.flatten()
        transm_skycalc = transm_skycalc.flatten()
        
        # Convert [um] -> [nm]
        wave_skycalc = wave_skycalc.value# * 1e3

        # Apply instrumental broadening
        transm_skycalc = self.instr_broadening(
            wave=wave_skycalc, 
            flux=transm_skycalc, 
            out_res=self.resolution, 
            in_res=sky_calc['wres']
            )

        if file_skycalc_transm is not None:
            # Save the skycalc transmissivity
            np.savetxt(
                file_skycalc_transm, 
                np.concatenate((wave_skycalc[:,None], 
                                transm_skycalc[:,None]), axis=1)
                )

        return wave_skycalc, transm_skycalc

class ModelSpectrum(Spectrum):

    def __init__(self, 
                 wave, 
                 flux, 
                 lbl_opacity_sampling=1, 
                 multiple_orders=False, 
                 high_pass_filtered=False, 
                 ):

        super().__init__(wave, flux)

        if multiple_orders:
            # New instance is combination of previous (order) instances
            assert(self.wave.ndim == 3)
            assert(self.flux.ndim == 3)

            # Update the shape of the model spectrum
            self.n_orders, self.n_dets, self.n_pixels = self.flux.shape

            # Update whether the orders were high-pass filtered
            self.high_pass_filtered = high_pass_filtered

            # Update the order wavelength ranges
            mask_order_wlen_ranges = \
                (self.order_wlen_ranges.min(axis=(1,2)) > self.wave.min() - 5) & \
                (self.order_wlen_ranges.max(axis=(1,2)) < self.wave.max() + 5)
                
            self.order_wlen_ranges = self.order_wlen_ranges[mask_order_wlen_ranges,:,:]

        # Model resolution depends on the opacity sampling
        self.resolution = int(1e6/lbl_opacity_sampling)

    def rebin(self, d_wave, replace_wave_flux=False):

        # Interpolate onto the observed spectrum's wavelength grid
        flux_rebinned = np.interp(d_wave, xp=self.wave, fp=self.flux)

        if replace_wave_flux:
            self.flux = flux_rebinned
            self.wave = d_wave

            # Update the isfinite mask
            self.update_isfinite_mask()
        
        return flux_rebinned

    def shift_broaden_rebin(self, 
                            rv, 
                            vsini, 
                            epsilon_limb=0, 
                            out_res=1e6, 
                            in_res=1e6, 
                            d_wave=None, 
                            rebin=True, 
                            ):

        # Apply Doppler shift, rotational/instrumental broadening, 
        # and rebin onto a new wavelength grid
        self.rv_shift(rv, replace_wave=True)
        self.rot_broadening(vsini, epsilon_limb, replace_wave_flux=True)
        self.flux = self.instr_broadening(self.wave, self.flux, out_res, in_res)
        if rebin:
            self.rebin(d_wave, replace_wave_flux=True)


class Photometry:

    def __init__(self, magnitudes):

        # Magnitudes of 2MASS, MKO, WISE, etc.
        self.magnitudes = magnitudes
        # Filter names
        self.filters = list(self.magnitudes.keys())

        # Convert the magnitudes to broadband fluxes
        self.mag_to_flux_conversion()

        # Retrieve the filter transmission curves
        self.get_transm_curves()

    def mag_to_flux_conversion(self):

        # Mag-to-flux conversion using the species package
        import species
        species.SpeciesInit()

        self.fluxes = {}
        for filter_i in self.filters:
            # Convert the magnitude to a flux density and propagate the error
            synphot = species.SyntheticPhotometry(filter_i)
            self.fluxes[filter_i] = synphot.magnitude_to_flux(*self.magnitudes[filter_i])

            self.fluxes[filter_i] = np.array(list(self.fluxes[filter_i]))
        
            # Convert [W m^-2 um^-1] -> [erg s^-1 cm^-2 nm^-1]
            self.fluxes[filter_i] = self.fluxes[filter_i] * 1e7 / (1e2)**2 / 1e3

        return self.fluxes

    def get_transm_curves(self):

        self.transm_curves = {}
        if os.path.exists('./transm_curves.pk'):
            # Read the filter information
            with open('./transm_curves.pk', 'rb') as f:
                self.transm_curves = pickle.load(f)
            
            # Retrieve the filter if not downloaded before
            filters_to_download = []
            for filter_i in self.filters:
                if filter_i not in list(self.transm_curves.keys()):
                    filters_to_download.append(filter_i)
        else:
            filters_to_download = self.filters

        if len(filters_to_download) > 0:
            import urllib.request

            # Base url name
            url_prefix = 'http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id='

            for filter_i in self.filters:
                # Add filter name to the url prefix
                url = url_prefix + filter_i

                # Download the transmission curve
                urllib.request.urlretrieve(url, 'transm_curve.dat')

                # Read the transmission curve
                transmission = np.genfromtxt('transm_curve.dat')
                # Convert the wavelengths [A -> nm]
                transmission[:,0] /= 10

                if transmission.size == 0:
                    raise ValueError('The filter data of {} could not be downloaded'.format(filter_i))
                
                # Store the transmission curve in the dictionary
                self.transm_curves.setdefault(filter_i, transmission)
                
                # Remove the temporary file
                os.remove('transm_curve.dat')

            # Save the requested transmission curves
            with open('transm_curves.pk', 'wb') as outp:
                pickle.dump(self.transm_curves, outp, pickle.HIGHEST_PROTOCOL)
