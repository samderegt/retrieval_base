import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter

import pickle
import os

from PyAstronomy import pyasl
import petitRADTRANS.nat_cst as nc
from petitRADTRANS.retrieval import rebin_give_width as rgw

class Spectrum:

    # The wavelength ranges of each detector and order
    order_wlen_ranges = np.array([
        [1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128],
        [1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165],
        [2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392],
        [2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386],
        [2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835],
        [2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534],
        [2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388],
        ])
    n_orders = 7
    n_dets   = 3
    n_pixels = 2048

    def __init__(self, wave, flux, err=None):

        self.wave = wave
        self.flux = flux
        self.err  = err

        # Make the isfinite mask
        self.update_isfinite_mask()

    def update_isfinite_mask(self):
        self.mask_isfinite = np.isfinite(self.flux)
        self.n_data_points = self.mask_isfinite.sum()

    def rv_shift(self, rv, replace_wave=False):

        # Apply a Doppler shift to the model spectrum
        wave_shifted = self.wave * (1 + rv/(nc.c*1e-5))
        if replace_wave:
            self.wave = wave_shifted
        
        return wave_shifted
    
    def high_pass_filter(self, removal_mode='divide', filter_mode='gaussian', sigma=300, replace_flux_err=False):

        # Prepare an array of low-frequency structure
        low_pass_flux = np.ones_like(flux) * np.nan

        for i, (wave_min, wave_max) in enumerate(Spectrum.order_wlen_ranges):

            # Apply high-pass filter to each detector/order separately
            mask_wave = (self.wave >= wave_min - 0.5) & \
                        (self.wave <= wave_max + 0.5)
            mask_det  = (mask_wave & self.mask_isfinite)
    
            if mask_det.any():
                if filter_mode == 'gaussian':
                    # Find low-frequency structure
                    low_pass_flux[mask_det] = gaussian_filter1d(self.flux[mask_det], sigma=sigma)

                else:
                    # TODO: savgol filter
                    pass

        if removal_mode == 'divide':
            # Divide out the low-frequency structure
            high_pass_flux = self.flux / low_pass_flux
            if self.err is not None:
                high_pass_err  = self.err / low_pass_flux

        elif removal_mode == 'subtract':
            # Subtract away the low-frequency structure
            high_pass_flux = self.flux - low_pass_flux
            if self.err is not None:
                # TODO: how to handle errors for this case?
                high_pass_err  = None

        if replace_flux_err:
            self.flux = high_pass_flux
            if self.err is not None:
                self.err  = high_pass_err

        return high_pass_flux, high_pass_err

    def sigma_clip_poly(self, sigma=5, poly_deg=1, replace_flux=False):

        flux_copy = self.flux.copy()

        # Loop over the orders
        for i in range(Spectrum.n_orders):

            # Select only pixels within the order, should be 3*2048
            mask_wave  = (self.wave >= Spectrum.order_wlen_ranges[i*3+0].min() - 0.5) & \
                         (self.wave <= Spectrum.order_wlen_ranges[i*3+2].max() + 0.5)
            mask_order = (mask_wave & self.mask_isfinite)

            if mask_order.any():

                flux_i = flux_copy[mask_clipped]
                
                # Fit an n-th degree polynomial to this order
                p = np.polyfit(self.wave[mask_order], flux_i, 
                               w=1/self.err[mask_order], deg=poly_deg)

                # Polynomial representation of order                
                poly_model = np.poly1d(p)(self.wave[mask_order])

                # Subtract the polynomial approximation
                residuals = flux_i - poly_model

                # Sigma-clip the residuals
                mask_clipped = (np.abs(residuals) > sigma*np.std(residuals))

                # Set clipped values to NaNs
                flux_i[mask_clipped]  = np.nan
                flux_copy[mask_order] = flux_i

        # TODO: figure

        if replace_flux:
            self.flux = flux_copy

            # Update the isfinite mask
            self.update_isfinite_mask()

        return flux_copy
    

class DataSpectrum(Spectrum):

    def __init__(self, wave, flux, err, ra, dec, mjd, pwv):

        super().__init__(wave, flux, err)

        # Reshape the orders and detectors
        self.reshape_orders_dets()

        self.ra, self.dec, self.mjd, self.pwv = ra, dec, mjd, pwv

    def bary_corr(self):

        # Barycentric velocity (using Paranal coordinates)
        self.v_bary, _ = pyasl.helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, 
                                       ra2000=self.ra, dec2000=self.dec, 
                                       jd=self.mjd+2400000.5
                                       )
        print('Barycentric velocity: {:.2f} km/s'.format(self.v_bary))

        # Apply barycentric correction
        self.rv_shift(self.v_bary, replace_wave=True)

    def reshape_orders_dets(self):

        # Ordered arrays of shape (n_orders, n_dets, n_pixels)
        wave_ordered = np.ones((Spectrum.n_orders, Spectrum.n_dets, Spectrum.n_pixels)) * np.nan
        flux_ordered = np.ones((Spectrum.n_orders, Spectrum.n_dets, Spectrum.n_pixels)) * np.nan
        err_ordered  = np.ones((Spectrum.n_orders, Spectrum.n_dets, Spectrum.n_pixels)) * np.nan

        # Loop over the orders and detectors
        for i in range(Spectrum.n_orders):
            for j in range(Spectrum.n_dets):

                # Select only pixels within the detector, should be 2048
                mask_wave = (self.wave >= Spectrum.order_wlen_ranges[i*3+j].min() - 0.5) & \
                            (self.wave <= Spectrum.order_wlen_ranges[i*3+j].max() + 0.5)

                if mask_wave.any():
                    wave_ordered[i,j] = self.wave[mask_wave]
                    flux_ordered[i,j] = self.flux[mask_wave]
                    err_ordered[i,j]  = self.err[mask_wave]

        self.wave = wave_ordered
        self.flux = flux_ordered
        self.err  = err_ordered

        # Remove empty orders / detectors
        self.clear_empty_orders_dets()

        # Update the isfinite mask
        self.update_isfinite_mask()

    def clear_empty_orders_dets(self):

        # If all pixels are NaNs within an order/detector...
        mask_empty = (~np.isfinite(self.flux)).all(axis=-1)
        
        # ... remove that order/detector
        self.wave = self.wave[mask_empty,:]
        self.flux = self.flux[mask_empty,:]
        self.err  = self.err[mask_empty,:]

    def get_delta_wave(self):

        '''
        self.delta_wave = np.ones((self.n_orders, self.n_dets, 
                                   self.n_data_points, 
                                   self.n_data_points)) * np.nan
        '''
        # Wavelength separation between pixels (within an order/detector)
        self.delta_wave = self.wave[:,:,:,None] - self.wave[:,:,None,:]

    def clip_det_edges(self, n_edge_pixels=50):
        
        # Loop over the orders and detectors
        for i, (wave_min, wave_max) in enumerate(Spectrum.order_wlen_ranges):
            
            mask_wave = (self.wave >= wave_min - 0.5) & \
                        (self.wave <= wave_max + 0.5)
            mask_det  = (mask_wave & self.mask_isfinite)

            if mask_det.any():

                flux_i = self.flux[mask_det]

                # Set the first and last N pixels of each detector to NaN
                flux_i[:n_edge_pixels]  = np.nan
                flux_i[-n_edge_pixels:] = np.nan

                # Set clipped values to NaNs
                self.flux[mask_det] = flux_i

        # Update the isfinite mask
        self.update_isfinite_mask()

    def get_transmission(self, T=10000, ref_rv=0, mode='bb'):

        if mode == 'bb':

            # Retrieve a Planck spectrum for the given temperature
            ref_flux = nc.b(T, nu=(nc.c*1e7)/self.wave.flatten())

            # Convert [erg s^-1 cm^-2 Hz^-1 sr^-1] -> [erg s^-1 cm^-2 Hz^-1]
            ref_flux *= 4*np.pi

            # Convert [erg s^-1 cm^-2 Hz^-1] -> [erg s^-1 cm^-2 nm^-1]
            ref_flux *= (nc.c*1e7)/self.wave.flatten()**2

            # Mask the standard star's hydrogen lines
            ref_flux[(self.wave.flatten()>2166-7) & (self.wave.flatten()<2166+7)] = np.nan
            ref_flux[(self.wave.flatten()>1944-5) & (self.wave.flatten()<1944+5)] = np.nan

        else:

            # TODO: PHOENIX spectrum?
            pass

        # Retrieve and normalize the transmissivity
        self.transm = self.flux / ref_flux
        self.transm /= np.nanmax(self.transm)

        self.transm_err = self.err / ref_flux 
        self.transm_err /= np.nanmax(self.flux / ref_flux)

    def flux_calib_2MASS(self, transm, transm_err, skycalc_transm, photom_2MASS, filter_2MASS, tell_threshold=0.2):

        # Retrieve an approximate telluric transmission spectrum
        wave_skycalc, transm_skycalc = run_skycalc(ra=self.ra, dec=self.dec, mjd=self.mjd, pwv=self.pwv)
        # Interpolate onto the data wavelength grid
        transm_skycalc = np.interp(self.wave, xp=wave_skycalc, fp=transm_skycalc)

        # Linear fit to the continuum, where telluric absorption is minimal
        mask_high_transm = (transm_skycalc > 0.98)
        p = np.polyfit(self.wave[mask_high_transm & self.mask_isfinite].flatten(), 
                       transm[mask_high_transm & self.mask_isfinite].flatten(), 
                       deg=1
                       )
        poly_model = np.poly1d(p)(self.wave)

        # Apply correction for telluric transmission
        tell_corr_flux = self.flux / transm
        # Replace the deepest tellurics with NaNs
        tell_corr_flux[(transm/poly_model / np.nanmax(transm/poly_model)) < tell_threshold] = np.nan

        tell_corr_err = np.sqrt((self.err/transm)**2 + \
                                (tell_corr_flux*transm_err/transm)**2
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

        # TODO: figure
        
        if replace_flux_err:
            self.flux = calib_flux
            self.err  = calib_err

        return calib_flux, calib_err


class ModelSpectrum(Spectrum):

    def __init__(self, wave, flux):

        super().__init__(wave, flux)

    def rot_broadening(self, vsini, epsilon_limb=0, replace_flux=False):

        # Evenly space the wavelength grid
        wave_even = np.linspace(self.wave.min(), self.wave.max(), 
                                self.n_data_points
                                )
        flux_even = np.interp(wave_even, xp=self.wave, fp=self.flux)

        # Rotational broadening of the model spectrum
        flux_rot_broad = pyasl.fastRotBroad(wave_even, flux_even, 
                                            epsilon=epsilon_limb, 
                                            vsini=vsini
                                            )
        if replace_flux:
            self.flux = flux_rot_broad
        
        return flux_rot_broad

    def instr_broadening(self, out_res=1e6, in_res=1e6, replace_flux=False):

        # Delta lambda of resolution element is FWHM of the LSF's standard deviation
        sigma_LSF = np.sqrt(1/out_res**2 - 1/in_res**2) / \
                    (2*np.sqrt(2*np.log(2)))

        spacing = np.mean(2*np.diff(self.wave) / \
                          (self.wave[1:] + self.wave[:-1])
                          )

        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF / spacing
        
        # Apply gaussian filter to broaden with the spectral resolution
        flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter, 
                                   mode='nearest'
                                   )

        if replace_flux:
            self.flux = flux_LSF
        
        return flux_LSF

    def rebin(self, new_wave, new_wave_bins, replace_flux=False):

        # model_flux_rebinned = np.interp(data_wave, model_wave_even, 
        #                                 model_flux_instr_broad)

        # Interpolate onto the observed spectrum's wavelength grid
        flux_rebinned = rgw.rebin_give_width(self.wave, self.flux, 
                                             new_wave, new_wave_bins
                                             )

        if replace_flux:
            self.flux = flux_rebinned
        
        return flux_rebinned

    def shift_broaden_rebin(self, new_wave, new_wave_bins, 
                            rv, vsini, epsilon_limb=0, 
                            out_res=1e6, in_res=1e6
                            ):

        # Apply Doppler shift, rotational/instrumental broadening, 
        # and rebin onto a new wavelength grid
        self.rv_shift(rv, replace_wave=True)
        self.rot_broadening(vsini, epsilon_limb, replace_flux=True)
        self.instr_broadening(out_res, in_res, replace_flux=True)
        self.rebin(new_wave, new_wave_bins, replace_flux=True)

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

        filters_to_download = []

        self.transm_curves = {}
        if os.path.exists('./transm_curves.pk'):
            # Read the filter information
            with open('./transm_curves.pk', 'rb') as f:
                self.transm_curves = pickle.load(f)
            
            # Retrieve the filter if not downloaded before
            for filter_i in self.filters:
                if filter_i not in list(self.transm_curves.keys()):
                    filters_to_download.append(filter_i)

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
