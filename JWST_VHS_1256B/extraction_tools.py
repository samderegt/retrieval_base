import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy.io import fits
from photutils.aperture import EllipticalAperture, aperture_photometry

# from jwst.residual_fringe.utils import fit_residual_fringes_1d

import warnings
import os

class NIRSpecExtraction:

    def __init__(self, file_combined, files_dither, grating, detector):

        self.grating = grating
        self.detector = detector

        self.data_combined = {}
        self.data_combined['wave'], self.data_combined['cube'], self.data_combined['cube_err'] = self._load_data(file_combined)

        self.data_dither = [{} for _ in range(len(files_dither))]
        for i, file_dither in enumerate(files_dither):
            self.data_dither[i]['wave'], self.data_dither[i]['cube'], self.data_dither[i]['cube_err'] = self._load_data(file_dither)

    @classmethod
    def _load_data(cls, filename):

        # Get the wavelength array
        file_x1d = filename.replace('s3d', 'x1d')
        wave = fits.getdata(file_x1d)['WAVELENGTH']

        # 3D spectral cube
        hdu = fits.open(filename)
        cube = hdu['SCI'].data
        cube_err = hdu['ERR'].data

        # Convert flux [MJy/sr] -> [Jy]
        pixel_area = hdu['SCI'].header['PIXAR_SR']
        cube     *= pixel_area*1e6 
        cube_err *= pixel_area*1e6

        # Convert [Jy] to [W/m^2/micron]
        cube, cube_err = convert_Jy_to_F_lam(wave[:,None,None], cube, cube_err)

        return wave, cube, cube_err
    
    def _get_gaussian_model(cls, im, xy_guess):
        
        from astropy.modeling import models

        y_pix, x_pix = np.mgrid[:im.shape[0], :im.shape[1]]

        # Search a 5x5 box around xy_guess for a better initial guess
        x_min = max(0, int(xy_guess[0])-2); x_max = min(im.shape[1], int(xy_guess[0])+3)
        y_min = max(0, int(xy_guess[1])-2); y_max = min(im.shape[0], int(xy_guess[1])+3)
        sub_image = np.abs(im[y_min:y_max, x_min:x_max])
        local_max_idx = np.unravel_index(np.argmax(sub_image), sub_image.shape)

        xy_guess = (x_min + local_max_idx[1], y_min + local_max_idx[0])
        amp_guess = im[int(xy_guess[1]), int(xy_guess[0])]

        # Create a Gaussian model
        gaussian = models.Gaussian2D(
            amplitude=amp_guess, x_mean=xy_guess[0], y_mean=xy_guess[1], x_stddev=1., y_stddev=1., theta=0.
        )
        # Set the bounds for the parameters
        gaussian.x_stddev.bounds = (0.3, 10)
        gaussian.y_stddev.bounds = (0.3, 10)
        gaussian.y_stddev.tied = lambda m: m.x_stddev

        gaussian.x_mean.bounds = (0, im.shape[1]-1)
        gaussian.y_mean.bounds = (0, im.shape[0]-1)
        gaussian.theta.fixed = True

        return gaussian, y_pix, x_pix

    def _extract_per_channel(cls, cube, cube_err=None, **aper_kwargs):
        """Extract the flux per channel from a spectral cube using aperture photometry."""
        
        flux = np.zeros(cube.shape[0])
        flux_err = np.zeros(cube.shape[0])

        if cube_err is None:
            cube_err = [None] * cube.shape[0]

        # Loop over all spectral channels and extract the flux
        for i, (image, image_err) in enumerate(zip(cube, cube_err)):
            # Create an aperture
            aperture = EllipticalAperture(**aper_kwargs)
            mask = np.isnan(image)
            if mask.all():
                flux[i] = np.nan; flux_err[i] = np.nan
                continue
            phot_table = aperture_photometry(image, aperture, error=image_err)

            flux[i] = phot_table['aperture_sum'].data[0]
            if image_err is not None:
                flux_err[i] = phot_table['aperture_sum_err'].data[0]

        return flux, flux_err
    
    def fit_aperture_on_combined(self, xy_guess):

        # Median-combine the cube along the wavelength axis
        median_image = np.nanmedian(self.data_combined['cube'], axis=0)
        median_image -= np.nanmedian(median_image) # Remove background
        median_image /= np.nanmax(median_image[4:-4,4:-4]) # Normalize
        median_image = np.nan_to_num(median_image, nan=0.0)

        gaussian, y_pix, x_pix = self._get_gaussian_model(median_image, xy_guess)
        
        from astropy.modeling import fitting
        fitter = fitting.LevMarLSQFitter()
        fit = fitter(gaussian, x_pix, y_pix, median_image)

        self.sigma = fit.x_stddev.value
        self.xy_combined = (fit.x_mean.value, fit.y_mean.value)

    def fit_aperture_on_dithers(self, *xy_guesses):

        self.xy_dithers = []
        for i, (data_dither_i, xy_guess_i) in enumerate(zip(self.data_dither, xy_guesses)):

            # Median-combine the cube along the wavelength axis
            median_image = np.nanmedian(data_dither_i['cube'], axis=0)
            median_image -= np.nanmedian(median_image) # Remove background
            median_image /= np.nanmax(median_image[4:-4,4:-4]) # Normalize
            median_image = np.nan_to_num(median_image, nan=0.0)

            gaussian, y_pix, x_pix = self._get_gaussian_model(median_image, xy_guess_i)

            # Fix the sigma to that from the combined image
            gaussian.x_stddev.value = self.sigma
            gaussian.y_stddev.value = self.sigma
            gaussian.x_stddev.fixed = True
            gaussian.y_stddev.fixed = True

            from astropy.modeling import fitting
            fitter = fitting.LevMarLSQFitter()
            fit = fitter(gaussian, x_pix, y_pix, median_image)

            # Ignore fits with low amplitude
            # print(fit.amplitude.value, 0.2*np.nanmax(median_image))
            # if np.abs(fit.amplitude.value) < 0.2*np.nanmax(median_image):
            #     continue

            self.xy_dithers.append((fit.x_mean.value, fit.y_mean.value))

    def correct_dithers(self, radius_inflation=6, plot=True):
        
        for i, data_dither_i in enumerate(self.data_dither):

            # Mask the fitted Gaussians
            mask = np.zeros_like(data_dither_i['cube'][0])
            for xy_j in self.xy_dithers:
                aperture_j = EllipticalAperture(
                    positions=xy_j, 
                    theta=0., 
                    a=self.sigma*radius_inflation, 
                    b=self.sigma*radius_inflation
                    )
                mask += aperture_j.to_mask().to_image(mask.shape)

            mask = (mask!=0.)
            masked_cube = np.copy(data_dither_i['cube'])
            masked_cube[:,mask] = np.nan
            masked_cube[:,:,:2] = np.nan; masked_cube[:,:,-2:] = np.nan
            
            # Remove the horizontal stripes
            horizontal_collapsed = np.nanmedian(masked_cube, axis=2, keepdims=True)
            horizontal_collapsed = np.nan_to_num(horizontal_collapsed, nan=0.0)
            # horizontal_collapsed *= 0.
            data_dither_i['cube_corrected'] = np.copy(data_dither_i['cube']) - horizontal_collapsed
            
            # vertical_collapsed = np.nanmedian(masked_cube, axis=1, keepdims=True)
            # vertical_collapsed = np.nan_to_num(vertical_collapsed, nan=0.0)
            # data_dither_i['cube_corrected'] -= vertical_collapsed

            if plot:
                # Plot the results
                median_image = np.nanmedian(data_dither_i['cube'], axis=0)
                median_image_corrected = np.nanmedian(data_dither_i['cube_corrected'], axis=0)
                
                vmax = np.nanpercentile(median_image, 97)

                fig, ax = plt.subplots(figsize=(7,2.7), ncols=3, gridspec_kw={'width_ratios': [1,1,0.05]})
                ax[0].imshow(median_image, origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax)
                ax[0].set_title('Median image')
                im = ax[1].imshow(median_image_corrected, origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax)
                ax[1].set_title('After correction')

                # Plot the fitted Gaussians
                for ax_i in ax[:2]:
                    for xy_j in self.xy_dithers:
                        ellipse = Ellipse(
                            xy=xy_j,
                            width=2*self.sigma*radius_inflation,
                            height=2*self.sigma*radius_inflation,
                            angle=0.,
                            edgecolor='k', facecolor='none', lw=1.5
                        )
                        ax_i.add_patch(ellipse)
                        ax_i.scatter(
                            xy_j[0], xy_j[1], 
                            marker='+', color='k', s=50, lw=2
                        )

                # Add a colorbar to the right of the plots
                plt.colorbar(im, cax=ax[-1], orientation='vertical')

                plt.show()

    def extract_1d_dithers(self, radius=4):

        for i, data_dither_i in enumerate(self.data_dither):

            aper_kwargs = dict(
                positions=self.xy_dithers[i], theta=0., a=self.sigma*radius, b=self.sigma*radius
            )
            data_dither_i['flux'], data_dither_i['flux_err'] = extract_per_channel(
                data_dither_i['cube_corrected'], data_dither_i['cube_err'], **aper_kwargs
            )

    def combine_dithers(self, sigma_clip=20, plot=True, **axis_kwargs):
        
        wave = np.array([data_dither_i['wave'] for data_dither_i in self.data_dither])
        flux = np.array([data_dither_i['flux'] for data_dither_i in self.data_dither])
        flux_err = np.array([data_dither_i['flux_err'] for data_dither_i in self.data_dither])

        if (wave != wave[[0],:]).any():
            raise ValueError('Wavelength grids do not match!')
        
        weight = 1/flux_err**2

        flux_mean = np.ones_like(wave[0]) * np.nan
        flux_err_mean = np.ones_like(wave[0]) * np.nan
        
        is_in_any_dither = np.zeros_like(wave[0], dtype=bool)
        is_clipped = np.zeros_like(wave, dtype=bool)

        # Loop over all spectral channels
        for i, (flux_i, flux_err_i, weight_i) in enumerate(zip(flux.T, flux_err.T, weight.T)):

            # Only include valid values
            is_in_dither_i = np.isfinite(flux_i) & np.isfinite(flux_err_i)
            if not is_in_dither_i.any():
                # No spectrum has valid values in this channel
                continue
            is_in_any_dither[i] = True

            if sigma_clip is not None:
                # Sigma-clip outliers wrt median spectrum, removing pixels
                # with mistaken similar weights as other spectra/dithers
                median_flux     = np.nanmedian(flux_i[is_in_dither_i])
                median_flux_err = np.nanmedian(flux_err_i[is_in_dither_i])

                is_in_dither_i = is_in_dither_i & (
                    np.abs(flux_i-median_flux) < sigma_clip*median_flux_err
                )

            # Update the clipped values
            is_clipped[~is_in_dither_i,i] = True

            if not is_in_dither_i.any():
                # No valid values left after clipping
                continue

            # Calculate the weighted mean and error
            flux_mean[i] = np.average(flux_i[is_in_dither_i], weights=weight_i[is_in_dither_i])
            flux_err_mean[i] = np.sqrt(
                np.sum((weight_i[is_in_dither_i]*flux_err_i[is_in_dither_i])**2) / (np.sum(weight_i[is_in_dither_i])**2)
            )

        if plot:
            
            fig, ax = plt.subplots(figsize=(10,6), nrows=2, sharex=True, gridspec_kw={'height_ratios':[1,0.3]})
            for flux_i, flux_err_i, is_clipped_i in zip(flux, flux_err, is_clipped):
                
                flux_masked_i = flux_i.copy()
                flux_masked_i[is_clipped_i] = np.nan
                line = ax[0].plot(wave[0], flux_i, lw=0.5, alpha=0.3)
                ax[0].plot(wave[0], flux_masked_i, lw=0.8, c=line[0].get_color())
                ax[1].plot(wave[0], flux_i/flux_err_i, lw=0.8, c=line[0].get_color())

                ax[0].plot(wave[0][is_clipped_i], flux_i[is_clipped_i], 'C6x', zorder=10, ms=5)

            ax[0].plot(wave[0], flux_mean, lw=1.2, c='k')
            ax[1].plot(wave[0], flux_mean/flux_err_mean, lw=1.2, c='k')

            if 'xlim' in axis_kwargs:
                xlim = axis_kwargs.pop('xlim')
            else:
                xlim = (wave[0][np.isfinite(flux_mean)][0]-0.02, wave[0][np.isfinite(flux_mean)][-1]+0.02)
            ax[0].set(xlim=xlim)

            if 'ylim' in axis_kwargs:
                ylim = axis_kwargs.pop('ylim')
            else:
                ylim = (np.nanpercentile(flux, 1)*1/1.2, np.nanpercentile(flux_mean, 99)*1.2)
            ax[0].set(ylim=ylim, ylabel='Flux [W/m^2/micron]', **axis_kwargs)

            ylim = (np.nanpercentile(flux/flux_err, 1)*1/1.2, np.nanpercentile(flux_mean/flux_err_mean, 99)*1.2)
            ax[1].set(ylim=ylim, ylabel='S/N', xlabel='Wavelength [micron]', **axis_kwargs)
            plt.show()

        # Remove pixels that are not valid in any dither
        self.wave = wave[0][is_in_any_dither]
        self.flux = flux_mean[is_in_any_dither]
        self.flux_err = flux_err_mean[is_in_any_dither]

        # Remove first and last pixels
        self.wave = self.wave[1:-1]
        self.flux = self.flux[1:-1]
        self.flux_err = self.flux_err[1:-1]

    def aperture_correction_from_dithers(self, radius_tot=10, plot=True):

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10,2))

        for i, data_dither_i in enumerate(self.data_dither):

            aper_kwargs_tot = dict(
                positions=self.xy_dithers[i], theta=0., a=self.sigma*radius_tot, b=self.sigma*radius_tot
            )
            flux_tot, flux_err_tot = extract_per_channel(
                data_dither_i['cube_corrected'], data_dither_i['cube_err'], **aper_kwargs_tot
            )

            flux_measured, flux_err_measured = data_dither_i['flux'], data_dither_i['flux_err']

            ratio = flux_tot / flux_measured
            ratio_err = np.sqrt((flux_tot/flux_measured**2*flux_err_measured)**2 + (1/flux_measured*flux_err_tot)**2)
            mask = np.isfinite(ratio)
            print(mask.sum(), 'valid points for dither', i)
            print(ratio[mask][-1], data_dither_i['wave'][mask][-1], 'last point for dither', i)
            polynomial = np.polyfit(data_dither_i['wave'][mask], ratio[mask], w=1/ratio_err[mask]**2, deg=1)
            correction_factor = np.polyval(polynomial, data_dither_i['wave'])

            if plot:
                ax.plot(data_dither_i['wave'][mask], ratio[mask], alpha=0.3, c=f'C{i}')
                ax.plot(data_dither_i['wave'][mask], correction_factor[mask], c=f'C{i}', zorder=10)
                # ax.set(xlabel='Wavelength [micron]', ylabel=f'[{radius_tot}*sigma] / [4*sigma]', ylim=(1,1.2), title=f'Aperture correction for dither {i+1}')

            self.data_dither[i]['flux'] *= correction_factor
            self.data_dither[i]['flux_err'] *= correction_factor

        if plot:
            ax.set(ylim=(1,1.2))
            plt.show()
            plt.close()

    def aperture_correction_from_combined(self, radius=4, radius_tot=10, plot=True):
        
        aper_kwargs = dict(
            positions=self.xy_combined, theta=0., a=self.sigma*radius, b=self.sigma*radius
        )
        aper_kwargs_tot = dict(
            positions=self.xy_combined, theta=0., a=self.sigma*radius_tot, b=self.sigma*radius_tot
        )

        flux_measured, flux_err_measured = extract_per_channel(
            self.data_combined['cube'], self.data_combined['cube_err'], **aper_kwargs
        )
        flux_tot, flux_err_tot = extract_per_channel(
            self.data_combined['cube'], self.data_combined['cube_err'], **aper_kwargs_tot
        )

        ratio = flux_tot / flux_measured
        ratio_err = np.sqrt((flux_tot/flux_measured**2*flux_err_measured)**2 + (1/flux_measured*flux_err_tot)**2)
        mask = np.isfinite(ratio)
        polynomial = np.polyfit(self.data_combined['wave'][mask], ratio[mask], w=1/ratio_err[mask]**2, deg=1)
        self.correction_factor = np.polyval(polynomial, self.wave)

        self.flux *= self.correction_factor
        self.flux_err *= self.correction_factor

        self.flux_combined = np.interp(self.wave, self.data_combined['wave'][mask], flux_measured[mask]) * self.correction_factor
        self.flux_err_combined = np.interp(self.wave, self.data_combined['wave'][mask], flux_err_measured[mask]) * self.correction_factor

        if plot: 
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10,2))
            ax.plot(self.data_combined['wave'], ratio)
            ax.plot(self.wave, self.correction_factor)
            ax.set(xlabel='Wavelength [micron]', ylabel=f'[{radius_tot}*sigma] / [{radius}*sigma]', ylim=(1,1.2), title='Aperture correction')
            plt.show()

    def save_spectrum(self, filename):
        
        np.savetxt(
            filename, np.array([self.wave, self.flux, self.flux_err]).T, delimiter=',', 
            header='Wavelength (microns), Flux(W/m^2/microns), Flux Error(W/m^2/microns)', 
            )
        
class SpectralExtraction:

    def __init__(self, file_s3d, file_wave, AC=None, **AC_kwargs):

        self.file_s3d = file_s3d
        self.file_wave = file_wave
        self._load_spectral_cube()

        # Create an ApertureCorrection instance
        if AC is None:
            return
            self.AC = ApertureCorrection(self.wave, **AC_kwargs)
        else:
            self.AC = AC # Adopt from previous extraction

    def _load_spectral_cube(self):
        """Load the spectral cube from the FITS file."""
        self.wave = fits.getdata(self.file_wave)['WAVELENGTH']

        hdu = fits.open(self.file_s3d)
        self.cube = hdu['SCI'].data
        self.cube_err = hdu['ERR'].data

        pixel_area = hdu['SCI'].header['PIXAR_SR']
        self.cube     *= pixel_area*1e6 # Convert to flux [MJy/sr] -> [Jy]
        self.cube_err *= pixel_area*1e6

        # Convert [Jy] to [W/m^2/micron]
        self.cube, self.cube_err = convert_Jy_to_F_lam(
            self.wave[:,None,None], self.cube, self.cube_err
            )

    def _fit_gaussians(self, cube, *xy_to_mask, model=None):

        # Median-combine the cube along the wavelength axis
        median_image = np.nanmedian(cube, axis=0)
        median_image -= np.nanmedian(median_image) # Remove background
        median_image /= np.nanmax(median_image[4:-4,4:-4]) # Normalize
        median_image = np.nan_to_num(median_image, nan=0.0)

        y_pix, x_pix = np.mgrid[:median_image.shape[0], :median_image.shape[1]]

        from astropy.modeling import models, fitting
        fitter = fitting.LevMarLSQFitter()

        idx_to_extract = 0
        if model is not None:
            # Fit the model to the median image
            fit = fitter(model, x_pix, y_pix, median_image)
            return fit, idx_to_extract
        
        # If no model is provided, create a new model based on the provided positions
        for i, xy_i in enumerate(xy_to_mask):
            # Search a 5x5 box around xy_i for a better initial guess
            x_min = max(0, int(xy_i[0])-2); x_max = min(median_image.shape[1], int(xy_i[0])+3)
            y_min = max(0, int(xy_i[1])-2); y_max = min(median_image.shape[0], int(xy_i[1])+3)
            sub_image = np.abs(median_image[y_min:y_max, x_min:x_max])
            local_max_idx = np.unravel_index(np.argmax(sub_image), sub_image.shape)
            xy_i = (x_min + local_max_idx[1], y_min + local_max_idx[0])

            amp_i = median_image[int(xy_i[1]), int(xy_i[0])]
            
            if amp_i > 0.:
                idx_to_extract = i
            
            # Add another Gaussian to the model
            gaussian_i = models.Gaussian2D(
                amplitude=amp_i, x_mean=xy_i[0], y_mean=xy_i[1], x_stddev=1., y_stddev=1., theta=0.
            )

            # Set the bounds for the parameters
            gaussian_i.x_stddev.bounds = (0.5, 6)
            gaussian_i.y_stddev.bounds = (0.5, 6)
            gaussian_i.y_stddev.tied = lambda m: m[i].x_stddev

            gaussian_i.x_mean.bounds = (0, median_image.shape[1]-1)
            gaussian_i.y_mean.bounds = (0, median_image.shape[0]-1)

            gaussian_i.theta.fixed = True
            gaussian_i.amplitude.bounds = (-1.5, 1.5)

            # If this is the first Gaussian, initialize the model
            if i == 0:
                model = gaussian_i
            else:
                model += gaussian_i

        # Fit the model to the median image
        fit = fitter(model, x_pix, y_pix, median_image)
        return fit, idx_to_extract

    def correct_horizontal_background(self, *xy_to_mask, radius_inflation=5, plot=True):
        """Correct horizontal stripes in the spectral cube."""

        # Fit the Gaussians to the provided positions
        fit, idx_to_extract = self._fit_gaussians(self.cube, *xy_to_mask)
        
        # Mask the fitted Gaussians
        mask = np.zeros_like(self.cube[0])
        for i, gaussian_i in enumerate(fit):
            aperture_i = EllipticalAperture(
                positions=(gaussian_i.x_mean.value, gaussian_i.y_mean.value), 
                theta=gaussian_i.theta.value, 
                a=gaussian_i.x_stddev.value*radius_inflation, 
                b=gaussian_i.y_stddev.value*radius_inflation
                )
            mask += aperture_i.to_mask().to_image(mask.shape)

        mask = (mask!=0.)

        masked_cube = np.copy(self.cube)
        masked_cube[:,mask] = np.nan

        # Remove the horizontal stripes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            horizontal_collapsed = np.nanmedian(masked_cube, axis=2, keepdims=True)
        horizontal_collapsed = np.nan_to_num(horizontal_collapsed, nan=0.0)

        # horizontal_collapsed *= 0.
        cube_corrected = np.copy(self.cube) - horizontal_collapsed

        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', category=RuntimeWarning)
        #     vertical_collapsed = np.nanmedian(cube_corrected, axis=1, keepdims=True)
        # vertical_collapsed = np.nan_to_num(vertical_collapsed, nan=0.0)

        # cube_corrected -= vertical_collapsed

        if plot:
            # Plot the results
            median_image = np.nanmedian(self.cube, axis=0)
            median_image_corrected = np.nanmedian(cube_corrected, axis=0)
            
            vmax = np.nanpercentile(median_image, 97)

            fig, ax = plt.subplots(figsize=(7,2.7), ncols=3, gridspec_kw={'width_ratios': [1,1,0.05]})
            ax[0].imshow(median_image, origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax)
            ax[0].set_title('Median image')
            im = ax[1].imshow(median_image_corrected, origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax)
            ax[1].set_title('After correction')

            # Plot the fitted Gaussians
            for ax_i in ax[:2]:
                for i, gaussian_i in enumerate(fit):
                    ec = 'C1' if i==idx_to_extract else 'k'
                    ellipse = Ellipse(
                        xy=(gaussian_i.x_mean.value, gaussian_i.y_mean.value),
                        width=2*gaussian_i.x_stddev.value*radius_inflation,
                        height=2*gaussian_i.y_stddev.value*radius_inflation,
                        angle=np.rad2deg(gaussian_i.theta.value),
                        edgecolor=ec, facecolor='none', lw=1.5
                    )
                    ax_i.add_patch(ellipse)
                    ax_i.scatter(
                        gaussian_i.x_mean.value, gaussian_i.y_mean.value, 
                        marker='+', color=ec, s=50, lw=2
                    )

            # Add a colorbar to the right of the plots
            plt.colorbar(im, cax=ax[-1], orientation='vertical')

            plt.show()
        
        self.model_to_extract = fit[idx_to_extract]
        self.cube_corrected = cube_corrected

    def plot_radius_inflation(self, radii=np.arange(1,10,1)):
        # Fit a single Gaussian to the median image
        self.model_to_extract.y_stddev.tied = lambda m: m.x_stddev
        self.model_to_extract, _ = self._fit_gaussians(
            self.cube_corrected, None, model=self.model_to_extract
        )

        flux_per_inflation = np.zeros_like(radii, dtype=float)
        for i, radius_inflation_i in enumerate(radii):
            aper_kwargs = dict(
                positions=(self.model_to_extract.x_mean.value, self.model_to_extract.y_mean.value), 
                theta=self.model_to_extract.theta.value, 
                a=self.model_to_extract.x_stddev.value*radius_inflation_i, 
                b=self.model_to_extract.y_stddev.value*radius_inflation_i
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                median_image = np.nanmedian(self.cube_corrected, axis=0, keepdims=True)
                median_image[np.isnan(median_image)] = 0.
            flux_per_inflation[i], _ = extract_per_channel(median_image, **aper_kwargs)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,2.5))
        ax.plot(self.model_to_extract.x_stddev.value*radii, flux_per_inflation, 'o')
        ax.set(xlabel='Aperture [pixels]', ylabel='Flux', xlim=(0,(radii.max()+1)*self.model_to_extract.x_stddev.value))
        plt.show()

    def extract_1d(self, radius_inflation=5, plot=True):

        # Fit a single Gaussian to the median image
        self.model_to_extract.y_stddev.tied = lambda m: m.x_stddev
        self.model_to_extract, _ = self._fit_gaussians(
            self.cube_corrected, None, model=self.model_to_extract
        )

        median_wave = np.nanmedian(
            self.wave[~np.all(np.isnan(self.cube_corrected), axis=(1,2))]
        )

        # Extract the 1D spectrum using aperture photometry per channel
        aper_kwargs = dict(
            positions=(self.model_to_extract.x_mean.value, self.model_to_extract.y_mean.value), 
            theta=self.model_to_extract.theta.value, 
            a=self.model_to_extract.x_stddev.value*radius_inflation, 
            b=self.model_to_extract.y_stddev.value*radius_inflation
        )
        self.flux, self.flux_err = extract_per_channel(
            self.cube_corrected, self.cube_err, 
            positions=aper_kwargs['positions'], theta=aper_kwargs['theta'], 
            a=aper_kwargs['a'], 
            b=aper_kwargs['b'],
        )

        # Correct for the missing flux due to the aperture size
        self.correction_factor = np.ones_like(self.flux)
        if hasattr(self, 'AC'):
            self.correction_factor = self.AC.get_correction_factor(aper_kwargs)
            
        self.flux     *= self.correction_factor
        self.flux_err *= self.correction_factor

        if plot:
            mask = np.isfinite(self.flux)
            xlim = (self.wave[mask][0]-0.02, self.wave[mask][-1]+0.02)
            ylim = (np.nanpercentile(self.flux, 1)*1/1.2, np.nanpercentile(self.flux, 99)*1.2)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7,8), nrows=4, gridspec_kw={'height_ratios': [1,1,0.15,0.6], 'hspace':0.05})
            ax[0].plot(self.wave, self.flux, 'k', lw=1)
            ax[0].set(xlim=xlim, ylim=ylim, ylabel='Flux [W/m^2/micron]', xticklabels=[])

            ylim = (np.nanpercentile(self.flux/self.flux_err, 1)*1/1.2, np.nanpercentile(self.flux/self.flux_err, 99)*1.2)
            ax[1].plot(self.wave, self.flux/self.flux_err, 'k', lw=1)
            ax[1].set(xlim=xlim, ylim=ylim, ylabel='S/N', xlabel='Wavelength [um]')

            ax[2].set_visible(False)

            ax[-1].plot(self.wave, self.correction_factor, 'k', lw=1)
            ax[-1].set(xlim=xlim, ylim=(1.0, 1.4), ylabel='Aperture Correction', xlabel='Wavelength [um]')
            plt.show()

def extract_per_channel(cube, cube_err=None, **aper_kwargs):
    """Extract the flux per channel from a spectral cube using aperture photometry."""
    
    flux = np.zeros(cube.shape[0])
    flux_err = np.zeros(cube.shape[0])

    if cube_err is None:
        cube_err = [None] * cube.shape[0]

    a = np.atleast_1d(aper_kwargs['a']) * np.ones_like(flux)
    b = np.atleast_1d(aper_kwargs['b']) * np.ones_like(flux)

    # Loop over all spectral channels and extract the flux
    for i, (image, image_err) in enumerate(zip(cube, cube_err)):
        # Create an aperture
        aperture = EllipticalAperture(
            positions=aper_kwargs['positions'], theta=aper_kwargs['theta'], a=a[i], b=b[i]
        )
        mask = np.isnan(image)
        if mask.all():
            flux[i] = np.nan; flux_err[i] = np.nan
            continue
        phot_table = aperture_photometry(image, aperture, error=image_err)

        flux[i] = phot_table['aperture_sum'].data[0]
        if image_err is not None:
            flux_err[i] = phot_table['aperture_sum_err'].data[0]

    return flux, flux_err

"""
def combine_extractions(*SEs, sigma_clip=20, plot=True, xlim=None):

    wave = np.array([SE.wave for SE in SEs])
    flux = np.array([SE.flux for SE in SEs])
    flux_err = np.array([SE.flux_err for SE in SEs])

    if (wave != wave[[0],:]).any():
        raise ValueError('Wavelength grids do not match!')
    
    weight = 1/flux_err**2

    flux_mean = np.ones_like(wave[0]) * np.nan
    flux_err_mean = np.ones_like(wave[0]) * np.nan
    
    is_in_any_dither = np.zeros_like(wave[0], dtype=bool)
    is_clipped = np.zeros_like(wave, dtype=bool)

    # Loop over all spectral channels
    for i, (flux_i, flux_err_i, weight_i) in enumerate(zip(flux.T, flux_err.T, weight.T)):

        # Only include valid values
        is_in_dither_i = np.isfinite(flux_i) & np.isfinite(flux_err_i)
        if not is_in_dither_i.any():
            # No spectrum has valid values in this channel
            continue
        is_in_any_dither[i] = True

        if sigma_clip is not None:
            # Sigma-clip outliers wrt median spectrum, removing pixels
            # with mistaken similar weights as other spectra/dithers
            median_flux     = np.nanmedian(flux_i[is_in_dither_i])
            median_flux_err = np.nanmedian(flux_err_i[is_in_dither_i])

            is_in_dither_i = is_in_dither_i & (
                np.abs(flux_i-median_flux) < sigma_clip*median_flux_err
            )

        # Update the clipped values
        is_clipped[~is_in_dither_i,i] = True

        if not is_in_dither_i.any():
            # No valid values left after clipping
            continue

        # Calculate the weighted mean and error
        flux_mean[i] = np.average(flux_i[is_in_dither_i], weights=weight_i[is_in_dither_i])
        flux_err_mean[i] = np.sqrt(
            np.sum((weight_i[is_in_dither_i]*flux_err_i[is_in_dither_i])**2) / (np.sum(weight_i[is_in_dither_i])**2)
        )

    if plot:

        fig, ax = plt.subplots(figsize=(10,6), nrows=2, sharex=True)
        for flux_i, flux_err_i, is_clipped_i in zip(flux, flux_err, is_clipped):
            
            flux_masked_i = flux_i.copy()
            flux_masked_i[is_clipped_i] = np.nan
            line = ax[0].plot(wave[0], flux_i, lw=0.5, alpha=0.3)
            ax[0].plot(wave[0], flux_masked_i, lw=0.8, c=line[0].get_color())
            ax[1].plot(wave[0], flux_i/flux_err_i, lw=0.8, c=line[0].get_color())

            ax[0].plot(wave[0][is_clipped_i], flux_i[is_clipped_i], 'C6x', zorder=10, ms=5)

        ax[0].plot(wave[0], flux_mean, lw=1.2, c='k')
        ax[1].plot(wave[0], flux_mean/flux_err_mean, lw=1.2, c='k')

        if xlim is None:
            xlim = (wave[0][np.isfinite(flux_mean)][0]-0.02, wave[0][np.isfinite(flux_mean)][-1]+0.02)

        ylim = (np.nanpercentile(flux, 1)*1/1.2, np.nanpercentile(flux_mean, 99)*1.2)
        ax[0].set(xlim=xlim, ylim=ylim, ylabel='Flux [W/m^2/micron]')

        ylim = (np.nanpercentile(flux/flux_err, 1)*1/1.2, np.nanpercentile(flux_mean/flux_err_mean, 99)*1.2)
        ax[1].set(ylim=ylim, ylabel='S/N', xlabel='Wavelength [micron]')
        plt.show()

    # Remove pixels that are not valid in any dither
    wave = wave[0][is_in_any_dither]
    flux_mean = flux_mean[is_in_any_dither]
    flux_err_mean = flux_err_mean[is_in_any_dither]

    # Remove first and last pixels
    wave = wave[1:-1]
    flux_mean = flux_mean[1:-1]
    flux_err_mean = flux_err_mean[1:-1]
    
    return wave, flux_mean, flux_err_mean
"""
def convert_Jy_to_F_lam(wave, flux, flux_err):

    from scipy.constants import c

    wave_m = 1e-6 * wave # [micron] -> [m]

    flux = 1e-26 * flux # [Jy] -> [W/m^2/Hz]
    flux = flux * c / wave_m**2 # [W/m^2/Hz] -> [W/m^2/m]
    flux = flux * 1e-6 # [W/m^2/m] -> [W/m^2/micron]

    flux_err = 1e-26 * flux_err # [Jy] -> [W/m^2/Hz]
    flux_err = flux_err * c / wave_m**2 # [W/m^2/Hz] -> [W/m^2/m]
    flux_err = flux_err * 1e-6 # [W/m^2/m] -> [W/m^2/micron]
    return flux, flux_err