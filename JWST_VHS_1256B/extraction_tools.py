import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy.io import fits
from astropy.modeling import models, fitting

from photutils.aperture import EllipticalAperture
from photutils.aperture import aperture_photometry

from jwst.residual_fringe.utils import fit_residual_fringes_1d

import os
os.environ['STPSF_PATH'] = '/net/schenk/data2/regt/JWST_reductions/stpsf-data'

class ApertureCorrection:

    def __init__(self, wave, N=5, disperser=None, filter=None, band=None):
    
        import stpsf

        if None not in [disperser, filter]:
            instr = stpsf.NIRSpec()
            instr.mode = 'IFU'
            instr.disperser = disperser
            instr.filter = filter
        elif band is not None:
            instr = stpsf.MIRI()
            instr.mode = 'IFU'
            instr.band = band
        else:
            raise ValueError('Provide either disperser and filter or band')

        self.instr = instr

        # Reduce the wavelength resolution
        self.wave        = wave
        self.wave_binned = np.linspace(wave.min(), wave.max(), N)

        # Calculate the PSF cube upon initialization
        self._get_psf()

    def _get_psf(self):
        
        import astropy.units as u

        # Calculate the PSF cube
        self.psf      = self.instr.calc_datacube(self.wave_binned*1e-6*u.meter)
        self.psf_cube = self.psf['DET_DIST'].data

    def get_correction_factor(self, aper_kwargs):

        # Extract the intensity within the aperture
        xcen, ycen = (self.psf_cube.shape[1]-1)/2, (self.psf_cube.shape[2]-1)/2
        aper_kwargs['positions'] = (xcen, ycen)

        self.flux_binned_in_aper, _ = extract_per_channel(self.psf_cube, None, **aper_kwargs)
        
        # Interpolate to the original wavelength grid
        self.flux_in_aper = np.interp(self.wave, self.wave_binned, self.flux_binned_in_aper)
        self.correction_factor = 1 / self.flux_in_aper

        return self.correction_factor
    
class SpectralExtraction:

    def __init__(self, file_s3d, file_wave, AC=None, **AC_kwargs):

        self.file_s3d = file_s3d
        self.file_wave = file_wave
        self._load_spectral_cube()

        # Create an ApertureCorrection instance
        if AC is None:
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

    def correct_horizontal_stripes(self, *xy_to_mask, radius_inflation=5, plot=True):
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
        horizontal_collapsed = np.nanmedian(masked_cube, axis=2, keepdims=True)
        horizontal_collapsed = np.nan_to_num(horizontal_collapsed, nan=0.0)

        cube_corrected = np.copy(self.cube) - horizontal_collapsed

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

    def extract_1d(self, radius_inflation=5, plot=True):

        # Fit a single Gaussian to the median image
        self.model_to_extract.y_stddev.tied = lambda m: m.x_stddev
        self.model_to_extract, _ = self._fit_gaussians(
            self.cube_corrected, None, model=self.model_to_extract
        )

        # Extract the 1D spectrum using aperture photometry per channel
        aper_kwargs = dict(
            positions=(self.model_to_extract.x_mean.value, self.model_to_extract.y_mean.value), 
            theta=self.model_to_extract.theta.value, 
            a=self.model_to_extract.x_stddev.value*radius_inflation, 
            b=self.model_to_extract.y_stddev.value*radius_inflation
        )
        self.flux, self.flux_err = extract_per_channel(
            self.cube_corrected, self.cube_err, **aper_kwargs
        )

        # Correct for the missing flux due to the aperture size
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
            ax[0].set(xlim=xlim, ylim=ylim, ylabel='Flux [Jy]', xticklabels=[])

            ylim = (np.nanpercentile(self.flux/self.flux_err, 1)*1/1.2, np.nanpercentile(self.flux/self.flux_err, 99)*1.2)
            ax[1].plot(self.wave, self.flux/self.flux_err, 'k', lw=1)
            ax[1].set(xlim=xlim, ylim=ylim, ylabel='S/N', xlabel='Wavelength [um]')

            ax[2].set_visible(False)

            ax[-1].plot(self.wave, self.correction_factor, 'k', lw=1)
            ax[-1].set(xlim=xlim, ylim=(1.0, 1.4), ylabel='Aperture Correction', xlabel='Wavelength [um]')
            plt.show()

def extract_per_channel(cube, cube_err=None, **aper_kwargs):
    """Extract the flux per channel from a spectral cube using aperture photometry."""
    
    from photutils.aperture import EllipticalAperture, aperture_photometry

    flux = np.zeros(cube.shape[0])
    flux_err = np.zeros(cube.shape[0])

    if cube_err is None:
        cube_err = [None] * cube.shape[0]

    # Create an aperture
    aperture = EllipticalAperture(**aper_kwargs)

    # Loop over all spectral channels and extract the flux
    for i, (image, image_err) in enumerate(zip(cube, cube_err)):
        phot_table = aperture_photometry(image, aperture, error=image_err)

        flux[i] = phot_table['aperture_sum'].data[0]
        if image_err is not None:
            flux_err[i] = phot_table['aperture_sum_err'].data[0]

    return flux, flux_err

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
            
            ax[0].plot(wave[0], flux_i, lw=0.8)
            ax[1].plot(wave[0], flux_i/flux_err_i, lw=0.8)

            ax[0].plot(wave[0][is_clipped_i], flux_i[is_clipped_i], 'rx')

        ax[0].plot(wave[0], flux_mean, lw=1.2, c='k')
        ax[1].plot(wave[0], flux_mean/flux_err_mean, lw=1.2, c='k')

        if xlim is None:
            xlim = (wave[0][np.isfinite(flux_mean)][0]-0.02, wave[0][np.isfinite(flux_mean)][-1]+0.02)

        ylim = (np.nanpercentile(flux, 1)*1/1.2, np.nanpercentile(flux_mean, 99)*1.2)
        ax[0].set(xlim=xlim, ylim=ylim, ylabel='Flux [Jy]')

        ylim = (np.nanpercentile(flux/flux_err, 1)*1/1.2, np.nanpercentile(flux_mean/flux_err_mean, 99)*1.2)
        ax[1].set(ylim=ylim, ylabel='S/N', xlabel='Wavelength [micron]')
        plt.show()

