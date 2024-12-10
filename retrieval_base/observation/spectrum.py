import numpy as np
import scipy.constants as sc

from .. import utils

class Spectrum:

    def rv_shift(self, rv, replace=False):
        # Apply a Doppler shift to the wavelength array
        wave_shifted = self.wave * (1 + rv/(sc.c*1e-3))
        if replace:
            self.wave = wave_shifted

        return wave_shifted

    def barycentric_correction(self, ra, dec, mjd, observatory_coords=None):
        
        if observatory_coords is None:
            raise ValueError('Observatory coordinates must be provided')
        
        # Set the coordinates and MJD
        self.ra, self.dec, self.mjd = ra, dec, mjd

        # Calculate barycentric velocity correction
        from PyAstronomy import pyasl
        self.v_bary, _ = pyasl.helcorr(
            ra2000=self.ra, dec2000=self.dec, jd=self.mjd+2400000.5, 
            **observatory_coords
            )

        # Apply barycentric correction to the wavelength array
        self.rv_shift(self.v_bary, replace=True)

    def reshape_spectrum(self):
        # Reshape the spectrum to a 2D array
        self.wave = self.wave.reshape(self.n_chips, self.n_pixels)
        self.flux = self.flux.reshape(self.n_chips, self.n_pixels)
        self.err  = self.err.reshape(self.n_chips, self.n_pixels)
    
    def crop_spectrum(self, wave_range):
        
        # Mask the spectrum outside the wavelength range
        mask_wave = (self.wave > wave_range[0]) & (self.wave < wave_range[1])
        self.flux[~mask_wave] *= np.nan
        self.err[~mask_wave]  *= np.nan

    def remove_empty_chips(self, remove_empty_pixels=False):
        # Remove any empty chips
        self.non_empty_chips = ~np.isnan(self.flux).all(axis=1)
        self.wave = self.wave[self.non_empty_chips]
        self.flux = self.flux[self.non_empty_chips]
        self.err  = self.err[self.non_empty_chips]
        if not remove_empty_pixels:
            return
        
        # Remove any empty pixels
        self.non_empty_pixels = ~np.isnan(self.flux).all(axis=0)
        self.wave = self.wave[:,self.non_empty_pixels]
        self.flux = self.flux[:,self.non_empty_pixels]
        self.err  = self.err[:,self.non_empty_pixels]

    def mask_wavelength_ranges(self, wave_ranges=None, pad=0.):
        
        if wave_ranges is None:
            return
        
        if not isinstance(wave_ranges[0], (list, tuple, np.ndarray)):
            # If only one range is provided, convert to list
            wave_ranges = [wave_ranges]
        
        for (wave_min, wave_max) in wave_ranges:
            # Mask the spectrum inside the wavelength range
            mask_wave = (self.wave >= wave_min-pad) & (self.wave <= wave_max+pad)
            self.flux[mask_wave] *= np.nan
            self.err[mask_wave]  *= np.nan

    def save_to_pickle(self, data_dir):
        # Pickle save the object
        utils.save_pickle(self, data_dir / self.pickle_filename)
        
        # Save wave, flux, and err to a npz file
        np.savez(data_dir / (self.pickle_filename.stem + '_wave_flux_err.npz'), 
             wave=self.wave, flux=self.flux, err=self.err)