import numpy as np
import scipy.constants as sc

from .. import utils

class Spectrum:

    def rv_shift(self, rv, replace=False):
        """
        Apply a Doppler shift to the wavelength array.

        Args:
            rv (float): The radial velocity shift in km/s.
            replace (bool): If True, replace the original wavelength array with the shifted one.

        Returns:
            numpy.ndarray: The Doppler-shifted wavelength array.
        """
        # Apply a Doppler shift to the wavelength array
        wave_shifted = self.wave * (1 + rv/(sc.c*1e-3))
        if replace:
            self.wave = wave_shifted

        return wave_shifted

    def barycentric_correction(self, ra, dec, mjd, observatory_coords=None):
        """
        Apply barycentric correction to the wavelength array.

        Args:
            ra (float): Right ascension of the target in degrees.
            dec (float): Declination of the target in degrees.
            mjd (float): Modified Julian Date of the observation.
            observatory_coords (dict): Dictionary containing observatory coordinates.

        Raises:
            ValueError: If observatory coordinates are not provided.
        """
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
        """Reshape the spectrum to a 2D array."""
        self.wave = self.wave.reshape(self.n_chips, self.n_pixels)
        self.flux = self.flux.reshape(self.n_chips, self.n_pixels)
        self.err  = self.err.reshape(self.n_chips, self.n_pixels)
    
    def crop_spectrum(self, wave_range):
        """
        Crop the spectrum to a specified wavelength range.

        Args:
            wave_range (tuple): The wavelength range (min, max) to keep.
        """
        # Mask the spectrum outside the wavelength range
        mask_wave = (self.wave > wave_range[0]) & (self.wave < wave_range[1])
        self.flux[~mask_wave] *= np.nan
        self.err[~mask_wave]  *= np.nan

    def remove_empty_chips(self, remove_empty_pixels=False):
        """
        Remove any empty chips and optionally empty pixels.

        Args:
            remove_empty_pixels (bool): If True, remove empty pixels as well.
        """
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
        """
        Mask the spectrum inside specified wavelength ranges.

        Args:
            wave_ranges (list): List of wavelength ranges to mask.
            pad (float): Padding to add to each wavelength range.
        """
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