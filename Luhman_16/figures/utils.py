import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import retrieval_base.auxiliary_functions as af

import petitRADTRANS.nat_cst as nc

from tqdm import tqdm
import copy

def get_cmap(colors=['#4A0C0C','#fff0e6']):
    cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', colors)
    cmap.set_bad('w'); cmap.set_over('w'); cmap.set_under('k')
    return cmap

def high_pass_filter(flux):
    # Apply Savitzky-Golay filter to remove broad structure
    from scipy.signal import savgol_filter
    lp_flux = savgol_filter(
        flux, window_length=301, polyorder=2, mode='nearest'
        )
    return flux - lp_flux

def convert_CCF_to_SNR(rv, CCF, rv_sep=100):
    # Convert the cross-correlation function to a S/N function
    rv_mask = np.abs(rv) > rv_sep

    mean_CCF = np.mean(CCF[rv_mask])
    std_CCF  = np.std(CCF[rv_mask])

    return (CCF - mean_CCF) / std_CCF

class RetrievalResults:
    
    def __init__(self, prefix, m_set='K2166_cloudy', w_set='K2166'):

        self.prefix = prefix
        self.m_set  = m_set
        self.w_set  = w_set

        self.n_params = self.load_object('LogLike').N_params
            
        import pymultinest
        # Set-up analyzer object
        analyzer = pymultinest.Analyzer(
            n_params=self.n_params, outputfiles_basename=prefix
            )
        stats = analyzer.get_stats()
        self.ln_Z = stats['nested importance sampling global log-evidence']

        # Read the parameters of the best-fitting model
        self.bestfit_values = np.array(stats['modes'][0]['maximum a posterior'])

        import json
        # Read the best-fitting parameters
        with open(self.prefix+'data/bestfit.json') as f:
            self.bestfit_params = json.load(f)['params']

    def compare_evidence(self, ln_Z_other):
        '''Convert log-evidences of two models to a sigma confidence level'''

        from scipy.special import lambertw as W
        from scipy.special import erfcinv

        ln_B = self.ln_Z - ln_Z_other
        B = np.exp(ln_B)
        p = np.real(np.exp(W((-1.0/(B*np.exp(1))),-1)))
        sigma = np.sqrt(2)*erfcinv(p)

        _ln_B = ln_Z_other - self.ln_Z
        _B = np.exp(_ln_B)
        _p = np.real(np.exp(W((-1.0/(_B*np.exp(1))),-1)))
        _sigma = np.sqrt(2)*erfcinv(_p)

        print('Current vs. given: lnB={:.2f} | sigma={:.2f}'.format(ln_B, sigma))
        print('Given vs. current: lnB={:.2f} | sigma={:.2f}'.format(_ln_B, _sigma))
        return B, sigma

    def load_object(self, name, bestfit_prefix=True):

        setting = self.m_set
        if name == 'd_spec':
            setting = self.w_set

        file_name = f'{self.prefix}data/bestfit_{name}_{setting}.pkl'
        if not bestfit_prefix:
            file_name = file_name.replace('bestfit_', '')

        try:
            return af.pickle_load(file_name)
        except FileNotFoundError:
            pass

        file_name = f'{self.prefix}data/bestfit_{name}.pkl'
        if not bestfit_prefix:
            file_name = file_name.replace('bestfit_', '')

        if name in ['LogLike', 'Cov']:
            return af.pickle_load(self.prefix+f'data/bestfit_{name}.pkl')
        
    def load_objects_as_attr(self, names):

        if isinstance(names, str):
            names = [names]

        for name in names:
            if hasattr(self, name):
                continue
            setattr(self, name, self.load_object(name))
    
    def get_model_spec(self, is_local=False, m_set=None, line_species=None):
        
        # Load the necessary objects
        self.load_objects_as_attr(['Chem', 'PT'])
        pRT_atm = self.load_object('pRT_atm_broad', bestfit_prefix=False)

        # Change the parameters for a local, un-broadened model
        if m_set is None:
            m_set = self.m_set
        params = copy.copy(self.bestfit_params[m_set])

        mf = copy.deepcopy(self.Chem.mass_fractions)

        if is_local:
            params['vsini'] = 0

            # Remove any spots/bands
            keys_to_delete = [
                'lat_band', 'lon_band', 'lon_spot', 'lon_spot_0', 
                'is_within_patch', 'is_outside_patch'
                ]
            for key in keys_to_delete:
                params.pop(key, None)

            # Change the local abundances
            if line_species is not None:
                for key_i in mf.keys():
                    if key_i == line_species:
                        continue
                    # Set abundances of other species to 0
                    mf[key_i] *= 0

        # Only consider a single model
        pRT_atm.sum_m_spec = False

        # Update the pRT model
        m_spec = pRT_atm(
            mf, self.PT.temperature, params, 
            get_contr=False, get_full_spectrum=True
            )
        
        wave_pRT_grid = copy.copy(pRT_atm.wave_pRT_grid)
        flux_pRT_grid = copy.copy(pRT_atm.flux_pRT_grid)

        # Rotation model
        Rot = copy.copy(pRT_atm.Rot)

        return wave_pRT_grid, flux_pRT_grid, Rot
    
    def add_patch_to_Rot(self, Rot, Rot_spot):

        # Fill in the patch with another Rot-instance
        mask_patch = Rot_spot.included_segments
        Rot.int_flux[mask_patch] = Rot_spot.int_flux[mask_patch]
    
    def get_CCF(
            self, wave_local, flux_local, flux_global, 
            rv=np.arange(-300,300+1e-6,0.5), 
            subtract_global=False, 
            high_pass={'m_res': high_pass_filter}, 
            **kwargs
            ):

        # Load the necessary objects
        self.load_objects_as_attr(['Cov', 'LogLike'])
        self.d_spec = self.load_object('d_spec', bestfit_prefix=False)

        # Cross-correlation
        CCF = np.nan * np.ones(
            (len(rv), self.d_spec.n_orders, self.d_spec.n_dets)
            )
    
        d_wave = np.copy(self.d_spec.wave)
        # Residual wrt the combined model
        d_res = np.copy(self.d_spec.flux) - np.copy(self.LogLike.m_flux_phi)

        if high_pass.get('d_res') is not None:
            # Apply a high-pass filter
            d_res = high_pass.get('d_res')(d_res)

        for i, rv_i in enumerate(tqdm(rv)):

            for j in range(self.d_spec.n_orders):

                # Shift the model spectrum
                m_wave_local = np.copy(wave_local[j]) * (1+rv_i/(nc.c*1e-5))
                m_flux_local = np.copy(flux_local[j])

                for k in range(self.d_spec.n_dets):

                    # Ignore the nans
                    mask_jk = self.d_spec.mask_isfinite[j,k]
                    if not mask_jk.any():
                        continue

                    m_res_local_jk = m_flux_local.copy()

                    # Subtract the global model from the template
                    if subtract_global:
                        m_res_local_jk -= flux_global[j]

                    # Interpolate onto the data wavelength-grid
                    m_res_local_jk = np.interp(
                        d_wave[j,k], xp=m_wave_local, fp=m_res_local_jk
                        )
                    m_res_local_jk *= self.LogLike.phi[j,k,0]

                    if high_pass.get('m_res') is not None:
                        # Apply a high-pass filter
                        m_res_local_jk = high_pass.get('m_res')(m_res_local_jk)
                    
                    # Compute the cross-correlation coefficient
                    CCF[i,j,k] = np.dot(
                        m_res_local_jk[mask_jk], 
                        1/self.LogLike.s2[j,k] * self.Cov[j,k].solve(d_res[j,k,mask_jk])
                    )
        
        CCF_SNR = convert_CCF_to_SNR(rv, CCF.sum(axis=(1,2)), **kwargs)

        return rv, CCF, CCF_SNR

class SpherePlot:

    def _latlon_to_xy(cls, lat, lon, lat_0, lon_0, R=1):

        x = R * np.cos(lat) * np.sin(lon-lon_0)
        y = R * (np.cos(lat_0)*np.sin(lat) - \
                 np.sin(lat_0)*np.cos(lat)*np.cos(lon-lon_0))

        # Check if coordinate is outside observed half-sphere
        c = np.arccos(
            np.sin(lat_0)*np.sin(lat) + \
            np.cos(lat_0)*np.cos(lat)*np.cos(lon-lon_0)
        )

        mask = (c > -np.pi/2) & (c < np.pi/2)
        x[~mask] = np.nan
        y[~mask] = np.nan
        
        return x, y

    def _latlon_to_polar(cls, lat, lon, lat_0, lon_0, R=1):
        
        # Convert to Cartesian? coordinates
        x, y = cls._latlon_to_xy(lat, lon, lat_0, lon_0, R)

        # Convert to polar coordinates
        r   = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        return r, phi

    def __init__(self, Rot, fig, ax, cax=None):

        self.Rot = Rot

        self.fig = fig
        self.ax  = ax
        self.cax = cax

    def plot_map(self, attr, cmap=None, **kwargs):

        z = getattr(self.Rot, attr)
        if kwargs.get('vmin') is None:
            kwargs['vmin'] = np.nanmin(z)
        if kwargs.get('vmax') is None:
            kwargs['vmax'] = np.nanmax(z)

        for i, r_i in enumerate(self.Rot.unique_r):
            th_i = self.Rot.theta_grid[self.Rot.r_grid==r_i]
            z_i  = z[self.Rot.r_grid==r_i]
            r_i  = np.array([r_i])

            th_i = np.concatenate((th_i-th_i.min(), [2*np.pi]))

            r_i = np.array([
                np.sin(self.Rot.unique_c[i] - (np.pi/2)/self.Rot.n_c/2), 
                np.sin(self.Rot.unique_c[i] + (np.pi/2)/self.Rot.n_c/2*2), 
                ])

            zz_shape = (len(r_i)-1,len(th_i)-1)

            tt, rr = np.meshgrid(th_i, r_i)
            zz = z_i.reshape(zz_shape)

            cntr = self.ax.pcolormesh(
                np.pi/2-tt, rr, zz, shading='auto', cmap=cmap, **kwargs
                )
            
    def configure_cax(
            self, label=None, xlim=None, xticks=None, vmin=0, vmax=1, 
            cb_width=0.1, scale=1.1, N=256, cmap=None, flip_cb=False, 
            ):

        # Define a grid to plot with pcolormesh
        Z = np.array([np.linspace(0,1+1e-6,N)]*2)
        X = Z.copy()
        if flip_cb:
            X = np.flip(X, axis=-1)
        Y = np.array([np.zeros(N),np.ones(N)])

        self.cax.pcolormesh(
            X, Y, Z, transform=self.cax.transAxes, 
            cmap=cmap, vmin=0, vmax=1, zorder=-1, 
            shading='auto', edgecolors='face', lw=1e-6, 
            )
        
        # Remove ticks/grid
        self.cax.spines['polar'].set_visible(True)
        self.cax.tick_params(top=True, bottom=False)
        self.cax.grid(False)

        self.cax.set(ylim=(0,1), yticks=[])

        # Cut hole in polar axis, creates arch
        r_origin = self.cax.get_ylim()[1] - \
            (self.cax.get_ylim()[1]-self.cax.get_ylim()[0]) / cb_width
        self.cax.set_rorigin(r_origin)

        if xlim is not None:
            # Change the shown angle, rescale and center
            self.set_thetalim(xlim, ax=self.cax, scale=scale)

        if xticks is not None:
            # Add ticks to the colorbar
            theta_min = np.min(self.cax.get_xlim())
            theta_max = np.max(self.cax.get_xlim())

            xticks = np.array(xticks)

            frac   = (xticks-vmin) / (vmax-vmin)
            xticks = xticks[(frac>=0) & (frac<=1)]
            frac   = frac[(frac>=0) & (frac<=1)]

            theta_ticks = (theta_max-theta_min)*frac + theta_min
            self.cax.set_xticks(theta_ticks)
            self.cax.set_xticklabels(xticks)

        if label is not None:
            # Add a label to the axis
            self.cax.annotate(
                label, xy=(0.5,3.5*scale), xycoords=self.cax.get_yaxis_transform(), 
                ha='center', va='center'
                )
            
    def configure_ax(self, xlim=None, sep_spine_lw=5, plot_grid=True, grid_lw=0.4):

        self.ax.grid(False)
        self.ax.set(xticks=[], yticks=[], ylim=(0,1))

        if xlim is not None:
            # Change the shown angle, rescale and center
            self.set_thetalim(xlim)

        if plot_grid:
            # Plot the latitude/longitude grid-lines
            self.grid(
                inc=np.rad2deg(self.Rot.inc), 
                lon_0=np.rad2deg(self.Rot.lon_0), 
                c='k', lw=grid_lw, 
                )
        
        if sep_spine_lw is None:
            return
        
        for i, key_i in enumerate(['start', 'end']):
            # Make the spine a separating white line
            self.ax.spines[key_i].set_linewidth(sep_spine_lw)
            self.ax.spines[key_i].set_color('w')
            self.ax.spines[key_i].set_capstyle('round')
            
            # Plot a black spine around the inner axis edge
            self.ax.axvline(self.ax.get_xlim()[i], lw=sep_spine_lw+0.8*2, c='k')

    def grid(
            self, 
            lat_grid=np.arange(-90,90+1e-6,30), 
            lon_grid=np.arange(0,360,30), 
            inc=0, lon_0=0, **kwargs
            ):
        
        # Fine grid of the lines
        lat = np.deg2rad(np.linspace(-90,90,100))
        lon = np.deg2rad(np.linspace(0,360,100))

        inc   = np.deg2rad(inc)
        lon_0 = np.deg2rad(lon_0)

        # Convert to the plot-coordinates and draw
        for lat_i in np.deg2rad(lat_grid):
            r, phi = self._latlon_to_polar(lat_i, lon, inc, lon_0)
            self.ax.plot(phi, r, **kwargs)

        for lon_i in np.deg2rad(lon_grid):
            r, phi = self._latlon_to_polar(lat, lon_i, inc, lon_0)
            self.ax.plot(phi, r, **kwargs)

    def set_thetalim(self, xlim, ax=None, scale=1):

        if ax is None:
            ax = self.ax

        r_origin = ax.get_rorigin()

        # Establish current figure coordinates
        xy_center = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,r_origin))
            )
        xy_right = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,1))
            )
        width = xy_right[0] - xy_center[0]

        # Change thetalim
        ax.set_thetalim(xlim)

        # Coordinates after thetalim change
        new_xy_center = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,r_origin))
            )
        new_xy_right = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,1))
            )
        new_width = new_xy_right[0] - new_xy_center[0]

        scale *= width/new_width

        # Rescale the axis
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l, b, w*scale, h*scale])

        # Coordinates after re-scaling
        new_xy_center = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,r_origin))
            )
        
        xy_offset = xy_center - new_xy_center

        # Recenter the axis
        l, b, w, h = ax.get_position().bounds
        ax.set_position(
            [l+xy_offset[0], b+xy_offset[1], w, h]
            )
        
class CrossCorrPlot:
    
    def __init__(self, fig, ax, cmap='RdBu_r', vsini=0):

        self.fig = fig
        self.ax  = ax

        self.cmap = cmap
        self.vsini = vsini

    def colorbar(self, h=0.03):

        ylim = self.ax.get_ylim()

        # Plot a colorbar at the bottom of the axis
        Z = np.linspace(0,1+1e-6,50).reshape(-1,1).T
        self.ax.imshow(
            Z, extent=[-self.vsini,self.vsini,0,h], origin='lower', 
            aspect='auto', transform=self.ax.get_xaxis_transform(), 
            zorder=-2, cmap=self.cmap, vmin=0, vmax=1, 
            )
        
        self.ax.set(ylim=ylim)
        
    def multicolor(self, rv, CCF, lw, **kwargs):

        points = np.array([rv, CCF]).T[:,None,:]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(vmin=-self.vsini, vmax=self.vsini)
        lc = mpl.collections.LineCollection(
            segments, norm=norm, cmap=self.cmap, capstyle='round', lw=lw
            )
        
        # Set the values used for colormapping
        lc.set_array(rv)
        self.ax.add_collection(lc)

    def plot(
            self, rv, CCF, CCF_other=None, plot_colorbar=False, plot_multicolor=True, **kwargs
            ):

        self.ax.plot(rv, CCF, **kwargs)

        new_kwargs = kwargs.copy()

        if plot_multicolor:
            # Plot a multi-colored line
            rv_mask = (np.abs(rv) < self.vsini)
            new_kwargs['lw'] = kwargs['lw']*4/3
            self.ax.plot(
                rv[rv_mask], CCF[rv_mask], solid_capstyle='round', **new_kwargs
                )
            
            new_kwargs['lw'] = kwargs['lw']*1.7/3           
            self.multicolor(rv[rv_mask], CCF[rv_mask], **new_kwargs)

        if CCF_other is not None:
            new_kwargs['lw']    = kwargs['lw']*1/2
            new_kwargs['alpha'] = 0.3
            self.ax.plot(rv, CCF_other, **new_kwargs, zorder=-1)

        if plot_colorbar:
            # Colorbar at the bottom of the axis
            self.colorbar()

    def configure_ax(
            self, xlabel=r'$v_\mathrm{rad}\ \mathrm{(km\ s^{-1})}$', 
            ylabel=None, xlim=(-120,120), ylim=(-6,6)
            ):

        self.ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        
        self.ax.axvline(0, ymin=0, ymax=1, c='k', lw=0.8, zorder=-1)
        self.ax.spines[['right','top']].set_visible(False)
        