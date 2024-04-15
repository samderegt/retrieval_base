import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def orthographic_to_xy(lat, lon, lat_0, lon_0, R=1):

    x = R * np.cos(lat) * np.sin(lon-lon_0)
    y = R * (np.cos(lat_0)*np.sin(lat) - np.sin(lat_0)*np.cos(lat)*np.cos(lon-lon_0))

    c = np.arccos(
        np.sin(lat_0)*np.sin(lat) + np.cos(lat_0)*np.cos(lat)*np.cos(lon-lon_0)
    )

    mask = (c > -np.pi/2) & (c < np.pi/2)
    x[~mask] = np.nan
    y[~mask] = np.nan
    return x, y, c

def plot_grid(
        ax, 
        lat_grid=np.arange(-90,90+1e-6,30), 
        lon_grid=np.arange(0,360,30), 
        inclination=26, 
        lon_0=0, 
        **kwargs
        ):

    lat = np.deg2rad(np.linspace(-90,90,100))
    lon = np.deg2rad(np.linspace(0,360,100))

    inclination = np.deg2rad(inclination)
    lon_0 = np.deg2rad(lon_0)

    for lat_i in np.deg2rad(lat_grid):

        # Orthographic projection
        x, y, c = orthographic_to_xy(lat_i, lon, inclination, lon_0)

        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)

        ax.plot(phi, r, **kwargs)

    for lon_i in np.deg2rad(lon_grid):
        
        # Orthographic projection
        x, y, c = orthographic_to_xy(lat, lon_i, inclination, lon_0)

        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)

        ax.plot(phi, r, **kwargs)

def plot_map(ax, attr, Rot, cax=None, **kwargs):

    z = getattr(Rot, attr)

    for i, r_i in enumerate(Rot.unique_r):
        th_i = Rot.theta_grid[Rot.r_grid==r_i]
        z_i  = z[Rot.r_grid==r_i]
        r_i  = np.array([r_i])

        th_i = np.concatenate((th_i-th_i.min(), [2*np.pi]))

        r_i = np.array([
            np.sin(Rot.unique_c[i] - (np.pi/2)/Rot.n_c/2), 
            np.sin(Rot.unique_c[i] + (np.pi/2)/Rot.n_c/2)*2, 
            ])

        zz_shape = (len(r_i)-1,len(th_i)-1)

        tt, rr = np.meshgrid(th_i, r_i)
        zz = z_i.reshape(zz_shape)

        #ax.plot(0, Rot.unique_r[i], 'k.')
        cntr = ax.pcolormesh(np.pi/2-tt, rr, zz, shading='auto', **kwargs)

    if cax is not None:
        import matplotlib as mpl

        norm = mpl.colors.Normalize(vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax'))
        plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=kwargs.get('cmap')), 
            ax=ax, cax=cax, orientation='vertical'
            )

def set_axis(ax, Rot, sep_spine_lw=None, grid=True):

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0,1)

    inc   = np.rad2deg(Rot.inc)
    lon_0 = np.rad2deg(Rot.lon_0)

    if grid:
        # Plot the spherical grid
        plot_grid(ax, inclination=inc, lon_0=lon_0, c='k', alpha=1, lw=0.4)

    if sep_spine_lw is None:
        return

    for key_i in ['start', 'end']:
        # Make the spine a separating white line
        ax.spines[key_i].set_linewidth(sep_spine_lw)
        ax.spines[key_i].set_color('w')
        ax.spines[key_i].set_capstyle('round')
        #ax.spines[key_i].set_joinstyle('round')

    # Plot a black spine around the inner axis edge
    ax.axvline(ax.get_xlim()[0], lw=sep_spine_lw+0.8*2, c='k')
    ax.axvline(ax.get_xlim()[1], lw=sep_spine_lw+0.8*2, c='k')

def set_cb_axis(cax, theta_lim, cmap, r_origin=-14):

    norm = mpl.colors.Normalize(*theta_lim)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.outline.set_visible(False)

    cax.spines['polar'].set_visible(True)
    cax.set_rlim(0,1)
    cax.set_rorigin(r_origin)
    cax.tick_params(top=True, bottom=False)
    cax.grid(False)

    return cb

def label_cb_axis(cax, ticks, val_range, label, annotate_kwargs):

    theta_min = np.min(cax.get_xlim())
    theta_max = np.max(cax.get_xlim())

    val_frac = (ticks - val_range[0]) / np.abs(val_range[1]-val_range[0])
    mask_valid_frac = (val_frac>=0) & (val_frac<=1)

    val_frac = val_frac[mask_valid_frac]
    ticks    = ticks[mask_valid_frac]

    ticks_theta = (theta_max-theta_min)*val_frac + theta_min

    cax.set_xticks(ticks_theta)
    cax.set_xticklabels(ticks)

    cax.annotate(label, **annotate_kwargs)

def set_thetalim(ax, theta_cen, theta_width):
    ax.set_thetalim(theta_cen-theta_width/2, theta_cen+theta_width/2)

def rescale_and_center(
        fig, 
        ax, 
        ax_ref, 
        coord_min=(0,0), 
        coord_max=(0,1), 
        x_offset=0, 
        y_offset=0, 
        scale=1
        ):
    
    # Compute the size (in figure fraction) of the axes
    xy_ref_0 = fig.transFigure.inverted().transform(ax_ref.transData.transform((0,0)))
    xy_ref_1 = fig.transFigure.inverted().transform(ax_ref.transData.transform((0,1)))
    r_ref = np.sqrt((xy_ref_0[0]-xy_ref_1[0])**2 + (xy_ref_0[1]-xy_ref_1[1])**2)

    xy_min = fig.transFigure.inverted().transform(ax.transData.transform(coord_min))
    xy_max = fig.transFigure.inverted().transform(ax.transData.transform(coord_max))
    r = np.sqrt((xy_max[0]-xy_min[0])**2 + (xy_max[1]-xy_min[1])**2)

    # Bbox coordinates
    l_ref, b_ref, w_ref, h_ref = ax_ref.get_position().bounds
    l, b, w, h = ax.get_position().bounds

    # Re-scale the axis to match width/height of ax_ref
    ax.set_position([l, b, w*(scale*r_ref/r), h*(scale*r_ref/r)])

    # Update the coordinates
    l, b, w, h = ax.get_position().bounds

    # Update coordinates of axis center
    xy_min = fig.transFigure.inverted().transform(ax.transData.transform(coord_min))

    # Correct the offsets
    delta_xy = xy_ref_0 - xy_min
    if x_offset != 0:
        x_offset += w_ref
    ax.set_position([l+delta_xy[0]+x_offset, b+delta_xy[1]+y_offset, w, h])

def plot_multicolor_line(ax, x, y, z, vmin, vmax, cmap='viridis', **kwargs):
    
    points = np.array([x, y]).T[:,None,:]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    lc = mpl.collections.LineCollection(
        segments, cmap=cmap, norm=norm, capstyle='round', **kwargs
        )
    
    # Set the values used for colormapping
    lc.set_array(z)
    line = ax.add_collection(lc)