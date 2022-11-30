'''Utils function for AE notebook.

@Author  :   Jakob Schlör 
@Time    :   2022/03/18 15:09:03
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy as ctp


# Plotting
# ========

def create_map_plot(ax=None, ctp_projection='PlateCarrree',
                    central_longitude=0):
    """Generate cartopy figure for plotting.

    Args:
        ax ([type], optional): [description]. Defaults to None.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarrree'.
        central_longitude (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns:
        ax (plt.axes): Matplotplib axes object.
    """
    # set projection
    if ctp_projection == 'Mollweide':
        proj = ctp.crs.Mollweide(central_longitude=central_longitude)
    elif ctp_projection == 'EqualEarth':
        proj = ctp.crs.EqualEarth(central_longitude=central_longitude)
    elif ctp_projection == 'Robinson':
        proj = ctp.crs.Robinson(central_longitude=central_longitude)
    elif ctp_projection == 'PlateCarree':
        proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
    else:
        raise ValueError(
            f'This projection {ctp_projection} is not available yet!')
    
    if ax is None:
        fig, ax = plt.subplots()
        ax = plt.axes(projection=proj)

    # axes properties
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)    
    ax.add_feature(ctp.feature.RIVERS)
    ax.add_feature(ctp.feature.BORDERS, linestyle=':')

    return ax


def plot_map(dmap, central_longitude=0, vmin=None, vmax=None,
             ax=None, fig=None, color='RdBu_r', bar=True,
             ctp_projection='PlateCarree', label=None, **kwargs):
    """Simple map plotting using xArray.

    Args:
        dmap ([type]): [description]
        central_longitude (int, optional): [description]. Defaults to 0.
        vmin ([type], optional): [description]. Defaults to None.
        vmax ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.
        color (str, optional): [description]. Defaults to 'RdBu_r'.
        bar (bool, optional): [description]. Defaults to True.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarree'.
        label ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # create figure
    ax = create_map_plot(ax=ax, ctp_projection=ctp_projection,
                         central_longitude=central_longitude)

    # set colormap
    cmap = plt.get_cmap(color)
    kwargs_pl = dict() # kwargs plot function
    kwargs_cb = dict() # kwargs colorbar 
    if bar=='discrete':
        normticks = np.arange(0, dmap.max(skipna=True)+2, 1, dtype=int)
        kwargs_pl['norm'] = mpl.colors.BoundaryNorm(normticks, cmap.N)
        kwargs_cb['ticks'] = normticks + 0.5 
    
     # choose symmetric vmin and vmax
    if vmin is None and vmax is None:
        vmin = dmap.min(skipna=True)
        vmax = dmap.max(skipna=True)
        vmax = vmax if vmax > (-1*vmin) else (-1*vmin) 
        vmin = -1*vmax

    # plot map
    im = ax.pcolormesh(
        dmap.coords['lon'], dmap.coords['lat'], dmap.data,
        cmap=cmap, vmin=vmin, vmax=vmax, 
        transform=ctp.crs.PlateCarree(central_longitude=central_longitude),
        **kwargs_pl
    )

    # set colorbar
    shrink = kwargs.pop('shrink', 0.8)
    if bar:
        label = dmap.name if label is None else label
        cbar = plt.colorbar(im, extend='both', orientation='horizontal',
                            label=label, shrink=shrink, ax=ax, **kwargs_cb)

        if bar=='discrete':
            cbar.ax.set_xticklabels(normticks[:-1]+1)

    return {'ax': ax, "im": im}


def plot_2dgaussian(mu, cov, p=0.5, ax=None, **kwargs):
    """Plot 2d gaussian at p-confidence level.

    Args:
        mu (np.array): (2) Mean.
        cov (np.array): (2,2) Covariance matrix.
        p (float, optional): Confidence. Defaults to 0.95.
        ax (plt.Axes, optional): Axes. Defaults to None.

    Returns:
        [type]: [description]
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    s = -2 * np.log(1-p) 
    v, w = np.linalg.eigh(cov*s)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(
        mu, v[0], v[1], 180 + angle, **kwargs
    )
    
#    ell.set_clip_box(ax.bbox)
    ax.add_patch(ell)
    ax.autoscale_view()
    
    return ax
