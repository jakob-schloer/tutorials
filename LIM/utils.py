''' Utils functions for LIM. 

@Author  :   Jakob Schlör 
@Time    :   2023/03/16 13:33:56
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

def map2flatten(x_map: xr.Dataset) -> list:
    """Flatten dataset/dataarray and remove NaNs.

    Args:
        x_map (xr.Dataset/ xr.DataArray): Dataset or DataArray to flatten. 

    Returns:
        x_flat (xr.DataArray): Flattened dataarray without NaNs 
        ids_notNaN (xr.DataArray): Boolean array where values are on the grid. 
    """
    if type(x_map) == xr.core.dataset.Dataset:
        x_stack_vars = [x_map[var] for var in list(x_map.data_vars)]
        x_stack_vars = xr.concat(x_stack_vars, dim='var')
        x_stack_vars = x_stack_vars.assign_coords({'var': list(x_map.data_vars)})
        x_flatten = x_stack_vars.stack(z=('var', 'lat', 'lon')) 
    else:
        x_flatten = x_map.stack(z=('lat', 'lon'))

    # Flatten and remove NaNs
    if 'time' in x_flatten.dims:
        idx_notNaN = ~np.isnan(x_flatten.isel(time=0))
    else:
        idx_notNaN = ~np.isnan(x_flatten)
    x_proc = x_flatten.isel(z=idx_notNaN.data)

    return x_proc, idx_notNaN


def flattened2map(x_flat: np.ndarray, ids_notNaN: xr.DataArray, times: np.ndarray = None) -> xr.Dataset:
    """Transform flattened array without NaNs to gridded data with NaNs. 

    Args:
        x_flat (np.ndarray): Flattened array of size (n_times, n_points) or (n_points).
        ids_notNaN (xr.DataArray): Boolean dataarray of size (n_points).
        times (np.ndarray): Time coordinate of xarray if x_flat has time dimension.

    Returns:
        xr.Dataset: Gridded data.
    """
    if len(x_flat.shape) == 1:
        x_map = xr.full_like(ids_notNaN, np.nan, dtype=float)
        x_map[ids_notNaN.data] = x_flat
    else:
        temp = np.ones((x_flat.shape[0], ids_notNaN.shape[0])) * np.nan
        temp[:, ids_notNaN.data] = x_flat
        if times is None:
            times = np.arange(x_flat.shape[0]) 
        x_map = xr.DataArray(data=temp, coords={'time': times, 'z': ids_notNaN['z']})

    if 'var' in list(x_map.get_index('z').names):
        x_map = x_map.unstack()

        if 'var' in list(x_map.dims): # For xr.Datasset only
            da_list = [xr.DataArray(x_map.isel(var=i), name=var) 
                       for i, var in enumerate(x_map['var'].data)]
            x_map = xr.merge(da_list, compat='override')
            x_map = x_map.drop(('var'))
    else:
        x_map = x_map.unstack()
    
    return x_map


class SpatioTemporalPCA:
    """PCA of spatio-temporal data.

    Wrapper for sklearn.decomposition.PCA with xarray.DataArray input.
     
    See EOF tutorial.

    Args:
        ds (xr.DataArray or xr.Dataset): Input dataarray to perform PCA on.
            Array dimensions (time, 'lat', 'lon')
        n_components (int): Number of components for PCA
    """
    def __init__(self, ds, n_components, **kwargs):
        self.ds = ds

        self.X, self.ids_notNaN = map2flatten(self.ds)

        # PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.X.data)

        self.n_components = self.pca.n_components


    def eofs(self):
        """Return components of PCA.

        Return:
            components (xr.dataarray): Size (n_components, N_x, N_y)
        """
        # EOF maps
        eof_map = []
        for i, comp in enumerate(self.pca.components_):
            eof = flattened2map(comp, self.ids_notNaN)
            eof = eof.drop(['time', 'month'])
            eof_map.append(eof)

        return xr.concat(eof_map, dim='eof')
    

    def principal_components(self):
        """Returns time evolution of components.

        Return:
            time_evolution (xr.Dataarray): Principal components of shape (n_components, time)
        """
        time_evolution = []
        for i, comp in enumerate(self.pca.components_):
            ts = self.X.data @ comp

            da_ts = xr.DataArray(
                data=ts,
                dims=['time'],
                coords=dict(time=self.X['time']),
            )
            time_evolution.append(da_ts)
        return xr.concat(time_evolution, dim='eof')
    

    def explained_variance(self):
        return self.pca.explained_variance_ratio_
    

    def reconstruction(self, z, newdim='x'):
        """Reconstruct the dataset from components and time-evolution.

        Args:
            z (np.ndarray): Low dimensional vector of size (time, n_components)
            newdim (str, optional): Name of dimension. Defaults to 'x'.

        Returns:
            _type_: _description_
        """
        reconstruction = z.T @ self.pca.components_

        rec_map = []
        for rec in reconstruction:
            x = flattened2map(rec, self.ids_notNaN)
            rec_map.append(x)

        rec_map = xr.concat(rec_map, dim=newdim)

        return rec_map


def matrix_decomposition(A):
    """Decompose square matrix:
    
        A = U D V.T
    Args:
        A (np.ndarray): Square matrix 

    Returns:
        w (np.ndarray): Sorted eigenvalues
        U (np.ndarray): Matrix of sorted eigenvectors of A
        V (np.ndarray): Matrix of sorted eigenvectors of A.T 
    """
    w, U = np.linalg.eig(A)
    idx_sort = np.argsort(w)[::-1]
    w = w[idx_sort]
    U = U[:, idx_sort]

    w_transpose, V = np.linalg.eig(A.T)
    idx_sort = np.argsort(w_transpose)[::-1]
    V = V[:, idx_sort]

    return w, U, V