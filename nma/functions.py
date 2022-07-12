import torch
import numpy as np


def create_spatial_grid(stim_ecc, gridsize):
    """This function samples the visual space for a given eccentricity coverage and gridsize.
    The x and y coordinate outputs arethen used for creating the pRF parameters (see create_prf_params function).
    NMA is strictly implemented for a formulated space, so the model will only work
    if the receptive fields are defined within the same grid as the input stimulus. """

    coord = np.sqrt(((stim_ecc)**2)/2)
    x_coordinates = torch.linspace(-coord, coord, gridsize)
    y_coordinates = torch.linspace(coord, -coord, gridsize)
    y_grid, x_grid = torch.meshgrid(x_coordinates, y_coordinates)
    y_grid = -1*y_grid
    x_grid = -1*x_grid
    return x_coordinates, y_coordinates, x_grid, y_grid


def prepare_stimulus(stim, gridsize):
    """This function uses 1D interpolation across columns and rows separately, and resizes the original
    stimulus input to the desired grid size"""
    old_dim, new_dim = stim.shape[0], gridsize
    col_im = np.zeros((new_dim, old_dim))
    stim_reshaped = np.zeros((new_dim, new_dim, stim.shape[2]))
    nls, ols = np.linspace(0, 1, new_dim), np.linspace(0, 1, old_dim)

    for stim_idx in range(0, stim.shape[2]):
        for col in range(old_dim):
            col_im[:, col] = np.interp(nls, ols, stim[:, col, stim_idx])
        for col in range(new_dim):
            stim_reshaped[col, :, stim_idx] = np.interp(nls, ols, col_im[col, :])

    stim_reshaped = torch.from_numpy(stim_reshaped)
    stim_reshaped = stim_reshaped.to(torch.float32)

    return stim_reshaped


def create_prf_centers_torch(x_coordinates, y_coordinates):
    """Combines each element of x coordinates with y coordinates to tile the visual
    field"""

    prf_parameters = np.zeros((2, len(x_coordinates) * len(y_coordinates)))
    iter_idx = 0

    for i in range(0, len(y_coordinates)):
        for j in range(0, len(x_coordinates)):
            prf_parameters[0, iter_idx] = x_coordinates[j]
            prf_parameters[1, iter_idx] = y_coordinates[i]
            iter_idx = iter_idx + 1

    prf_parameters = torch.from_numpy(prf_parameters)

    return prf_parameters[0, :], prf_parameters[1, :]


def create_gaussian_field(x_grid, y_grid, x, y, sigma, type=None, normalize=True, flat=True):
    """Creates a 2D Gaussian for the given X and Y center and the sigma factor.
    Has the option to 1) normalize the Gaussian field to the unit volume and
    2) flatten the Gaussian (vectorize it)"""

    gaussian = torch.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma) ** 2)

    if type == 'euclidean_distance':
        distance = torch.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
        gaussian = torch.exp(-.5 * (distance / sigma) ** 2)
    if normalize:
        gaussian = gaussian / torch.linalg.norm(gaussian)
    if flat:
        gaussian = gaussian.flatten()

    gaussian = gaussian.to(torch.float32)

    return gaussian


def interpolate_voxel_center(x_grid, y_grid, voxeldata):
    voxel_lookup_indices = torch.empty(voxeldata.shape[1], dtype=torch.long)
    distance_vector = torch.empty((len(x_grid) * len(y_grid), voxeldata.shape[1]))
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    for idx in range(0, voxeldata.shape[1]):
        distance_vector[:, idx] = torch.sqrt((voxeldata[0, idx] - x_grid) ** 2 + (voxeldata[1, idx] - y_grid) ** 2)
        voxel_lookup_indices[idx] = torch.argmin(torch.abs(distance_vector[:, idx]))

    return distance_vector, voxel_lookup_indices