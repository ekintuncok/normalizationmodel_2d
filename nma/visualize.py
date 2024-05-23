import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
from normalizationmodelofattention import NormalizationModelofAttention
from functions import prepare_stimulus
from functions import create_spatial_grid
from functions import create_prf_centers_torch

from mpl_toolkits.axes_grid1 import make_axes_locatable

# load the stimulus
stimpath = '/Users/ekintuncok/Documents/MATLAB/normalizationmodel_2d/matlab_scripts/stimfiles'
stim_ecc = 12
gridsize = 64

stimtemp = scipy.io.loadmat(stimpath + '/stim.mat')
stimtemp = stimtemp['stim']
stimorig = stimtemp[:,:,0:48]

input_stim = prepare_stimulus(stimorig, gridsize)
x_coordinates, y_coordinates, X, Y = create_spatial_grid(stim_ecc, gridsize)
lookup_prf_centers = create_prf_centers_torch(x_coordinates, y_coordinates)


#### 
normalization_sigma = 0.01
rf_sigma_factor = 0.05
attgain     = 4
attx0       = 0
atty0       = 5
attsd       = 1
sigmaNorm   = 0.01
suppWeight  = 3
summWeight  = 3
attention_ctr = [-4,0]

nma = NormalizationModelofAttention(normalization_sigma, attention_ctr, stim_ecc,
                                    prf_sigma_scale_factor = torch.tensor([0.06], dtype=torch.float32),
                                    attention_field_sigma = torch.tensor([1], dtype=torch.float32),
                                    attention_field_gain = torch.tensor([4], dtype=torch.float32),
                                    suppressive_surround_sigma_factor= torch.tensor([3], dtype=torch.float32),
                                    summation_field_sigma_factor=torch.tensor([3], dtype=torch.float32))


numerator, surroundresponse, neuralresponse, populationresponse = nma(input_stim)


numerator = numerator.detach().numpy()
surroundresponse = surroundresponse.detach().numpy()
neuralresponse = neuralresponse.detach().numpy()
populationresponse = populationresponse.detach().numpy()

### VISUALIZE
stimidx = 13

numerator_toplot = np.reshape(numerator[:,stimidx],(gridsize,gridsize))
surroundresponse_toplot = np.reshape(surroundresponse[:,stimidx],(gridsize,gridsize))
neuralresponse_toplot = np.reshape(neuralresponse[:,stimidx],(gridsize,gridsize))
populationresponse_toplot = np.reshape(populationresponse[:,stimidx],(gridsize,gridsize))

colorm = 'inferno'
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=[22,22])
im1 = ax1.imshow(numerator_toplot, cmap = colorm)
ax1.set_title('Stimulus drive under attention')
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="10%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1)

im2 = ax2.imshow(surroundresponse_toplot, cmap = colorm)
ax2.set_title('Suppressive drive under attention')
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="10%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2)

im3 = ax3.imshow(neuralresponse_toplot , cmap = colorm)
ax3.set_title('Normalized population response before summation')
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="10%", pad=0.05)
cbar3 = plt.colorbar(im3, cax=cax3)

im4 = ax4.imshow(populationresponse_toplot, cmap = colorm)
ax4.set_title('Normalized population response after summation');
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("right", size="10%", pad=0.05)
cbar4 = plt.colorbar(im4, cax=cax4)