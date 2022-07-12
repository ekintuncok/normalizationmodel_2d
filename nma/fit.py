import scipy.io
import torch
import csv
import numpy as np
from functions import prepare_stimulus
from normalizationmodelofattention import NormalizationModelofAttention
from tqdm import tqdm

project_path = '/scratch/et2160/nma/'
#project_path = '/Volumes/server/Projects/attentionpRF/Simulations/python_scripts/'
stimpath = project_path + 'stimfiles' + '/stim.mat'
save_pth = project_path + '/results'
save_name = '/model_output.csv'

stimtemp = scipy.io.loadmat(stimpath)
stimtemp = torch.from_numpy(stimtemp['stim']).to(torch.float32)
stimorig = stimtemp[:,:,0:48]
gridsize = 64
stim_ecc = 10

input_stim = prepare_stimulus(stimorig, gridsize)
attention_ctr = (5,0)
normalization_sigma = 0.01

# load the simulated data and the pRF centers for the simulated data
simulated_data_dir = '/Volumes/server/Projects/attentionpRF/Simulations/python_scripts/data/'
measured_response_simulated = np.load(simulated_data_dir + 'predicted_response.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
voxeldata_simulated = np.load(simulated_data_dir + 'simulated_prf_centers.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
measured_response_simulated = torch.from_numpy(measured_response_simulated)
measured_response_simulated = measured_response_simulated.to(torch.float32)


nma_fit = NormalizationModelofAttention(normalization_sigma, voxeldata_simulated, attention_ctr, stim_ecc)
predicted_voxel_response = nma_fit(input_stim)

optimizer = torch.optim.Adam([nma_fit.voxel_gain, # estimated per voxel
                              nma_fit.prf_sigma_scale_factor, # estimated for the ROI ? slope
                              nma_fit.attention_field_sigma, #  estimated for the ROI
                              nma_fit.attention_field_gain, #  estimated for the ROI
                              nma_fit.suppressive_surround_sigma_factor, #estimated per voxel? not sure if we will change
                              nma_fit.summation_field_sigma_factor],  # estimated for the ROI
                              lr=1e-1)

# define the loss function
loss_func = torch.nn.MSELoss()
losses = []
param_vals = {'prf_sigma_scale_factor': torch.tensor([]), 'attention_field_sigma': torch.tensor([]), 'attention_field_gain': torch.tensor([]),
              'suppressive_surround_sigma_factor': torch.tensor([]), 'summation_field_sigma_factor': torch.tensor([])}
number_of_iterations = 500
current_params = {}
pbar = tqdm(range(number_of_iterations))
for i in pbar:
    predicted_voxel_response = nma_fit(input_stim)
    loss = loss_func(predicted_voxel_response, measured_response_simulated)
    optimizer.zero_grad() # resetting the calculated gradients
    loss.backward() # this computes the gradient
    optimizer.step()
    losses.append(loss.item())
    for k, v in nma_fit.named_parameters():
        current_params[k] = v.clone().detach()
        param_vals[k] = torch.cat([param_vals[k], v.clone().detach()], -1)
    pbar.set_postfix(loss=losses[-1], **current_params)

# save the output
w = csv.writer(open(save_pth+save_name, "w"))

for key, val in param_vals.items():
    w.writerow([key,val])

w.writerow(['losses',losses])
