import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
from nma.functions import prepare_stimulus
from nma.functions import create_spatial_grid
from nma.functions import create_gaussian_field
from nma.functions import cart2pol
import nibabel.freesurfer.mghformat as mgh

model_path = '/Volumes/server/Projects/attentionpRF/Simulations/Attention_pRF/'
project_path = '/Volumes/server/Projects/attentionpRF/'
stimpath = model_path + 'stimfiles'
subject = 'sub-wlsubj141'
session = 'ses-nyu3t99'
prf_path = 'derivatives/prfs/' + subject + '/' + session + '/prfFolder/NYU_3T/'

eccen = 10
gridsize = 101
x_coordinates, y_coordinates, x_grid, y_grid = create_spatial_grid(eccen, gridsize)

stimtemp = scipy.io.loadmat(stimpath + '/stim.mat')
stimtemp = torch.from_numpy(stimtemp['stim']).to(torch.float32)
stimorig = stimtemp[:, :, 0:48]
input_stim = prepare_stimulus(stimorig, gridsize)

attention_ctr = (-6, 0)
attention_field_sigma = 3.5
attfield_gauss = create_gaussian_field(x_grid, y_grid, attention_ctr[0], attention_ctr[1], attention_field_sigma, False,
                                       False)

# load the subject data:
lh_x = mgh.load(project_path + prf_path + 'lh.x.mgz')
lh_x = lh_x.get_fdata()
rh_x = mgh.load(project_path + prf_path + 'rh.x.mgz')
rh_x = rh_x.get_fdata()
x = np.concatenate((lh_x, rh_x), axis=0)
x = x[:, 0, 0]
x = np.expand_dims(x, axis=1)

lh_y = mgh.load(project_path + prf_path + 'lh.y.mgz')
lh_y = lh_y.get_fdata()
rh_y = mgh.load(project_path + prf_path + 'rh.y.mgz')
rh_y = rh_y.get_fdata()
y = np.concatenate((lh_y, rh_y), axis=0)
y = y[:, 0, 0]
y = np.expand_dims(y, axis=1)
y = -1 * y

# load the sigma values of the stimulus driven pRFs:
lh_sigma = mgh.load(project_path + prf_path + 'lh.sigma.mgz')
lh_sigma = lh_sigma.get_fdata()
rh_sigma = mgh.load(project_path + prf_path + 'rh.sigma.mgz')
rh_sigma = rh_sigma.get_fdata()
sigma = np.concatenate((lh_sigma, rh_sigma), axis=0)
sigma = sigma[:, 0, 0]
sigma = np.expand_dims(sigma, axis=1)

# load the ROIs:
lh_ROIs = mgh.load(project_path + 'derivatives/' + 'freesurfer/' + subject + '/surf/lh.ROIs_V1-V4.mgz')
lh_ROIs = lh_ROIs.get_fdata()
rh_ROIs = mgh.load(project_path + 'derivatives/' + 'freesurfer/' + subject + '/surf/rh.ROIs_V1-V4.mgz')
rh_ROIs = rh_ROIs.get_fdata()
roi = np.concatenate((lh_ROIs, rh_ROIs), axis=2)
roi_mask = roi[0, 0, :]

# create ROI labels:
ROIs = {"V1": 1, "V2": 2, "V3": 3, "V3AB": 4, "hV4": 5}
labels = np.empty((x.shape[0], 5))
for roi in range(1, len(ROIs) + 1):
    labels[:, roi - 1] = roi_mask == roi

binary_bin_masks = np.empty((2, 8))
binary_bin_masks[0, :] = np.linspace(1, 8, 8).astype(int)
binary_bin_masks[1, :] = np.linspace(9, 16, 8).astype(int)

# implement the model:
for roi in range(1, len(ROIs) + 1):
    roi_mask = labels[:, roi - 1].astype(int)
    eccen_indices = np.argwhere(np.sqrt(x ** 2 + y ** 2) < 6)
    roi_indices = np.argwhere(roi_mask == 1)
    data_to_model = np.intersect1d(roi_indices, eccen_indices)
    masked_data = np.concatenate((x[data_to_model], y[data_to_model], sigma[data_to_model]), axis=1)

    # calculate the Klein (2014) model predictions for individual voxels:

    predicted_attend_sigma = np.sqrt(
        np.multiply(attention_field_sigma ** 2, masked_data[:, 2] ** 2) / (attention_field_sigma ** 2 +
                                                                           masked_data[:, 2] ** 2))
    predicted_attend_x = (np.multiply(attention_ctr[0], masked_data[:, 2] ** 2) + np.multiply(masked_data[:, 0],
                                                                                              attention_field_sigma ** 2)) / (
                                     attention_field_sigma ** 2 + masked_data[:, 2] ** 2)
    predicted_attend_y = (np.multiply(attention_ctr[1], masked_data[:, 2] ** 2) + np.multiply(masked_data[:, 1],
                                                                                              attention_field_sigma ** 2)) / (
                                     attention_field_sigma ** 2 + masked_data[:, 2] ** 2)

    # calculate the distance change, this metric will be called shift:
    base_distance = np.sqrt((attention_ctr[0] - masked_data[:, 0]) ** 2 + (attention_ctr[1] - masked_data[:, 1]) ** 2)
    predicted_new_distance = np.sqrt(
        (attention_ctr[0] - predicted_attend_x) ** 2 + (attention_ctr[1] - predicted_attend_y) ** 2)

    # calculate the Klein (2014) model predictions for binned voxels (this is what they are doing in the paper)
    eccen, polar_angle_rad = cart2pol(x[data_to_model], y[data_to_model])
    angle = polar_angle_rad * (180 / np.pi)

    eccen_binning = np.argwhere((eccen > 0.5) & (eccen < 1.25))

plt.figure()
plt.scatter(predicted_new_distance, base_distance)
plt.xlim([0, 16])
plt.ylim([0, 16])
plt.show()

plt.figure()
plt.scatter(np.abs(predicted_new_distance - base_distance), predicted_attend_sigma - masked_data[:, 2])
plt.show()
