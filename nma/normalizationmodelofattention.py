import torch
from functions import create_spatial_grid
from functions import create_gaussian_field
from functions import create_prf_centers_torch
from functions import interpolate_voxel_center


class NormalizationModelofAttention(torch.nn.Module):
    def __init__(self, normalization_sigma, attention_ctr, stim_ecc=10, voxeldata=None, prf_sigma_scale_factor=None,
                 attention_field_sigma=None,
                 attention_field_gain=None,
                 suppressive_surround_sigma_factor=None,
                 summation_field_sigma_factor=None):
        """
        The model is initialized using the X and Y coordinates of the estimated pRFs for each voxel,
        total number of voxels (this can be indexed on the first input), grid size (this will depend on how precise
        we want to get?), stimulys eccentricity (max) and the X and Y coordinates of attention field
        """
        super().__init__()
        self.gridsize = 64
        x_coordinates, y_coordinates, self.X, self.Y = create_spatial_grid(stim_ecc, self.gridsize)
        self.RF_X, self.RF_Y = create_prf_centers_torch(x_coordinates, y_coordinates)

        if eval('voxeldata') is None:
            self.distance_vector = []
            self.voxel_lookup_indices = range(0, self.gridsize * self.gridsize)
            print('Voxel data is not inputted, using the full RF lookup indices to simulate data...')
        else:
            self.distance_vector, self.voxel_lookup_indices = interpolate_voxel_center(self.X, self.Y, voxeldata)

        self.attention_ctr = attention_ctr
        self.ROI_normalization_sigma_factor = normalization_sigma
        self.voxel_gain = torch.ones((1, len(self.RF_X)), dtype=torch.float32)

        print('Evaluating if simulation parameters are inputted...')
        for k in ['prf_sigma_scale_factor', 'attention_field_sigma', 'attention_field_gain',
                  'suppressive_surround_sigma_factor', 'summation_field_sigma_factor']:
            v = eval(k)
            if v is not None:
                setattr(self, k, torch.nn.Parameter(torch.tensor(v, dtype=torch.float32)))
                print('Parameter %s is set to %f' % (k, v))
            else:
                setattr(self, k, torch.nn.Parameter(torch.rand(1, dtype=torch.float32)))

    def forward(self, stim):
        """
        Predicts the estimates of BOLD activity for a given input stimulus based on the normalization model of attention.
        """
        if stim.ndim != 3:
            stim = stim.unsqueeze(1)
            print("Stimulus is unsqueezed over the third dimension")

        # PREALLOCATE
        receptive_field = torch.empty((self.gridsize, self.gridsize, self.gridsize * self.gridsize))
        suppressive_surround = torch.empty((self.gridsize * self.gridsize, self.gridsize * self.gridsize))
        summation_field = torch.empty((self.gridsize * self.gridsize, self.gridsize * self.gridsize))

        attfield = create_gaussian_field(self.X, self.Y, self.attention_ctr[0], self.attention_ctr[1],
                                         self.attention_field_sigma, None, False, True)
        attfield = self.attention_field_gain * attfield + 1

        rf_sigma = 0.07 + self.prf_sigma_scale_factor * (torch.sqrt(self.RF_X ** 2 + self.RF_Y ** 2))
        rf_supp_sigma = rf_sigma * self.suppressive_surround_sigma_factor
        rf_summ_sigma = rf_sigma * self.summation_field_sigma_factor

        for rf in range(0, self.gridsize * self.gridsize):
            receptive_field[..., rf] = create_gaussian_field(self.X, self.Y, self.RF_X[rf], self.RF_Y[rf], rf_sigma[rf],
                                                             None, True, False)
            suppressive_surround[..., rf] = create_gaussian_field(self.X, self.Y, self.RF_X[rf], self.RF_Y[rf],
                                                                  rf_supp_sigma[rf], 'euclidean_distance', True, True)
            summation_field[..., rf] = create_gaussian_field(self.X, self.Y, self.RF_X[rf], self.RF_Y[rf],
                                                             rf_summ_sigma[rf], 'euclidean_distance', True, True)


        stimulus_drive = torch.einsum('ijk,ija->ka', receptive_field, stim)
        numerator = torch.einsum('ij,i->ij', stimulus_drive, attfield)
        surroundresponse = torch.einsum('wr,rs->ws', suppressive_surround.T, numerator)

        # denominator = torch.clip(surroundresponse +  self.ROI_normalization_sigma_factor, 1e-6)
        # if surroundresponse +  self.ROI_normalization_sigma_factor < 1e-6:
        #    print("Clipping the denominator (original value: %f)" %(surroundresponse +  self.ROI_normalization_sigma_factor))

        predicted_neural_weights_lookup = numerator / (surroundresponse + self.ROI_normalization_sigma_factor)

        predicted_voxel_weigths_lookup = torch.einsum('wr,rs->ws', summation_field.T, predicted_neural_weights_lookup)

        predicted_neural_response = predicted_neural_weights_lookup[self.voxel_lookup_indices]
        predicted_voxel_response = predicted_voxel_weigths_lookup[self.voxel_lookup_indices]

        #return predicted_voxel_response
        return numerator, surroundresponse, predicted_neural_response, predicted_voxel_response
