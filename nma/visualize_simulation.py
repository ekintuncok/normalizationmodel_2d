import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functions import create_gaussian_field
import matplotlib.patches as mpatches
import matplotlib.pylab as pl


def visualize_simulated_neural_image(stimulus_drive, numerator, surroundresponse, neuralresponse, populationresponse,
                                     stimidx,
                                     gridsize=64):
    stimulus_drive_toplot = np.reshape(stimulus_drive[:, stimidx], (gridsize, gridsize))
    numerator_toplot = np.reshape(numerator[:, stimidx], (gridsize, gridsize))
    surroundresponse_toplot = np.reshape(surroundresponse[:, stimidx], (gridsize, gridsize))
    neuralresponse_toplot = np.reshape(neuralresponse[:, stimidx], (gridsize, gridsize))
    populationresponse_toplot = np.reshape(populationresponse[:, stimidx], (gridsize, gridsize))

    colorm = 'Reds'
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=[22, 22])
    im1 = ax1.imshow(stimulus_drive_toplot, cmap=colorm)
    ax1.set_title('Stimulus drive')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)

    im2 = ax2.imshow(numerator_toplot, cmap=colorm)
    ax2.set_title('Stimulus drive under attention')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)

    im3 = ax3.imshow(surroundresponse_toplot, cmap=colorm)
    ax3.set_title('Suppressive drive under attention')
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="10%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)

    im4 = ax4.imshow(neuralresponse_toplot, cmap=colorm)
    ax4.set_title('Normalized population response before summation')
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("right", size="10%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)

    im5 = ax5.imshow(populationresponse_toplot, cmap=colorm)
    ax5.set_title('Normalized population response after summation')
    divider = make_axes_locatable(ax5)
    cax5 = divider.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(im5, cax=cax5)

    plt.show()
    fig.savefig('simulated_neural_image_bar_%i' % stimidx + '.tiff')


def visualize_bar_activity_around_af(x_grid, y_grid, prf_x, prf_y, attention_ctr, attention_field_sigma, attention_field_gain,
                                     voxel_distance_from_af, predicted_neural_response, predicted_voxel_response, gridsize=64):

    # plot AF and the voxel 'slice'
    attention_field = create_gaussian_field(x_grid, y_grid, attention_ctr[0], attention_ctr[1], attention_field_sigma,
                                            normalize=False,
                                            flat=False)
    ####
    if not isinstance(voxel_distance_from_af, np.ndarray):
        voxel_distance_from_af = voxel_distance_from_af.detach().numpy()

    n = gridsize
    voxel_indices_x_slice = np.argwhere(prf_y == attention_ctr[1])
    voxel_indices_y_slice = np.argwhere(prf_x == attention_ctr[0])

    #####
    dist_of_plotted_voxels = np.unique(voxel_distance_from_af[voxel_indices_x_slice])
    dist_of_plotted_voxels = dist_of_plotted_voxels[1:len(dist_of_plotted_voxels)]

    arr = np.mod(voxel_indices_x_slice, gridsize)
    vxl_anchor = voxel_indices_x_slice[arr[0]]

    colors = pl.cm.magma(np.linspace(0, 1, len(dist_of_plotted_voxels)))
    assign_col = np.zeros((len(dist_of_plotted_voxels), colors.shape[1] + 1))
    assign_col[:, 0] = dist_of_plotted_voxels
    assign_col[:, 1:5] = colors

    fig = plt.figure(figsize=[20, 5])
    plt.suptitle('Spatially pooled responses')
    for i, gain_idx in enumerate(attention_field_gain):
        plt.subplot(2, attention_field_gain.shape[0] + 1, 1)
        fig = plt.imshow(attention_field)
        plt.title('Attention field')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        rect = mpatches.Rectangle((0, int(vxl_anchor / gridsize)), gridsize, 1, alpha=0.7, facecolor="red")
        plt.gca().add_patch(rect)
        plt.subplot(2, attention_field_gain.shape[0] + 1, i + 2)
        plt.title('Attention gain = %i' % attention_field_gain[i])
        upper_limit = np.max(predicted_voxel_response[-1, voxel_indices_x_slice, :])
        for idx in np.arange(0, n):
            color_idx = np.argwhere(assign_col[:, 0] == voxel_distance_from_af[voxel_indices_x_slice[idx]])
            if voxel_distance_from_af[voxel_indices_x_slice[idx]] != 0:
                plt.plot(np.transpose(predicted_voxel_response[i, voxel_indices_x_slice[idx], :]),
                         color=colors[color_idx[0]])
            else:
                plt.plot(np.transpose(predicted_voxel_response[i, voxel_indices_x_slice[idx], :]), '--',
                         linewidth=2, color="red")
            plt.ylim(0, upper_limit)
            hori_center = np.argmax(predicted_voxel_response[i, vxl_anchor, 1:24])
        vert_center = np.argmax(predicted_voxel_response[i, vxl_anchor, 25:48])
        rect1 = mpatches.Rectangle((hori_center, 0), 1, upper_limit, alpha=0.7, facecolor="black")
        rect2 = mpatches.Rectangle((vert_center + 24, 0), 1, upper_limit, alpha=0.7, facecolor="black")

        plt.gca().add_patch(rect1)
        plt.gca().add_patch(rect2)
    plt.tight_layout()

    dist_of_plotted_voxels = np.unique(voxel_distance_from_af[voxel_indices_y_slice])
    dist_of_plotted_voxels = dist_of_plotted_voxels[1:len(dist_of_plotted_voxels)]
    arr = np.mod(voxel_indices_y_slice, gridsize)
    vxl_anchor = voxel_indices_y_slice[arr[0]]

    colors = pl.cm.magma(np.linspace(0, 1, len(dist_of_plotted_voxels)))
    assign_col = np.zeros((len(dist_of_plotted_voxels), 5))
    assign_col[:, 0] = dist_of_plotted_voxels
    assign_col[:, 1:5] = colors

    for i, gain_idx in enumerate(attention_field_gain):
        plt.subplot(2, attention_field_gain.shape[0] + 1, attention_field_gain.shape[0] + 2)
        fig = plt.imshow(attention_field)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        rect = mpatches.Rectangle((int(vxl_anchor / gridsize), 0), 1, gridsize, alpha=0.7, facecolor="red")
        plt.gca().add_patch(rect)
        plt.subplot(2, attention_field_gain.shape[0] + 1, i + attention_field_gain.shape[0] + 3)
        upper_limit = np.max(predicted_voxel_response[-1, voxel_indices_y_slice, :])
        lower_limit = np.min(predicted_voxel_response[-1, voxel_indices_y_slice, :])

        for idx in np.arange(0, n):
            color_idx = np.argwhere(assign_col[:, 0] == voxel_distance_from_af[voxel_indices_y_slice[idx]])
            if voxel_distance_from_af[voxel_indices_y_slice[idx]] != 0:
                plt.plot(np.transpose(predicted_voxel_response[i, voxel_indices_y_slice[idx], :]),
                         color=colors[color_idx[0]])
            else:
                plt.plot(np.transpose(predicted_voxel_response[i, voxel_indices_y_slice[idx], :]), '--',
                         linewidth=2, color="red")
            plt.ylim(lower_limit, upper_limit)
            hori_center = np.argmax(predicted_voxel_response[i, vxl_anchor, 1:24])
        vert_center = np.argmax(predicted_voxel_response[i, vxl_anchor, 25:48])
        rect1 = mpatches.Rectangle((hori_center, 0), 1, upper_limit, alpha=0.7, facecolor="black")
        rect2 = mpatches.Rectangle((vert_center + 24, 0), 1, upper_limit, alpha=0.7, facecolor="black")
        plt.gca().add_patch(rect1)
        plt.gca().add_patch(rect2)
    plt.tight_layout()
    plt.show()

    dist_of_plotted_voxels = np.unique(voxel_distance_from_af[voxel_indices_x_slice])
    dist_of_plotted_voxels = dist_of_plotted_voxels[1:len(dist_of_plotted_voxels)]

    arr = np.mod(voxel_indices_x_slice, gridsize)
    vxl_anchor = voxel_indices_x_slice[arr[0]]
    np.set_printoptions(suppress=True, precision=16)

    colors = pl.cm.magma(np.linspace(0, 1, len(dist_of_plotted_voxels)))
    assign_col = np.zeros((len(dist_of_plotted_voxels), colors.shape[1] + 1))
    assign_col[:, 0] = dist_of_plotted_voxels
    assign_col[:, 1:5] = colors

    fig = plt.figure(figsize=[20, 5])
    plt.suptitle('Normalized neural responses')
    for i, gain_idx in enumerate(attention_field_gain):
        plt.subplot(2, attention_field_gain.shape[0] + 1, 1)
        fig = plt.imshow(attention_field)
        plt.title('Attention field')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        rect = mpatches.Rectangle((0, int(vxl_anchor / gridsize)), gridsize, 1, alpha=0.7, facecolor="red")
        plt.gca().add_patch(rect)
        plt.subplot(2, attention_field_gain.shape[0] + 1, i + 2)
        plt.title('Attention gain = %i' % attention_field_gain[i])
        upper_limit = np.max(predicted_neural_response[-1, voxel_indices_x_slice, :])
        lower_limit = np.min(predicted_neural_response[-1, voxel_indices_x_slice, :])

        for idx in np.arange(0, n):
            color_idx = np.argwhere(assign_col[:, 0] == voxel_distance_from_af[voxel_indices_x_slice[idx]])
            if voxel_distance_from_af[voxel_indices_x_slice[idx]] != 0:
                plt.plot(np.transpose(predicted_neural_response[i, voxel_indices_x_slice[idx], :]),
                         color=colors[color_idx[0]])
            else:
                plt.plot(np.transpose(predicted_neural_response[i, voxel_indices_x_slice[idx], :]), '--',
                         linewidth=2, color="red")
            plt.ylim(lower_limit, upper_limit)
            hori_center = np.argmax(predicted_neural_response[i, vxl_anchor, 1:24])
        vert_center = np.argmax(predicted_neural_response[i, vxl_anchor, 25:48])
        rect1 = mpatches.Rectangle((hori_center, 0), 1, upper_limit, alpha=0.7, facecolor="black")
        rect2 = mpatches.Rectangle((vert_center + 24, 0), 1, upper_limit, alpha=0.7, facecolor="black")

        plt.gca().add_patch(rect1)
        plt.gca().add_patch(rect2)
    plt.tight_layout()

    dist_of_plotted_voxels = np.unique(voxel_distance_from_af[voxel_indices_y_slice])
    dist_of_plotted_voxels = dist_of_plotted_voxels[1:len(dist_of_plotted_voxels)]
    arr = np.mod(voxel_indices_y_slice, gridsize)
    vxl_anchor = voxel_indices_y_slice[arr[0]]

    colors = pl.cm.magma(np.linspace(0, 1, len(dist_of_plotted_voxels)))
    assign_col = np.zeros((len(dist_of_plotted_voxels), 5))
    assign_col[:, 0] = dist_of_plotted_voxels
    assign_col[:, 1:5] = colors

    for i, gain_idx in enumerate(attention_field_gain):
        plt.subplot(2, attention_field_gain.shape[0] + 1, attention_field_gain.shape[0] + 2)
        fig = plt.imshow(attention_field)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        rect = mpatches.Rectangle((int(vxl_anchor / gridsize), 0), 1, gridsize, alpha=0.7, facecolor="red")
        plt.gca().add_patch(rect)
        plt.subplot(2, attention_field_gain.shape[0] + 1, i + attention_field_gain.shape[0] + 3)
        upper_limit = np.max(predicted_neural_response[-1, voxel_indices_y_slice, :])
        for idx in np.arange(0, n):
            color_idx = np.argwhere(assign_col[:, 0] == voxel_distance_from_af[voxel_indices_y_slice[idx]])
            if voxel_distance_from_af[voxel_indices_y_slice[idx]] != 0:
                plt.plot(np.transpose(predicted_neural_response[i, voxel_indices_y_slice[idx], :]),
                         color=colors[color_idx[0]])
            else:
                plt.plot(np.transpose(predicted_neural_response[i, voxel_indices_y_slice[idx], :]), '--',
                         linewidth=2, color="red")
            plt.ylim(0, upper_limit)
            hori_center = np.argmax(predicted_neural_response[i, vxl_anchor, 1:24])
        vert_center = np.argmax(predicted_neural_response[i, vxl_anchor, 25:48])
        rect1 = mpatches.Rectangle((hori_center, 0), 1, upper_limit, alpha=0.7, facecolor="black")
        rect2 = mpatches.Rectangle((vert_center + 24, 0), 1, upper_limit, alpha=0.7, facecolor="black")
        plt.gca().add_patch(rect1)
        plt.gca().add_patch(rect2)
    plt.tight_layout()
    plt.show()
