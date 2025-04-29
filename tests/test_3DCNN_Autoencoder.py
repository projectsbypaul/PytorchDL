from dl_torch.models.CNN3D_Autoencoder import Autoencoder, Autoencoder_binary
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
import torch
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F


def test_routine():
    mydata = InteractiveDataset.load_dataset("../data/datasets/ModelNet10/ModelNet10_AE_SDF_32_bin_test.torch")

    model = Autoencoder_binary()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('../data/model_weights/CNN3D_Autoencoder_bin/CNN3D_Autoencoder_bin_epoch_900.pth')
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval() # set model to evaluation model

    sample = mydata.data[351].unsqueeze(1)

    output = model(sample.to(device))

    sample = sample.detach().view(32,32,32).cpu().numpy()

    output = output.detach().view(32,32,32).cpu().numpy()


    """
    a1 = sample[plane_index, :, :]
    a2 = sample[:, plane_index, :]
    a3 = sample[:, :, plane_index]

    # Create a figure with 2 rows and 3 columns of subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Display arrays in the desired order:
    # Top row: array 1, array 3, array 5
    axes[0].imshow(a1, cmap='viridis')
    axes[0].set_title("Array 1")
    axes[1].imshow(a2, cmap='viridis')
    axes[1].set_title("Array 3")
    axes[2].imshow(a3, cmap='viridis')
    axes[2].set_title("Array 5")

    # Remove axes for a cleaner display
    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    """
    grid_01 = pv.ImageData()
    grid_01.dimensions = np.array(sample.shape) + 1
    grid_01.origin = (0, 0, 0)
    grid_01.spacing = (1, 1, 1)

    # Add the numpy array as cell data (flattened in Fortran order)
    grid_01["values"] = sample.flatten(order="F")

    # Use a threshold filter to select cells with value equal to 1.
    # For floating point data, we use a small tolerance.
    thresholded_01 = grid_01.threshold(1, scalars="values")

    grid_02 = pv.ImageData()
    grid_02.dimensions = np.array(output.shape) + 1
    grid_02.origin = (0, 40, 0)
    grid_02.spacing = (1, 1, 1)

    # Add the numpy array as cell data (flattened in Fortran order)
    grid_02["values"] = output.flatten(order="F")

    # Use a threshold filter to select cells with value equal to 1.
    # For floating point data, we use a small tolerance.
    thresholded_02 = grid_02.threshold([0, 20] , scalars="values")


    # Visualize the selected voxels
    plotter = pv.Plotter()
    plotter.add_mesh(thresholded_01, show_edges=True, color="red", opacity=1)
    plotter.add_mesh(thresholded_02, show_edges=True, color="red", opacity=1)
    plotter.add_axes()
    plotter.show()



    return

def main() -> None:
    test_routine()
    return

if __name__ == "__main__":
    main()
