from utility.data_exchange import cppIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def read_array_test():
    # Load the array
    voxel_size, background, array = cppIO.read_3d_array_from_binary(r"C:\Local_Data\DL_Datasets\ModelNet10_SDF_32\bathtub\test\bathtub_0132 .bin")

    print(f"Voxel Size: {voxel_size}")
    print(f"Background: {background}")
    # print(array)

    # Define a custom colormap
    colors = ["blue", "white", "red"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot the array
    plt.imshow(array[15, :, :], cmap="seismic")
    plt.colorbar()  # Add a colorbar to show value mapping
    plt.title("2D NumPy Array Visualization")
    plt.show()


def main() -> None:
   pass

if __name__ == "__main__":
    main()

