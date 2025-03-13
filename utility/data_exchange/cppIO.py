import numpy as np

def read_3d_array_from_binary(filename: str) -> tuple[ : , : , np.ndarray]:
    with open(filename, "rb") as file:
        # Read Metadata
        voxel_size = np.fromfile(file, dtype=np.float64, count=1)[0]
        background = np.fromfile(file, dtype=np.float64, count=1)[0]

        # Read the array sizes
        dim1 = np.fromfile(file, dtype=np.uint64, count=1)[0]
        dim2 = np.fromfile(file, dtype=np.uint64, count=1)[0]
        dim3 = np.fromfile(file, dtype=np.uint64, count=1)[0]

        # Read the actual array data
        data: np.ndarray = np.fromfile(file, dtype=np.float32).reshape((dim1, dim2, dim3))

        return voxel_size, background, data