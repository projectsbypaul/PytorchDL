import numpy as np
import struct

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


def read_type_map_from_binary(filename : str):
    data = []
    with open(filename, "rb") as f:
        while True:
            row = []
            row_size_bytes = f.read(4)
            if not row_size_bytes:
                break  # EOF
            (row_size,) = struct.unpack('I', row_size_bytes)
            for _ in range(row_size):
                str_len_bytes = f.read(4)
                if not str_len_bytes:
                    raise ValueError("Unexpected EOF when reading string length")
                (str_len,) = struct.unpack('I', str_len_bytes)
                str_data = f.read(str_len)
                if len(str_data) != str_len:
                    raise ValueError("Unexpected EOF when reading string content")
                row.append(str_data.decode('utf-8'))
            data.append(row)
    return data

def read_float_matrix(filename):
    matrix = []
    with open(filename, "rb") as f:
        while True:
            row_size_bytes = f.read(4)
            if not row_size_bytes:
                break  # EOF
            (row_size,) = struct.unpack("I", row_size_bytes)
            float_data = f.read(row_size * 4)
            row = list(struct.unpack(f"{row_size}f", float_data))
            matrix.append(row)

    return matrix

def read_type_counts_from_binary(filename):
    surface_type_counts = {}

    with open(filename, "rb") as f:
        map_size = int.from_bytes(f.read(8), byteorder='little')

        for _ in range(map_size):
            key_size = int.from_bytes(f.read(8), byteorder='little')
            key = f.read(key_size).decode('utf-8')
            value = int.from_bytes(f.read(4), byteorder='little', signed=True)
            surface_type_counts[key] = value

    return surface_type_counts
