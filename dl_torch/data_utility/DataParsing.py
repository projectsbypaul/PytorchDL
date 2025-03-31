import torch
from matplotlib.image import imread
from sympy import false
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import os
from typing import List, Tuple, Dict, Optional
from utility.data_exchange import cppIO
import numpy as np
import yaml

def parse_bin_array_data_split_by_class(root_dir: str, exclude_subfolder : list[str] = None) -> tuple[Tensor, Tensor, dict[int, str]]:
    """
    Parses the dataset directory structure and returns:
    - A single tensor for data (X)
    - A one-hot encoded tensor for labels (Y)
    - A Tuple of dictionaries [0] class->index [1] index->class

    Args:
        - root_dir (str):param The root directory containing class folders.
        - exclude_subfolder : list[str]:param = [""] list of subfolder name you like to exclude from parsing


    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]: A tuple containing:
            - X:return Torch tensor of shape (num_samples, feature_dim)
            - Y:return One-hot encoded tensor of shape (num_samples, 1, num_classes)
            - class_dict:return a dictionary mapping index->class
    """
    X, Y = [], []
    class_list = sorted([name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))])
    class_to_idx : Dict[str, int] = {cls: idx for idx, cls in enumerate(class_list)}  # Class name -> index
    class_dict : Dict[int, str]  = {idx: cls for cls, idx in class_to_idx.items()}  # Index -> class name (reverse mapping)

    for class_name, class_idx in class_to_idx.items():
        class_path = os.path.join(root_dir, class_name)

        for subdir, _, files in os.walk(class_path):

            # Extract the immediate subfolder name relative to class_path
            relative_subfolder = os.path.relpath(subdir, class_path)

            # Check if exclude_subfolder is set and if the current subdir should be skipped
            if exclude_subfolder and relative_subfolder in exclude_subfolder:
                continue

            for file_name in files:
                    file_path = os.path.join(subdir, file_name)

                    if not os.path.isfile(file_path):
                        continue

                    try:
                        _, _, data = cppIO.read_3d_array_from_binary(file_path)
                        X.append(data)

                        # Create one-hot encoded label tensor
                        label_tensor = torch.zeros(1, len(class_list))
                        label_tensor[0, class_idx] = 1
                        Y.append(label_tensor)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")




    # Convert list of numpy arrays to a single numpy array before creating a tensor
    x = np.array(X, dtype=np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.stack(Y)

    return x_tensor, y_tensor, class_dict

def parse_yaml(file_name : str):
    # Load the YAML file

    with open(file_name, 'r') as file:
        data = yaml.safe_load(file)

    # Print the parsed data
    return data

def parse_obj(file_name :  str):
    """
    Parses an obj for faces and vertices.

    Args:
        - file_name:param : str - path to file location
    Returns:
        Parsed obj data
          - vertices:return - parsed vertices
          - faces:return -  parsed faces
    """

    vertices = []
    faces = []

    with open(file_name, 'r') as file:

        for line in file:

            if line.startswith('v '):

                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)

            elif line.startswith('f '):

                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)

    return vertices, faces

def main() -> None:
  return

if __name__ == "__main__":
    main()
