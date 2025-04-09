import numpy as np

def evaluate_voxel_class_kernel(grid, target_idx, k, class_weights):
    D, _, _, C = grid.shape
    x, y, z = target_idx
    half_k = k // 2

    # Define neighborhood bounds, clamped to grid dimensions
    x_min = max(x - half_k, 0)
    x_max = min(x + half_k + 1, D)
    y_min = max(y - half_k, 0)
    y_max = min(y + half_k + 1, D)
    z_min = max(z - half_k, 0)
    z_max = min(z + half_k + 1, D)

    # Extract the neighborhood (k x k x k x C)
    neighborhood = grid[x_min:x_max, y_min:y_max, z_min:z_max, :]

    # Count occurrences of each class
    class_counts = np.sum(neighborhood, axis=(0, 1, 2))

    # Apply class weights
    weighted_counts = class_counts * np.array(class_weights)

    # Return the index of the class with the highest weighted count
    return int(np.argmax(weighted_counts))

def main():
    pass

if __name__ == "__main__":
    main()
