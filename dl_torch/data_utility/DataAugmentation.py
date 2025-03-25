import torch

def random_flip_nd(tensor, p):
    """
    Randomly flips an N-dimensional tensor along each axis independently with probability p.

    Args:
        tensor (torch.Tensor): Input tensor of any shape.
        p (float): Probability of flipping along each axis.

    Returns:
        torch.Tensor: Flipped tensor.
    """
    dims = list(range(tensor.ndim))  # Get all dimensions
    for dim in dims:
        if torch.rand(1).item() < p:
            tensor = torch.flip(tensor, dims=[dim])  # Flip along the selected axis
    return tensor

def random_flip_dataset(dataset, p = 0.5):
    """
        Randomly flips an N-dimensional tensor along each axis independently with probability p.

        Args:
            dataset (torch.Tensor): Input tensor of any shape.
            p (float): Probability of flipping along each axis.

        Returns:
            torch.Tensor: Flipped dataset.
        """

    flipped_dataset = torch.stack([random_flip_nd(sample, p) for sample in dataset])
    return flipped_dataset

def main() -> None:
  return

if __name__ == "__main__":
    main()
