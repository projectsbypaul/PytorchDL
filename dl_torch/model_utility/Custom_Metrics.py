import torch

def voxel_accuracy(output, target):
    preds = torch.argmax(output, dim=1)
    correct = (preds == target).float()
    return correct.sum() / correct.numel()