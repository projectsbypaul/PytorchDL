import torch


def voxel_accuracy(output, target, ignore_index=None):
    preds = torch.argmax(output, dim=1)
    if ignore_index is not None:
        mask = target != ignore_index
        correct = (preds[mask] == target[mask]).float()
        return correct.sum() / mask.sum()
    else:
        correct = (preds == target).float()
        return correct.sum() / correct.numel()


def mesh_IOU(ftm_prediction, ftm_ground_truth):
    # Convert lists to tensors if necessary
    if isinstance(ftm_prediction, list):
        ftm_prediction = torch.tensor(ftm_prediction)
    if isinstance(ftm_ground_truth, list):
        ftm_ground_truth = torch.tensor(ftm_ground_truth)

    correct = (ftm_prediction == ftm_ground_truth).float()
    return  correct.sum() / correct.numel()