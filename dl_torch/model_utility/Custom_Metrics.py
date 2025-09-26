import torch
import numpy as np


def voxel_accuracy(output, target, ignore_index=None):
    preds = torch.argmax(output, dim=1)
    if ignore_index is not None:
        mask = target != ignore_index
        correct = (preds[mask] == target[mask]).float()
        return correct.sum() / mask.sum()

    else:
        correct = (preds == target).float()
        return correct.sum() / correct.numel()

def mesh_class_confusion_matrix(ftm_prediction, ftm_ground_truth, class_to_index):

    n_classes = len(class_to_index.keys())

    ccm = np.zeros((n_classes, n_classes), dtype=np.int32)

    ftm_ground_truth = [class_to_index[element] for element in ftm_ground_truth.values()]

    for i in range(len(ftm_prediction)):
        ccm[ftm_ground_truth[i], ftm_prediction[i]] += 1

    return ccm

def mesh_IOU(ftm_prediction, ftm_ground_truth):
    # Convert lists to tensors if necessary
    if isinstance(ftm_prediction, list):
        ftm_prediction = torch.tensor(ftm_prediction)
    if isinstance(ftm_ground_truth, list):
        ftm_ground_truth = torch.tensor(ftm_ground_truth)

    correct = (ftm_prediction == ftm_ground_truth).float()
    return  correct.sum() / correct.numel()