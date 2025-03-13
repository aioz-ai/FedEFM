import torch
from collections import OrderedDict
# from torchmetrics.functional.classification import dice as DiceMetric
from torch import nn, optim
import numpy as np

from medpy import metric

def binary_accuracy(preds, y):
    """

    :param preds:
    :param y:
    :return:
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def accuracy(preds, y):
    """

    :param preds:
    :param y:
    :return:
    """
    # print("PRED shape", predicted.shape)
    # print(y.shape)
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc

def calculate_metric_percase(pred, gt): 
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0
def dice(pred, gt): 
    pred = torch.from_numpy(pred).cpu()
    gt = gt.cpu()
    return 0
    # l = DiceMetric(pred,gt,num_classes=3,average = 'micro', ignore_index=0)
    # return l 
    # classes = 3
    # metric_list = []
    # for i in range(1, classes):
    #     pred_class = (pred==i).astype(int)
    #     label_class = (gt==i).astype(int)
    #     metric_list.append(calculate_metric_percase(pred_class, label_class))
    # return np.mean(metric_list)