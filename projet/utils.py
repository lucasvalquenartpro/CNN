import torch
from torch import nn, mean


def get_loss():
    return nn.SmoothL1Loss()

def predictions_to_labels(predictions):
    return torch.where(predictions >= 0, 1, -1)

def compute_fairness_score(acc_male, acc_female):
    return (acc_male+acc_female)/2 - 2*abs(acc_male - acc_female)

def metrics(predictions, labels, genres):
    pred_label = predictions_to_labels(predictions)
    correct = (pred_label == labels).float()
    acc = correct.sum() / len(correct)

    if (genres == 1).sum() > 0:
        acc_male = correct[genres == 1].float().mean().item()
    else:
        acc_male = 0.0

    if (genres == -1).sum() > 0:
        acc_female = correct[genres == -1].float().mean().item()
    else:
        acc_female = 0.0

    fairness_score = compute_fairness_score(acc_male, acc_female)

    return {
        'acc_global': acc,
        'acc_male': acc_male,
        'acc_female': acc_female,
        'fairness_score': fairness_score
    }

class Metrics:
    def __init__(self):
        super().__init__()
        self.preds = []
        self.labels = []
        self.genres = []

    def update(self, predictions, labels, genres):
        self.preds.append(predictions)
        self.labels.append(labels)
        self.genres.append(genres)

    def compute(self):
        if len(self.preds) == 0 or len(self.labels) == 0:
            return {
                'acc_global': 0,
                'acc_male': 0,
                'acc_female': 0,
                'fairness_score': 0
            }

        pred_labels = torch.cat(self.preds)
        labels = torch.cat(self.labels)
        genres = torch.cat(self.genres)

        metric = metrics(pred_labels, labels, genres)
        return metric