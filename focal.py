import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, num_labels, gamma=2.0, alpha=0.25, epsilon=1e-9, reduction="mean", activation_type="softmax"):

        super(FocalLoss, self).__init__()
        self.num_labels      = num_labels
        self.gamma           = gamma
        self.alpha           = alpha
        self.epsilon         = epsilon
        self.activation_type = activation_type
        self.reduction       = reduction

    def forward(self, preds, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == "softmax":
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = F.softmax(preds, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)

        elif self.activation_type == "sigmoid":
            multi_hot_key = target
            logits = F.sigmoid(preds)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()

        if self.reduction == "mean":
            loss = loss.mean()

        elif self.reduction == "sum":
            loss = loss.sum()

        elif self.reduction == "none":
            pass
        
        return loss