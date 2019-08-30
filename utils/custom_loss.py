import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import pdb


class SegCapsLoss(nn.Module):
    def __init__(self):
        super(SegCapsLoss, self).__init__()

    def forward(self, output, target):
        class_loss = (target * F.relu(0.9 - output) + 0.5 * (1 - target) * F.relu(output - 0.1)).mean()
        return class_loss


class CapsNetMarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(CapsNetMarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        pdb.set_trace()

        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()

class CapsNetReconLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(CapsNetReconLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_
        self.capsnetloss = CapsNetMarginLoss(self.m_pos,self.m_neg,self.lambda_)

    def forward(self, output, probs, target,data):
        reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
        margin_loss = self.capsnetloss(probs, target)
        loss = 0.0005 * reconstruction_loss + margin_loss
        return loss



