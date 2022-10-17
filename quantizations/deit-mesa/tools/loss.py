
import torch
import torch.nn as nn
import torch.nn.functional as F

# label smooth
class CrossEntropyLabelSmooth(nn.Module):
  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


# KD
# https://github.com/peterliht/knowledge-distillation-pytorch/blob/ef06124d67a98abcb3a5bc9c81f7d0f1f016a7ef/model/net.py#L100
def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.distill_loss_alpha
    if alpha < 1.0:
        loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    else:
        loss = 0

    if hasattr(params, 'distill_loss_type') and params.distill_loss_type == 'hard':
        distillation_loss = F.cross_entropy(outputs, teacher_outputs.argmax(dim=1))
    else:
        T = params.distill_loss_temperature
        distillation_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                                 F.softmax(teacher_outputs/T, dim=1)) * (T * T)

    loss = loss + distillation_loss * alpha
    return loss

