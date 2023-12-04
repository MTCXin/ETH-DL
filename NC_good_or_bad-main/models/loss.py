import torch.nn.functional as F
import torch
import numpy as np
from parse_config import ConfigParser
import torch.nn as nn
from torch.autograd import Variable
import math
from typing import Type, Any, Callable, Union, List, Optional
from utils import sigmoid_rampup, sigmoid_rampdown, cosine_rampup, cosine_rampdown, linear_rampup

cross_entropy = nn.CrossEntropyLoss


class cross_entropy_iden(nn.Module):
    def forward(self, output, target):
        features, output = output
        # Add a nuclear norm maximization term for the correct class (features)
        # return F.cross_entropy(output, target) - 0.01 * torch.linalg.norm(features, ord="nuc")
        identity_constraint = torch.linalg.norm(torch.eye(output.shape[0]).to(output.device) - features @ features.T)
        return F.cross_entropy(output, target) + 0.0001 * identity_constraint


class squaredLoss(nn.Module):
    def forward(self, feature, target, model):
        # changed from https://github.com/MTandHJ/roboc/blob/master/src/loss_zoo.py
        weight = model.linear.weight.data.clone()
        targets_fea = weight[target]
        return F.mse_loss(feature, targets_fea, reduction="mean")


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = beta
        

    def forward(self, index, output, label):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss +  self.config['train_loss']['args']['lambda']*elr_reg
        return  final_loss

    
class elr_plus_loss(nn.Module):
    def __init__(self, num_examp, config, device, num_classes=10, beta=0.3):
        super(elr_plus_loss, self).__init__()
        self.config = config
        self.pred_hist = (torch.zeros(num_examp, num_classes)).to(device)
        self.q = 0
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, iteration, output, y_labeled):
        y_pred = F.softmax(output,dim=1)

        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled*self.q
            y_labeled = y_labeled/(y_labeled).sum(dim=1,keepdim=True)

        ce_loss = torch.mean(-torch.sum(y_labeled * F.log_softmax(output, dim=1), dim = -1))
        reg = ((1-(self.q * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + sigmoid_rampup(iteration, self.config['coef_step'])*(self.config['train_loss']['args']['lambda']*reg)
      
        return  final_loss, y_pred.cpu().detach()

    def update_hist(self, epoch, out, index= None, mix_index = ..., mixup_l = 1):
        y_pred_ = F.softmax(out,dim=1)
        self.pred_hist[index] = self.beta * self.pred_hist[index] +  (1-self.beta) *  y_pred_/(y_pred_).sum(dim=1,keepdim=True)
        self.q = mixup_l * self.pred_hist[index]  + (1-mixup_l) * self.pred_hist[index][mix_index]
        

class focalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N, )` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(N, C, requires_grad=True)
        >>> target = torch.empty(N, dtype=torch.long).random_(C)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 num_class: Optional[int] = 10,
                 reduction: Optional[str] = 'mean', **kwargs: Any) -> None:
        super(focalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.num_class: Optional[int] = num_class
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            **kwargs: Any
    ) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 2:
            raise ValueError("Invalid input shape, we expect NxC. Got: {}"
                             .format(input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=self.num_class, device=input.device, dtype=input.dtype)
        # compute the actual focal loss
        weight = torch.pow(1. - input_soft, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss

def one_hot(
        targets: torch.Tensor,
        num_classes: int = 10,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(targets)}")

    if not targets.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {targets.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = targets.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, targets.unsqueeze(1), 1.0) + eps