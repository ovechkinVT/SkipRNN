import torch
from torch import nn


class hinton_binarize(torch.autograd.Function):
    """
    Binarize function from the paper
    'SKIP RNN: LEARNING TO SKIP STATE UPDATES IN RECURRENT NEURAL NETWORKS'
    https://openreview.net/forum?id=HkwVAXyCW
    Works as round function but has a unit gradient:
    Binarize(x) := (x > 0.5).float()
    d Binarize(x) / dx := 1
    """

    @staticmethod
    def forward(ctx, x, threshold=0.5):
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class HintonBinarizer(nn.Module):
    """
    Binarize function from the paper
    'SKIP RNN: LEARNING TO SKIP STATE UPDATES IN RECURRENT NEURAL NETWORKS'
    https://openreview.net/forum?id=HkwVAXyCW
    Works as round function but has a unit gradient:
    Binarize(x) := (x > 0.5).float()
    d Binarize(x) / dx := 1
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold=threshold

    def forward(self, x, threshold=None):
        threshold = threshold if threshold is not None else self.threshold
        return hinton_binarize.apply(x, threshold)


def kuma_reparametrization(a, b):
    u = torch.rand_like(a)
    k = (1 - (1 - u) ** (1 / (b+1e-8))) ** (1 / (a+1e-8))
    return k

class Rectifier(nn.Module):
    def __init__(self, l=-0.1, r=1.1):
        super().__init__()
        self.l = l
        self.r = r
        self.eps = 1e-7

    def forward(self, x, l=None, r=None):
        l = l if l is not None else self.l
        r = r if r is not None else self.r

        t = l + (r - l)*x # extension
        t = torch.nn.functional.hardtanh(t, 0, 1) # truncation
        return t

class HardKumaBinarizer(nn.Module):
    def __init__(self, l=-0.1, r=1.1):
        super().__init__()
        self.rectifier = Rectifier(l, r)

    def forward(self, a, b, l=None, r=None):
        k = kuma_reparametrization(torch.exp(a), torch.exp(b))
        t = self.rectifier(k, l, r)
        return t


def concrete_binarize(x, t=0.1, u=None):
    """
    Binarize function from the paper
    'DropMax: Adaptive Variational Softmax'
    http://papers.nips.cc/paper/7371-dropmax-adaptive-variational-softmax
    Args:
        x (torch.FloatTensor): logits to binarize
        t (float): temperature
        u (torch.FloatTensor): uniformly distributed noise to mix-up with logits
    """
    u = torch.rand(*x.shape).to(x.device) if u is None else u
    eps = 1e-7
    return torch.sigmoid(1/t * (x + torch.log(u/(1-u) + eps)))

class ConcreteBinarizer(nn.Module):
    def __init__(self, t=0.1):
        """
        class-wrapper for binarize function from the paper
        'DropMax: Adaptive Variational Softmax'
        http://papers.nips.cc/paper/7371-dropmax-adaptive-variational-softmax
        accepts logit as input
        Args:
            t (float): temperature for the concrete distrivution
        """
        super().__init__()
        self.t = t

    def forward(self, x, u=None, t=None):
        t = t if t is not None else self.t
        return concrete_binarize(x, t, u)
