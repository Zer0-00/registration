import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class KL:
    """KL divergence for VAE"""
    def loss(self, _, param):
        mu = param[0]
        log_var = param[1]
        dims = list(range(len(mu.shape)))
        losses = -0.5 * torch.sum(1+log_var-mu.pow(2)-log_var.exp(), dim=dims)

        return losses.mean()

class LCC:
    """local cross correlation """
    def __init__(self, win = None, ndims = 3, device = torch.device('cpu')):
        if win is None:
            win = 8
        self.ndims = ndims
        self.win = [win]*ndims
        self.rate = torch.tensor(np.prod(self.win), device=device)
        self.sum_filt = torch.ones([1,1,*self.win], device=device)

        self.dif_filt = -torch.ones([1,1,*self.win], device=device) * self.rate
        center = (int(win/2),) * ndims
        self.dif_filt[0,0,center] = 1
        self.dif_filt = self.dif_filt

        self.conv = getattr(F, 'conv%dd' % ndims)
        self.padding = math.floor(win / 2)
        self.tau = 1e-15


    def loss(self, F, M):
        difM = self.conv(M,self.dif_filt)
        difF = self.conv(F,self.dif_filt)

        difM2 = difM * difM
        difF2 = difF * difF
        difMF = difM * difF

        difMF2 = difMF * difMF
        num = self.conv(difMF2, self.sum_filt)

        difM2_sum = self.conv(difM2, self.sum_filt)
        difF2_sum = self.conv(difF2, self.sum_filt)
        den = difF2_sum * difM2_sum + self.tau

        lcc = num / den

        return -torch.mean(lcc)


