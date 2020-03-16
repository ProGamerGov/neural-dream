import torch
import torch.nn as nn
import neural_dream.dream_utils as dream_utils
from neural_dream.dream_experimental import ChannelMod



# Define an nn Module to compute DeepDream loss in different ways
class DreamLossMode(torch.nn.Module):

    def __init__(self, loss_mode, use_fft, r):
        super(DreamLossMode, self).__init__()
        self.get_mode(loss_mode)
        self.use_fft = use_fft[0]
        self.fft_tensor = dream_utils.FFTTensor(r, use_fft[1])

    def get_mode(self, loss_mode):
        self.loss_mode_string = loss_mode
        if loss_mode.lower() == 'norm':
           self.get_loss = self.norm_loss
        elif loss_mode.lower() == 'mean':
           self.get_loss = self.mean_loss
        elif loss_mode.lower() == 'l2':
           self.get_loss = self.l2_loss
        elif loss_mode.lower() == 'mse':
           self.crit = torch.nn.MSELoss()
           self.get_loss = self.crit_loss
        elif loss_mode.lower() == 'bce':
           self.crit = torch.nn.BCEWithLogitsLoss()
           self.get_loss = self.crit_loss
        elif loss_mode.lower() == 'abs_mean':
           self.get_loss = self.abs_mean
        elif loss_mode.lower() == 'abs_l2':
           self.get_loss = self.abs_l2_loss

    def norm_loss(self, input):
        return input.norm()

    def mean_loss(self, input):
        return input.mean()

    def l2_loss(self, input):
        return input.pow(2).sum().sqrt()

    def abs_mean(self, input):
        return input.abs().mean()

    def abs_l2_loss(self, input):
        return input.abs().pow(2).sum().sqrt()

    def crit_loss(self, input, target):
        return self.crit(input, target)

    def forward(self, input):
        if self.use_fft:
           input = self.fft_tensor(input)
        if self.loss_mode_string != 'bce' and self.loss_mode_string != 'mse':
            loss = self.get_loss(input)
        else:
            target = torch.zeros_like(input.detach())
            loss = self.crit_loss(input, target)
        return loss


# Define an nn Module for DeepDream
class DeepDream(torch.nn.Module):

    def __init__(self, loss_mode, channels='-1', channel_mode='strong', \
    channel_capture='once', scale=4, sigma=1, use_fft=(True, 25), r=1, p_mode='fast', norm_p=0, abs_p=0, mean_p=0):
        super(DeepDream, self).__init__()
        self.get_loss = DreamLossMode(loss_mode, use_fft, r)
        self.channels = [int(c) for c in channels.split(',')]
        self.channel_mode = channel_mode
        self.get_channels = dream_utils.RankChannels(self.channels, self.channel_mode)
        self.lap_scale = scale
        self.sigma = sigma.split(',')
        self.channel_capture = channel_capture

        self.zero_weak = ChannelMod(p_mode, self.channels[0], norm_p, abs_p, mean_p)


    def capture(self, input, lp=True):
        if -1 not in self.channels and 'all' not in self.channel_mode:
            self.channels = self.get_channels(input)
        elif self.channel_mode == 'all' and -1 not in self.channels:
            self.channels = self.channels
        if self.lap_scale > 0 and lp == True:
            self.lap_pyramid = dream_utils.LaplacianPyramid(input.clone(), self.lap_scale, self.sigma)

    def get_channel_loss(self, input):
        loss = 0
        if 'once' not in self.channel_capture:
            self.capture(input, False)
        for c in self.channels:
            if input.dim() > 0:
                if int(c) < input.size(1):
                    loss += self.get_loss(input[:, int(c)])
                else:
                    loss += self.get_loss(input)
        return loss

    def forward(self, input):
        if self.lap_scale > 0:
            input = self.lap_pyramid(input)
        if self.zero_weak.enabled:
           input = self.zero_weak(input)
        if -1 in self.channels:
            loss = self.get_loss(input)
        else:
            loss = self.get_channel_loss(input)
        return loss



# Define an nn Module to collect DeepDream loss
class DreamLoss(torch.nn.Module):

    def __init__(self, loss_mode, strength, channels, channel_mode='all', **kwargs):
        super(DreamLoss, self).__init__()
        self.dream = DeepDream(loss_mode, channels, channel_mode, **kwargs)
        self.strength = strength
        self.mode = 'None'

    def forward(self, input):
        if self.mode == 'loss':
            self.loss = self.dream(input.clone()) * self.strength
        elif self.mode == 'capture':
            self.target_size = input.size()
            self.dream.capture(input.clone())
        return input


# Define a forward hook to collect DeepDream loss
class DreamLossHook(DreamLoss):

    def forward(self, module, input, output):
        if self.mode == 'loss':
            self.loss = self.dream(output.clone()) * self.strength
        elif self.mode == 'capture':
            self.target_size = output.size()
            self.dream.capture(output.clone())


# Define a pre forward hook to collect DeepDream loss
class DreamLossPreHook(DreamLoss):

    def forward(self, module, output):
        if self.mode == 'loss':
            self.loss = self.dream(output[0].clone()) * self.strength
        elif self.mode == 'capture':
            self.target_size = output[0].size()
            self.dream.capture(output[0].clone())


# Define an nn Module to compute l2 loss
class L2Regularizer(nn.Module):

    def __init__(self, strength):
        super(L2Regularizer, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.loss = self.strength * (input.clone().norm(3)/2)
        return input


# Define an nn Module to compute tv loss
class TVLoss(nn.Module):

    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:,:,1:,:] - input[:,:,:-1,:]
        self.y_diff = input[:,:,:,1:] - input[:,:,:,:-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input