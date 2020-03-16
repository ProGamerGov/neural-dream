import torch
import torch.nn as nn


# Define a function to partially zero inputs based on channel strength
def percentile_zero(input, p=99.98, mode='norm'):
    if 'norm' in mode:
        px = input.norm(1)
    elif 'sum' in mode:
        px = input.sum(1)
    elif 'mean' in mode:
        px = input.sum(1)
    if 'abs' in mode:
        px, tp = dream_utils.tensor_percentile(abs(px), p), abs(px)
    else:
        tp = dream_utils.tensor_percentile(px, p)

    th = (0.01*tp)

    if 'abs' in mode:
        input[abs(input) < abs(th)] = 0
    else:
        input[input < th] = 0
    return input


# Define an nn Module to mask inputs based on channel strength
class ChannelMask(torch.nn.Module):

    def __init__(self, mode, channels=-1, rank_mode='norm', channel_percent=-1):
        super(ChannelMask, self).__init__()
        self.mode = mode
        self.channels = channels
        self.rank_mode = rank_mode
        self.channel_percent = channel_percent

    def list_channels(self, input):
        if input.is_cuda:
            channel_list = torch.zeros(input.size(1), device=input.get_device())
        else:
            channel_list = torch.zeros(input.size(1))
        for i in range(input.size(1)):
            y = input.clone().narrow(1,i,1)
            if self.rank_mode == 'norm':
                y = torch.norm(y)
            elif self.rank_mode == 'sum':
                y = torch.sum(y)
            elif self.rank_mode == 'mean':
                y = torch.mean(y)
            elif self.rank_mode == 'norm-abs':
                y = torch.norm(torch.abs(y))
            elif self.rank_mode == 'sum-abs':
                y = torch.sum(torch.abs(y))
            elif self.rank_mode == 'mean-abs':
                y = torch.mean(torch.abs(y))
            channel_list[i] = y
        return channel_list

    def channel_strengths(self, input, num_channels):
        channel_list = self.list_channels(input)
        channels, idx = torch.sort(channel_list, 0, True)
        selected_channels = []
        for i in range(num_channels):
            if i < input.size(1):
                selected_channels.append(idx[i])
        return selected_channels

    def mask_threshold(self, input, channels, ft=1, fm=0.2, fw=5):
        t = torch.ones_like(input.squeeze(0)) * ft
        m = torch.ones_like(input.squeeze(0)) * fm
        for c in channels:
            m[c] = fw
        return (t * m).unsqueeze(0)

    def average_mask(self, input, channels):
        mask = torch.ones_like(input.squeeze(0))
        avg = torch.sum(channels)/input.size(1)
        for i in range(channels.size(1)):
            w = avg/channels[i]
            mask[i] = w
        return mask.unsqueeze(0)

    def average_tensor(self, input):
        channel_list = self.list_channels(input)
        mask = self.average_mask(input, channel_list)
        self.mask = mask

    def weak_channels(self, input):
        channel_list = self.channel_strengths(input, self.channels)
        mask = self.mask_threshold(input, channel_list, ft=1, fm=2, fw=0.2)
        self.mask = mask

    def strong_channels(self, input):
        channel_list = self.channel_strengths(input, self.channels)
        mask = self.mask_threshold(input, channel_list, ft=1, fm=0.2, fw=5)
        self.mask = mask

    def zero_weak(self, input):
        channel_list = self.channel_strengths(input, self.channels)
        mask = self.mask_threshold(input, channel_list, ft=1, fm=0, fw=1)
        self.mask = mask

    def mask_input(self, input):
        if self.channel_percent > 0:
           channels = int((float(self.channel_percent)/100) * float(input.size(1)))
           if channels < input.size(1) and channels > 0:
               self.channels = channels
           else:
               self.channels = input.size(1)
        if self.mode == 'weak':
            input = self.weak_channels(input)
        elif self.mode == 'strong':
            input = self.strong_channels(input)
        elif self.mode == 'average':
            input = self.average_tensor(input)
        elif self.mode == 'zero_weak':
            input = self.zero_weak(input)

    def capture(self, input):
        self.mask_input(input)

    def forward(self, input):
        return self.mask * input


# Define a function to partially zero inputs based on channel strength
class ChannelMod(torch.nn.Module):

    def __init__(self, p_mode='fast', channels=0, norm_p=0, abs_p=0, mean_p=0):
        super(ChannelMod, self).__init__()
        self.p_mode = p_mode
        self.channels = channels
        self.norm_p = norm_p
        self.abs_p = abs_p
        self.mean_p = mean_p
        self.enabled = False
        if self.norm_p > 0 and self.p_mode == 'slow':
            self.zero_weak_norm = ChannelMask('zero_weak', self.channels, 'norm', channel_percent=self.norm_p)
        if self.abs_p > 0 and self.p_mode == 'slow':
            self.zero_weak_abs = ChannelMask('zero_weak', self.channels, 'sum', channel_percent=self.abs_p)
        if self.mean_p > 0 and self.p_mode == 'slow':
            self.zero_weak_mean = ChannelMask('zero_weak', self.channels, 'mean', channel_percent=self.mean_p)

        if self.norm_p > 0 or self.abs_p > 0  or self.mean_p > 0:
            self.enabled = True

    def forward(self, input):
        if self.norm_p > 0 and self.p_mode == 'fast':
            input = percentile_zero(input, p=self.norm_p, mode='abs-norm')
        if self.abs_p > 0 and self.p_mode == 'fast':
            input = percentile_zero(input, p=self.abs_p, mode='abs-sum')
        if self.mean_p > 0 and self.p_mode == 'fast':
            input = percentile_zero(input, p=self.mean_p, mode='abs-mean')

        if self.norm_p > 0 and self.p_mode == 'slow':
            self.zero_weak_norm.capture(input.clone())
            input = self.zero_weak_norm(input)
        if self.abs_p > 0 and self.p_mode == 'slow':
            self.zero_weak_abs.capture(input.clone())
            input = self.zero_weak_abs(input)
        if self.mean_p > 0 and self.p_mode == 'slow':
            self.zero_weak_mean.capture(input.clone())
            input = self.zero_weak_mean(input)
        return input