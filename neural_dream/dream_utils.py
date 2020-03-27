import math
import random
import torch
import torch.nn as nn


# Get tensor percentile
def tensor_percentile(t, p=99.98):
    return t.view(-1).kthvalue(1 + round(0.01 * float(p) * (t.numel() - 1))).values.item()


# Shift tensor, possibly randomly
def roll_tensor(tensor, h_shift=None, w_shift=None):
    if h_shift == None:
        h_shift = torch.LongTensor(10).random_(-tensor.size(1), tensor.size(1))[0].item()
    if w_shift == None:
        w_shift = torch.LongTensor(10).random_(-tensor.size(2), tensor.size(2))[0].item()
    tensor = torch.roll(torch.roll(tensor, shifts=h_shift, dims=2), shifts=w_shift, dims=3)
    return tensor, h_shift, w_shift


# Define an nn Module to perform guassian blurring
class GaussianBlur(nn.Module):
    def __init__(self, k_size, sigma):
        super(GaussianBlur, self).__init__()
        self.k_size = k_size
        self.sigma = sigma

    def capture(self, input):
        if input.dim() == 4:
            d_val = 2
            self.groups = input.size(1)
        elif input.dim() == 2:
            d_val = 1
            self.groups = input.size(0)

        self.k_size, self.sigma = [self.k_size] * d_val, [self.sigma] * d_val
        kernel = 1

        meshgrid_tensor = torch.meshgrid([torch.arange(size, dtype=torch.float32, \
        device=input.get_device()) for size in self.k_size])

        for size, std, mgrid in zip(self.k_size, self.sigma, meshgrid_tensor):
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
            torch.exp(-((mgrid - ((size - 1) / 2)) / std) ** 2 / 2)

        kernel = (kernel / torch.sum(kernel)).view(1, 1, * kernel.size())
        kernel = kernel.repeat(self.groups, * [1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)

        if d_val == 2:
            self.conv = torch.nn.functional.conv2d
        elif d_val == 1:
            self.conv = torch.nn.functional.conv1d

    def forward(self, input, pad_mode='reflect'):
        d_val = input.dim()
        if input.dim() > 2:
            input = torch.nn.functional.pad(input, (2, 2, 2, 2), mode=pad_mode)
        else:
            input = input.view(1, 1, input.size(1))
        input = self.conv(input, weight=self.weight, groups=self.groups)
        if d_val == 2:
            p1d = nn.ConstantPad1d(2, 0)
            input = p1d(input)
            input = input.view(1, input.size(2))
        return input


# Define a Module to create guassian blur for laplacian pyramids
class GaussianBlurLP(GaussianBlur):

    def __init__(self, input, k_size=5, sigma=0):
        super(GaussianBlur, self).__init__()
        self.guass_blur = GaussianBlur(k_size, sigma)
        self.guass_blur.capture(input)

    def forward(self, input):
        return self.guass_blur(input)


# Define an nn Module to apply guassian blur as a layer
class GaussianBlurLayer(nn.Module):

    def __init__(self, k_size=5, sigma=0):
        super(GaussianBlurLayer, self).__init__()
        self.blur = GaussianBlur(k_size, sigma)
        self.mode = 'None'

    def forward(self, input):
        if self.mode == 'loss':
            input = self.blur(input)
        if self.mode == 'None':
            self.mode = 'capture'
        if self.mode == 'capture':
            self.blur.capture(input.clone())
            self.mode = 'loss'
        return input


# Define an nn Module to create a laplacian pyramid
class LaplacianPyramid(nn.Module):

    def __init__(self, input, scale=4, sigma=1):
        super(LaplacianPyramid, self).__init__()
        if len(sigma) == 1:
            sigma = (float(sigma[0]), float(sigma[0]) * scale)
        else:
            sigma = [float(s) for s in sigma]
        self.gauss_blur = GaussianBlurLP(input, 5, sigma[0])
        self.gauss_blur_hi = GaussianBlurLP(input, 5, sigma[1])
        self.scale = scale

    def split_lap(self, input):
        g = self.gauss_blur(input)
        gt = self.gauss_blur_hi(input)
        return g, input - gt

    def pyramid_list(self, input):
        pyramid_levels = []
        for i in range(self.scale):
            input, hi = self.split_lap(input)
            pyramid_levels.append(hi)
        pyramid_levels.append(input)
        return pyramid_levels[::-1]

    def lap_merge(self, pyramid_levels):
        b = torch.zeros_like(pyramid_levels[0])
        for p in pyramid_levels:
            b = b + p
        return b

    def forward(self, input):
        return self.lap_merge(self.pyramid_list(input))


# Define an nn Module to rank channels based on activation strength
class RankChannels(torch.nn.Module):

    def __init__(self, channels=1, channel_mode='strong'):
        super(RankChannels, self).__init__()
        self.channels = channels
        self.channel_mode = channel_mode

    def sort_channels(self, input):
        channel_list = []
        for i in range(input.size(1)):
            channel_list.append(torch.mean(input.clone().squeeze(0).narrow(0,i,1)).item())
        return sorted((c,v) for v,c in enumerate(channel_list))

    def get_middle(self, sequence):
        num = self.channels[0]
        m = (len(sequence) - 1)//2 - num//2
        return sequence[m:m+num]

    def remove_channels(self, cl):
        return [c for c in cl if c[1] not in self.channels]

    def rank_channel_list(self, input):
        top_channels = self.channels[0]
        channel_list = self.sort_channels(input)

        if 'strong' in self.channel_mode:
            channel_list.reverse()
        elif 'avg' in self.channel_mode:
            channel_list = self.get_middle(channel_list)
        elif 'ignore' in self.channel_mode:
            channel_list = self.remove_channels(channel_list)
            top_channels = len(channel_list)

        channels = []
        for i in range(top_channels):
            channels.append(channel_list[i][1])
        return channels

    def forward(self, input):
        return self.rank_channel_list(input)


class FFTTensor(nn.Module):

    def __init__(self, r=1, bl=25):
        super(FFTTensor, self).__init__()
        self.r = r
        self.bl = bl

    def block_low(self, input):
        if input.dim() == 5:
            hh, hw = int(input.size(2)/2), int(input.size(3)/2)
            input[:, :, hh-self.bl:hh+self.bl+1, hw-self.bl:hw+self.bl+1, :] = self.r
        elif input.dim() == 3:
            m = (input.size(1) - 1)//2 - self.bl//2
            input[:, m:m+self.bl, :] = self.r
        return input

    def fft_image(self, input, s=0):
        s_dim = 3 if input.dim() == 4 else 1
        s_dim = s_dim if s == 0 else s
        input = torch.rfft(input, signal_ndim=s_dim, onesided=False)
        real, imaginary = torch.unbind(input, -1)
        for r_dim in range(1, len(real.size())):
            n_shift = real.size(r_dim)//2
            if real.size(r_dim) % 2 != 0:
                n_shift += 1
            real = torch.roll(real, n_shift, dims=r_dim)
            imaginary = torch.roll(imaginary, n_shift, dims=r_dim)
        return torch.stack((real, imaginary), -1)

    def ifft_image(self, input, s=0):
        s_dim = 3 if input.dim() == 5 else 1
        s_dim = s_dim if s == 0 else s
        real, imaginary = torch.unbind(input, -1)
        for r_dim in range(len(real.size()) - 1, 0, -1):
            real = torch.roll(real, real.size(r_dim)//2, dims=r_dim)
            imaginary = torch.roll(imaginary, imaginary.size(r_dim)//2, dims=r_dim)
        return torch.irfft(torch.stack((real, imaginary), -1), signal_ndim=s_dim, onesided=False)

    def forward(self, input):
        input = self.block_low(self.fft_image(input))
        return torch.abs(self.ifft_image(input))


# Define an nn Module to apply jitter
class Jitter(torch.nn.Module):

    def __init__(self, jitter_val):
        super(Jitter, self).__init__()
        self.jitter_val = jitter_val

    def roll_tensor(self, input):
        h_shift = random.randint(-self.jitter_val, self.jitter_val)
        w_shift = random.randint(-self.jitter_val, self.jitter_val)
        return torch.roll(torch.roll(input, shifts=h_shift, dims=2), shifts=w_shift, dims=3)

    def forward(self, input):
        return self.roll_tensor(input)


# Define an nn Module to apply random transforms
class RandomTransform(torch.nn.Module):

    def __init__(self, t_val):
        super(RandomTransform, self).__init__()
        self.rotate, self.flip = False, False
        if t_val == 'all' or t_val == 'rotate':
            self.rotate = True
        if t_val == 'all' or t_val == 'flip':
            self.flip = True

    def rotate_tensor(self, input):
        if self.rotate:
            k_val = random.randint(0,3)
            input = torch.rot90(input, k_val, [2,3])
        return input

    def flip_tensor(self, input):
        if self.flip:
            flip_tensor = bool(random.randint(0,1))
            if flip_tensor:
                input = input.flip([2,3])
        return input

    def forward(self, input):
        return self.flip_tensor(self.rotate_tensor(input))


# Define an nn Module to label predicted channels
class Classify(nn.Module):

    def __init__(self, labels, k=1):
        super(Classify, self).__init__()
        self.labels = [str(n) for n in labels]
        self.k = k

    def forward(self, input):
        channel_ids = torch.topk(input, self.k).indices
        channel_ids = [n.item() for n in channel_ids[0]]
        label_names = ''
        for i in channel_ids:
            if label_names != '':
                label_names += ', ' + self.labels[i]
            else:
                label_names += self.labels[i]
        print('  Predicted labels: ' + label_names)


# Run inception modules with preprocessing layers
class ModelPlus(nn.Module):

    def __init__(self, input_net, net):
        super(ModelPlus, self).__init__()
        self.input_net = input_net
        self.net = net

    def forward(self, input):
        return self.net(self.input_net(input))