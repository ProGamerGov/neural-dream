import torch
import torch.nn as nn
import neural_dream.neural_dream_core.dream_utils as dream_utils


# Adjust tensor contrast
def adjust_contrast(t, r, p=99.98):
    return t * (r / dream_utils.tensor_percentile(t))


# Resize tensor
def resize_tensor(tensor, size, mode='bilinear'):
    return torch.nn.functional.interpolate(tensor.clone(), size=size, mode=mode, align_corners=True)


# Center crop a tensor
def center_crop(input, crop_val, mode='percent'):
       h, w = input.size(2), input.size(3)
       if mode == 'percent':
            h_crop = int((crop_val / 100) * input.size(2))
            w_crop = int((crop_val / 100) * input.size(3))
       elif mode == 'pixel':
            h_crop = input.size(2) - crop_val
            w_crop = input.size(3) - crop_val
       sw, sh = w // 2 - (w_crop // 2), h // 2 - (h_crop // 2)
       return input[:, :, sh:sh + h_crop, sw:sw + w_crop]


# Center crop and resize a tensor
def zoom(input, crop_val, mode='percent'):
    h, w = input.size(2), input.size(3)
    input = center_crop(input.clone(), crop_val, mode=mode)
    input = resize_tensor(input, (h,w))
    return input