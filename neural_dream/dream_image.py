import os
import torch
import torch.nn as nn
import neural_dream.dream_utils as dream_utils
from PIL import Image


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

# Create gif from images
def create_gif(frames_dir='.', base_name=None, duration=100):
    ext = [".jpg", ".jpeg", ".png", ".tiff"]	
    image_list = [file for file in os.listdir(frames_dir) if os.path.splitext(file)[1].lower() in ext]
    if base_name != None:
         image_list = [im for im in image_list if base_name in im]
    if "_" in image_list[0]:
        base_name = image_list[0].rsplit('_', 1)[0] if base_name == None else base_name
        fsorted = sorted(image_list,key=lambda x: int(os.path.splitext(x)[0].rsplit('_', 1)[1]))		
    else:
        base_name = image_list[0].rsplit('.', 1)[0] if base_name == None else base_name	
        fsorted = sorted(image_list[1:],key=lambda x: int(os.path.splitext(x)[0].rsplit('_', 1)[1]))
        fsorted.append(image_list[0])		

    frames = [Image.open(os.path.join(frames_dir, im)).convert('RGB') for im in fsorted]
    frames[0].save(os.path.join(frames_dir, base_name+'.gif'), format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
