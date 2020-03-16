#!/usr/bin/env python
import os
import copy
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from neural_dream.neural_dream_core.CaffeLoader import loadCaffemodel, ModelParallel
import neural_dream.neural_dream_core.dream_utils as dream_utils
import neural_dream.neural_dream_core.loss_layers as dream_loss_layers
import neural_dream.neural_dream_core.dream_image as dream_image
import neural_dream.neural_dream_core.dream_model as dream_model
from neural_dream.neural_dream_core.dream_auto import auto_model_mode, auto_mean

import argparse
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument("-content_image", help="Content target image", default='')
parser.add_argument("-image_size", help="Maximum height / width of generated image", type=int, default=512)
parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = c", default=0)

# Optimization options
parser.add_argument("-dream_weight", type=float, default=1000)
parser.add_argument("-normalize_weights", action='store_true')
parser.add_argument("-tv_weight", type=float, default=0)
parser.add_argument("-l2_weight", type=float, default=0)
parser.add_argument("-num_iterations", type=int, default=10)
parser.add_argument("-jitter", type=int, default=32)
parser.add_argument("-init", choices=['random', 'image'], default='image')
parser.add_argument("-optimizer", choices=['lbfgs', 'adam'], default='adam')
parser.add_argument("-learning_rate", type=float, default=1.5)
parser.add_argument("-lbfgs_num_correction", type=int, default=100)
parser.add_argument("-loss_mode", choices=['bce', 'mse', 'mean', 'norm', 'l2', 'abs_mean', 'abs_l2'], default='l2')

# Output options
parser.add_argument("-print_iter", type=int, default=1)
parser.add_argument("-print_octave_iter", type=int, default=0)
parser.add_argument("-save_iter", type=int, default=1)
parser.add_argument("-save_octave_iter", type=int, default=0)
parser.add_argument("-output_image", default='out.png')

# Octave options
parser.add_argument("-num_octaves", type=int, default=2)
parser.add_argument("-octave_scale", type=float, default=0.6)
parser.add_argument("-octave_iter", type=int, default=50)
parser.add_argument("-octave_mode", choices=['advanced', 'normal'], default='normal')

# Channel options
parser.add_argument("-channels", type=str, help="channels for DeepDream", default='-1')
parser.add_argument("-channel_mode", choices=['all', 'strong', 'avg', 'weak', 'ignore'], default='all')
parser.add_argument("-channel_capture", choices=['once', 'iter'], default='once')

# Guassian Blur Options
parser.add_argument("-layer_sigma", type=float, default=0)

# Laplacian pyramid options
parser.add_argument("-lap_scale", type=int, default=0)
parser.add_argument("-sigma", default='1')

# FFT options
parser.add_argument("-use_fft", action='store_true')
parser.add_argument("-fft_block", type=int, default=25)

# Zoom options
parser.add_argument("-zoom", type=int, default=0)
parser.add_argument("-zoom_mode", choices=['percent', 'pixel'], default='percent')

# Other options
parser.add_argument("-original_colors", type=int, choices=[0, 1], default=0)
parser.add_argument("-pooling", choices=['avg', 'max'], default='max')
parser.add_argument("-model_file", type=str, help="Ex: path/to/model | 'bvlc_googlenet.pth', 'googlenet_places365.pth', 'nin_imagenet.pth'", default='')
parser.add_argument("-model_type", choices=['caffe', 'pytorch', 'auto'], default='auto')
parser.add_argument("-model_mean", default='auto')
parser.add_argument("-label_file", type=str, default='')
parser.add_argument("-disable_check", action='store_true')
parser.add_argument("-backend", choices=['nn', 'cudnn', 'mkl', 'mkldnn', 'openmp', 'mkl,cudnn', 'cudnn,mkl'], default='nn')
parser.add_argument("-cudnn_autotune", action='store_true')
parser.add_argument("-seed", type=int, default=-1)
parser.add_argument("-clamp", action='store_true')
parser.add_argument("-random_transforms", choices=['none', 'all', 'flip', 'rotate'], default='none')
parser.add_argument("-adjust_contrast", type=float, help="try 99.98", default=-1)
parser.add_argument("-classify", type=int, default=0)

parser.add_argument("-dream_layers", help="layers for DeepDream", default='inception_4d_3x3_reduce')

parser.add_argument("-multidevice_strategy", default='4,7,29')

# Help options
parser.add_argument("-print_layers", action='store_true')
parser.add_argument("-print_channels", action='store_true')

# PYPI download models
parser.add_argument("-download_path", help="Path to where models should be downloaded to", type=str, default='')
parser.add_argument("-download_models", help="Comma separated list of models to dowload", nargs="?", type=str, const="none")

# Experimental params
parser.add_argument("-norm_percent", type=float, default=0)
parser.add_argument("-abs_percent", type=float, default=0)
parser.add_argument("-mean_percent", type=float, default=0)
parser.add_argument("-percent_mode", choices=['slow', 'fast'], default='fast')
params = parser.parse_args()


Image.MAX_IMAGE_PIXELS = 1000000000 # Support gigapixel images


def main():
    if params.download_models:
        download_models(params.download_models, params.download_path)

    if params.content_image == '':
        raise ValueError("You must select a content image.")
    if params.model_file == '':
        raise ValueError("You must select a model file.")


    dtype, multidevice, backward_device = setup_gpu()

    cnn, layerList = loadCaffemodel(params.model_file, params.pooling, params.gpu, params.disable_check, True)
    has_inception = cnn.has_inception
    if params.print_layers:
        print_layers(layerList, params.model_file, has_inception)

    params.model_type = auto_model_mode(params.model_file) if params.model_type == 'auto' else params.model_type
    input_mean = auto_mean(params.model_file, params.model_type) if params.model_mean == 'auto' else params.model_mean
    if params.model_mean != 'auto':
        input_mean = [float(m) for m in input_mean.split(',')]

    content_image = preprocess(params.content_image, params.image_size, params.model_type, input_mean).type(dtype)
    clamp_val = 256 if params.model_type == 'caffe' else 1

    if params.label_file != '':
        labels = load_label_file(params.label_file)
        params.channels = channel_ids(labels, params.channels)
    if params.classify > 0:
         if not has_inception:
             params.dream_layers += ',classifier'
         if params.label_file == '':
             labels = list(range(0, 1000))

    dream_layers = params.dream_layers.split(',')
    start_params = (dtype, params.random_transforms, params.jitter, params.tv_weight, params.l2_weight, params.layer_sigma)
    primary_params = (params.loss_mode, params.dream_weight, params.channels, params.channel_mode)
    secondary_params = {'channel_capture': params.channel_capture, 'scale': params.lap_scale, 'sigma': params.sigma, \
    'use_fft': (params.use_fft, params.fft_block), 'r': clamp_val, 'p_mode': params.percent_mode, 'norm_p': params.norm_percent, \
	'abs_p': params.abs_percent, 'mean_p': params.mean_percent}

    # Set up the network, inserting dream loss modules
    net_base, dream_losses, tv_losses, l2_losses, lm_layer_names, loss_module_list = dream_model.build_net(cnn, dream_layers, \
	has_inception, layerList, params.classify, start_params, primary_params, secondary_params)

    if params.classify > 0:
       classify_img = dream_utils.Classify(labels, params.classify)

    if multidevice and not has_inception:
        net_base = setup_multi_device(net_base)

    if not has_inception:
        print_torch(net_base, multidevice)

    # Initialize the image
    if params.seed >= 0:
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)
        torch.backends.cudnn.deterministic=True
        random.seed(params.seed)
    if params.init == 'random':
        base_img = torch.randn_like(content_image).mul(0.001)
    elif params.init == 'image':
        base_img = content_image.clone()

    if params.optimizer == 'lbfgs':
        print("Running optimization with L-BFGS")
    else:
        print("Running optimization with ADAM")

    for param in net_base.parameters():
        param.requires_grad = False

    for i in dream_losses:
        i.mode = 'capture'
    net_base(base_img.clone())

    if params.channels != '-1' or params.channel_mode != 'all' and params.channels != '-1':
        print_channels(dream_losses, dream_layers, params.print_channels)
    if params.classify > 0:
       feat = net_base(base_img.clone())
       classify_img(feat)

    for i in dream_losses:
        i.mode = 'None'

    current_img = new_img(base_img, -1)
    h, w = current_img.size(2), current_img.size(3)
    octave_list = ocatve_calc((h,w), params.octave_scale, params.num_octaves, params.octave_mode)
    print_octave_sizes(octave_list)

    total_dream_losses, total_loss = [], [0]

    for iter in range(1, params.num_iterations+1):
        for octave, octave_sizes in enumerate(octave_list, 1):
            net = copy.deepcopy(net_base) if not has_inception else net_base
            for param in net.parameters():
                param.requires_grad = False
            dream_losses, tv_losses, l2_losses = [], [], []
            if not has_inception:
                for i, layer in enumerate(net):
                    if isinstance(layer, dream_loss_layers.TVLoss):
                        tv_losses.append(layer)
                    if isinstance(layer, dream_loss_layers.L2Regularizer):
                        l2_losses.append(layer)
                    if 'DreamLoss' in str(type(layer)):
                        dream_losses.append(layer)
            elif has_inception:
                 net, dream_losses, tv_losses, l2_losses = dream_model.renew_net(start_params, net, loss_module_list, lm_layer_names)

            img = new_img(current_img.clone(), octave_sizes)

            net(img)
            for i in dream_losses:
                i.mode = 'loss'
            for i in dream_losses:
                i.mode = 'loss'

            # Maybe normalize dream weight
            if params.normalize_weights:
                normalize_weights(dream_losses)

            # Freeze the net_basework in order to prevent
            # unnecessary gradient calculations
            for param in net.parameters():
                param.requires_grad = False

            # Function to evaluate loss and gradient. We run the net_base forward and
            # backward to get the gradient, and sum up losses from the loss modules.
            # optim.lbfgs internally handles iteration and calls this function many
            # times, so we manually count the number of iterations to handle printing
            # and saving intermediate results.
            num_calls = [0]
            def feval():
                num_calls[0] += 1
                optimizer.zero_grad()
                net(img)
                loss = 0

                for mod in dream_losses:
                    loss += -mod.loss.to(backward_device)
                if params.tv_weight > 0:
                    for mod in tv_losses:
                        loss += mod.loss.to(backward_device)
                if params.l2_weight > 0:
                    for mod in l2_losses:
                        loss += mod.loss.to(backward_device)

                if params.clamp:
                     img.clamp(0, clamp_val)
                if params.adjust_contrast > -1:
                    img.data = dream_image.adjust_contrast(img, r=clamp_val, p=params.adjust_contrast)

                total_loss[0] += loss.item()

                loss.backward()

                maybe_print_octave_iter(num_calls[0], octave, params.octave_iter, dream_losses)
                maybe_save_octave(iter, num_calls[0], octave, img, content_image, input_mean)

                return loss

            optimizer, loopVal = setup_optimizer(img)
            while num_calls[0] <= params.octave_iter:
                optimizer.step(feval)

            if octave == 1:
                for mod in dream_losses:
                    total_dream_losses.append(mod.loss.item())
            else:
                for d_loss, mod in enumerate(dream_losses):
                    total_dream_losses[d_loss] += mod.loss.item()

            if img.size(2) != h or img.size(3) != w:
                current_img = dream_image.resize_tensor(img.clone(), (h,w))
            else:
                current_img = img.clone()

        maybe_print(iter, total_loss[0], total_dream_losses)
        maybe_save(iter, current_img, content_image, input_mean)
        total_dream_losses, total_loss = [], [0]

        if params.zoom > 0:
           current_img = dream_image.zoom(current_img, params.zoom, params.zoom_mode)
        if params.classify > 0:
            feat = net_base(base_img.clone())
            classify_img(feat)


def save_output(t, save_img, content_image, iter_name, model_mean):
    output_filename, file_extension = os.path.splitext(params.output_image)
    if t == params.num_iterations:
        filename = output_filename + str(file_extension)
    else:
        filename = str(output_filename) + iter_name + str(file_extension)
    disp = deprocess(save_img.clone(), params.model_type, model_mean)

    # Maybe perform postprocessing for color-independent style transfer
    if params.original_colors == 1:
        disp = original_colors(deprocess(content_image.clone(), params.model_type, model_mean), disp)

    disp.save(str(filename))


def maybe_save(t, save_img, content_image, input_mean):
    should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save:
        save_output(t, save_img, content_image, "_" + str(t), input_mean)


def maybe_save_octave(t, n, o, save_img, content_image, input_mean):
    should_save = params.save_octave_iter > 0 and n % params.save_octave_iter == 0
    should_save = should_save or params.save_octave_iter > 0 and n == params.octave_iter
    if o == params.num_octaves:
        should_save = False if params.save_iter > 0 and t % params.save_iter == 0 or t == params.num_iterations else should_save
    if should_save:
        save_output(t, save_img, content_image, "_" + str(t) + "_" + str(o) + "_" + str(n), input_mean)


def maybe_print(t, loss, dream_losses):
    if params.print_iter > 0 and t % params.print_iter == 0:
        print("Iteration " + str(t) + " / "+ str(params.num_iterations))
        for i, loss_module in enumerate(dream_losses):
            print("  DeepDream " + str(i+1) + " loss: " + str(loss_module))
        print("  Total loss: " + str(abs(loss)))


def maybe_print_octave_iter(t, n, total, dream_losses):
    if params.print_octave_iter > 0 and t % params.print_octave_iter == 0:
        print("Octave iter "+str(n) +" iteration " + str(t) + " / "+ str(total))
        for i, loss_module in enumerate(dream_losses):
            print("  DeepDream " + str(i+1) + " loss: " + str(loss_module.loss.item()))


def print_channels(dream_losses, layers, print_all_channels=False):
    print('\nSelected layer channels:')
    if not print_all_channels:
        for i, l in enumerate(dream_losses):
            if len(l.dream.channels) > 25:
                ch = l.dream.channels[0:25] + ['and ' + str(len(l.dream.channels[25:])) + ' more...']
            else:
                ch = l.dream.channels
            print('  ' + layers[i] + ': ', ch)
    elif print_all_channels:
        for i, l in enumerate(dream_losses):
            ch = l.dream.channels
            print('  ' + layers[i] + ': ', ch)


# Configure the optimizer
def setup_optimizer(img):
    if params.optimizer == 'lbfgs':
        optim_state = {
            'max_iter': params.num_iterations,
            'tolerance_change': -1,
            'tolerance_grad': -1,
            'lr': params.learning_rate
        }
        if params.lbfgs_num_correction != 100:
            optim_state['history_size'] = params.lbfgs_num_correction
        optimizer = optim.LBFGS([img], **optim_state)
        loopVal = 1
    elif params.optimizer == 'adam':
        optimizer = optim.Adam([img], lr = params.learning_rate)
        loopVal = params.num_iterations - 1
    return optimizer, loopVal


def setup_gpu():
    def setup_cuda():
        if 'cudnn' in params.backend:
            torch.backends.cudnn.enabled = True
            if params.cudnn_autotune:
                torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.enabled = False

    def setup_cpu():
        if 'mkl' in params.backend and 'mkldnn' not in params.backend:
            torch.backends.mkl.enabled = True
        elif 'mkldnn' in params.backend:
            raise ValueError("MKL-DNN is not supported yet.")
        elif 'openmp' in params.backend:
            torch.backends.openmp.enabled = True

    multidevice = False
    if "," in str(params.gpu):
        devices = params.gpu.split(',')
        multidevice = True

        if 'c' in str(devices[0]).lower():
            backward_device = "cpu"
            setup_cuda(), setup_cpu()
        else:
            backward_device = "cuda:" + devices[0]
            setup_cuda()
        dtype = torch.FloatTensor

    elif "c" not in str(params.gpu).lower():
        setup_cuda()
        dtype, backward_device = torch.cuda.FloatTensor, "cuda:" + str(params.gpu)
    else:
        setup_cpu()
        dtype, backward_device = torch.FloatTensor, "cpu"
    return dtype, multidevice, backward_device


def setup_multi_device(net_base):
    assert len(params.gpu.split(',')) - 1 == len(params.multidevice_strategy.split(',')), \
      "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."

    new_net_base = ModelParallel(net_base, params.gpu, params.multidevice_strategy)
    return new_net_base


# Preprocess an image before passing it to a model.
# Maybe rescale from [0, 1] to [0, 255], convert from RGB to BGR,
# and subtract the mean pixel.
def preprocess(image_name, image_size, mode='caffe', input_mean=[103.939, 116.779, 123.68]):
    image = Image.open(image_name).convert('RGB')
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    if mode == 'caffe':
        rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        Normalize = transforms.Compose([transforms.Normalize(mean=input_mean, std=[1,1,1])])
        tensor = Normalize(rgb2bgr(Loader(image) * 256)).unsqueeze(0)
    elif mode == 'pytorch':
        Normalize = transforms.Compose([transforms.Normalize(mean=input_mean, std=[1,1,1])])
        tensor = Normalize(Loader(image)).unsqueeze(0)
    return tensor


# Undo the above preprocessing.
def deprocess(output_tensor, mode='caffe', input_mean=[-103.939, -116.779, -123.68]):
    input_mean = [n * -1 for n in input_mean]
    if mode == 'caffe':
        Normalize = transforms.Compose([transforms.Normalize(mean=input_mean, std=[1,1,1])])
        bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 256
    elif mode == 'pytorch':
        Normalize = transforms.Compose([transforms.Normalize(mean=input_mean, std=[1,1,1])])
        output_tensor = Normalize(output_tensor.squeeze(0).cpu())
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    return image


# Combine the Y channel of the generated image and the UV/CbCr channels of the
# content image to perform color-independent style transfer.
def original_colors(content, generated):
    content_channels = list(content.convert('YCbCr').split())
    generated_channels = list(generated.convert('YCbCr').split())
    content_channels[0] = generated_channels[0]
    return Image.merge('YCbCr', content_channels).convert('RGB')


# Print like Lua/Torch7
def print_torch(net_base, multidevice):
    if multidevice:
        return
    simplelist = ""
    for i, layer in enumerate(net_base, 1):
        simplelist = simplelist + "(" + str(i) + ") -> "
    print("nn.Sequential ( \n  [input -> " + simplelist + "output]")

    def strip(x):
        return str(x).replace(", ",',').replace("(",'').replace(")",'') + ", "
    def n():
        return "  (" + str(i) + "): " + "nn." + str(l).split("(", 1)[0]

    for i, l in enumerate(net_base, 1):
         if "2d" in str(l):
             if "AdaptiveAvgPool2d" not in str(l) and "AdaptiveMaxPool2d" not in str(l) and "BasicConv2d" not in str(l):
                 ks, st, pd = strip(l.kernel_size), strip(l.stride), strip(l.padding)
             if "BasicConv2d" in str(l):
                 print(n())
             elif "Conv2d" in str(l):
                 ch = str(l.in_channels) + " -> " + str(l.out_channels)
                 print(n() + "(" + ch + ", " + (ks).replace(",",'x', 1) + st + pd.replace(", ",')'))
             elif "AdaptiveAvgPool2d" in str(l) or "AdaptiveMaxPool2d" in str(l):
                 print(n())
             elif "Pool2d" in str(l):
                 st = st.replace("  ",' ') + st.replace(", ",')')
                 print(n() + "(" + ((ks).replace(",",'x' + ks, 1) + st).replace(", ",','))
         else:
             print(n())
    print(")")


# Print planned octave image sizes
def print_octave_sizes(octave_list):
    print('\nPerforming ' + str(len(octave_list)) + ' octaves with the following image sizes:')
    for o, octave in enumerate(octave_list):
        print('  Octave ' + str(o+1) + ' image size: ' + \
        str(octave[0]) +'x'+ str(octave[1]))
    print()


def ocatve_calc(image_size, octave_scale, num_octaves, mode='advanced'):
   octave_list = []
   h_size, w_size = image_size[0], image_size[1]
   if mode == 'normal':
       for o in range(1, num_octaves+1):
           h_size *= octave_scale
           w_size *= octave_scale
           if o < num_octaves:
               octave_list.append((int(h_size), int(w_size)))
       octave_list.reverse()
       octave_list.append((image_size[0], image_size[1]))
   elif mode == 'advanced':
       for o in range(1, num_octaves+1):
           h_size = image_size[0] * (o * octave_scale)
           w_size = image_size[1] * (o * octave_scale)
           octave_list.append((int(h_size), int(w_size)))
   return octave_list


# Divide weights by channel size
def normalize_weights(dream_losses):
    for n, i in enumerate(dream_losses):
        i.strength = i.strength / max(i.target_size)


# Print all available/usable layer names
def print_layers(layerList, model_name, has_inception):
    print()
    print("\nUsable Layers For '" + model_name + "':")
    if not has_inception:
        for l_names in layerList:
            if l_names == 'P':
                n = '  Pooling Layers:'
            if l_names == 'C':
                n = '  Conv Layers:'
            if l_names == 'R':
                n = '  ReLU Layers:'
            elif l_names == 'BC':
                n = '  BasicConv2d Layers:'
            elif l_names == 'L':
                n = '  Linear/FC layers:'
            if l_names == 'D':
                n = '  Dropout Layers:'
            elif l_names == 'IC':
                n = '  Inception Layers:'
            print(n, ', '.join(layerList[l_names]))
    elif has_inception:
        for l in layerList:
            print(l)
    quit()


# Load a label file
def load_label_file(filename):
    with open(filename, 'r') as f:
        x = [l.rstrip('\n') for l in f.readlines()]
    return x


# Convert names to channel values
def channel_ids(l, channels):
    channels = channels.split(',')
    c_vals = ''
    for c in channels:
        if c.isdigit():
            c_vals += ',' + str(c)
        elif c.isalpha():
            v = ','.join(str(ch) for ch, n in enumerate(l) if c in n)
            v = ',' + v + ',' if len(v.split(',')) == 1 else v
            c_vals += v
    c_vals = '-1' if c_vals == '' else c_vals
    c_vals = c_vals.replace(',', '', 1) if c_vals[0] == ',' else c_vals
    return c_vals


# Prepare input image
def new_img(input_image, scale_factor, mode='bilinear'):
    img = input_image.clone()
    if scale_factor != -1:
        img = dream_image.resize_tensor(img, scale_factor, mode)
    return nn.Parameter(img)

	
def download_models(models, download_path):
    if os.name == 'nt':
        raise ValueError("You must manually download models for Windows OS.")
    from os import path
    from sys import version_info
    from collections import OrderedDict
    from torch.utils.model_zoo import load_url

    if version_info[0] < 3:
        import urllib
    else:
        import urllib.request 
	
    def download_file(fileurl, name, download_path):
        if version_info[0] < 3:
            urllib.URLopener().retrieve(fileurl, path.join(download_path, name))
        else:
            urllib.request.urlretrieve(fileurl, path.join(download_path, name))

    if models == 'none':
        models = 'caffe-googlenet-bvlc,caffe-nin'
    if download_path == 'none':
        download_path = os.path.expanduser("~")	
	
    options_list = ['all', 'caffe-vgg16', 'caffe-vgg19', 'caffe-nin', 'caffe-googlenet-places205', 'caffe-googlenet-places365', 'caffe-googlenet-bvlc', 'caffe-googlenet-cars', 'caffe-googlenet-sos', \
                    'caffe-resnet-opennsfw', 'pytorch-vgg16', 'pytorch-vgg19', 'pytorch-googlenet', 'pytorch-inceptionv3', 'tensorflow-inception5h', 'all-caffe', 'all-caffe-googlenet']

    if 'print-all' in models:
        print(options_list)
        quit()
		
    if models == 'all':
        models = options_list[1:15]
    elif 'all-caffe' in models and 'all-caffe-googlenet' not in models: 
        models = options_list[1:10] + models.split(',')
    elif 'all-caffe-googlenet' in models:
        models = options_list[4:9] + models.split(',')
    else:
        models = models.split(',')

    if 'caffe-vgg19' in models:
        # Download the VGG-19 ILSVRC model and fix the layer names
        print("Downloading the VGG-19 ILSVRC model")
        sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth")
        map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
        sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
        torch.save(sd, path.join(download_path, "vgg19-d01eb7cb.pth"))

    if 'caffe-vgg16' in models:
        # Download the VGG-16 ILSVRC model and fix the layer names
        print("Downloading the VGG-16 ILSVRC model")
        sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth")
        map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
        sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
        torch.save(sd, path.join(download_path, "vgg16-00b39a1b.pth"))     
    
    if 'caffe-nin' in models:
        # Download the NIN model
        print("Downloading the NIN model")  
        fileurl = "https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth"
        name = "nin_imagenet.pth"
        download_file(fileurl, name, download_path)
        
    if 'caffe-googlenet-places205' in models:
        # Download the Caffe GoogeLeNet Places205 model
        print("Downloading the Places205 GoogeLeNet model")  
        fileurl = "https://github.com/ProGamerGov/pytorch-places/raw/master/googlenet_places205.pth"
        name = "googlenet_places205.pth"
        download_file(fileurl, name, download_path)
        
    if 'caffe-googlenet-places365' in models:
        # Download the Caffe GoogeLeNet Places365 model 
        print("Downloading the Places365 GoogeLeNet model")  
        fileurl = "https://github.com/ProGamerGov/pytorch-places/raw/master/googlenet_places365.pth"
        name = "googlenet_places365.pth"
        download_file(fileurl, name, download_path)
        
    if 'caffe-googlenet-bvlc' in models:
        # Download the Caffe BVLC GoogeLeNet model
        print("Downloading the BVLC GoogeLeNet model")  
        fileurl = "https://github.com/ProGamerGov/pytorch-old-caffemodels/raw/master/bvlc_googlenet.pth"
        name = "bvlc_googlenet.pth"
        download_file(fileurl, name, download_path)
        
    if 'caffe-googlenet-cars' in models:
        # Download the Caffe GoogeLeNet Cars model
        print("Downloading the Cars GoogeLeNet model")  
        fileurl = "https://github.com/ProGamerGov/pytorch-old-caffemodels/raw/master/googlenet_finetune_web_cars.pth"
        name = "googlenet_finetune_web_cars.pth"
        download_file(fileurl, name, download_path)
        
    if 'caffe-googlenet-sos' in models:
        # Download the Caffe GoogeLeNet SOS model
        print("Downloading the SOD GoogeLeNet model")  
        fileurl = "https://github.com/ProGamerGov/pytorch-old-caffemodels/raw/master/GoogleNet_SOS.pth"
        name = "GoogleNet_SOS.pth"
        download_file(fileurl, name, download_path)    
        
    if 'pytorch-vgg19' in models:
        # Download the PyTorch VGG19 model
        print("Downloading the PyTorch VGG 19 model")   
        fileurl = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
        name = "vgg19-dcbb9e9d.pth"
        download_file(fileurl, name, download_path)
   
    if 'pytorch-vgg16' in models:
        # Download the PyTorch VGG16 model
        print("Downloading the PyTorch VGG 16 model")
        fileurl = "https://download.pytorch.org/models/vgg16-397923af.pth"
        name = "vgg16-397923af.pth"
        download_file(fileurl, name, download_path)

    if 'pytorch-googlenet' in models:
        # Download the PyTorch GoogLeNet model
        print("Downloading the PyTorch GoogLeNet model")
        fileurl = "https://download.pytorch.org/models/googlenet-1378be20.pth"
        name = "googlenet-1378be20.pth"
        download_file(fileurl, name, download_path)

    if 'pytorch-inception' in models:
        # Download the PyTorch Inception V3 model
        print("Downloading the PyTorch Inception V3 model")
        fileurl = "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
        name = "inception_v3_google-1a9a5a14.pth"
        download_file(fileurl, name, download_path)    

    if 'tensorflow-inception5h' in models:
        # Download the Inception5h model
        print("Downloading the Tensorflow Inception5h model")
        fileurl = "https://github.com/ProGamerGov/pytorch-old-tensorflow-models/raw/master/inception5h.pth"
        name = "inception5h.pth"
        download_file(fileurl, name, download_path)  
    
    if 'caffe-resnet-opennsfw' in models:
        # Download the ResNet Yahoo Open NSFW model
        print("Downloading the ResNet Yahoo Open NSFW model")
        fileurl = "https://github.com/ProGamerGov/pytorch-old-caffemodels/raw/master/ResNet_50_1by2_nsfw.pth"
        name = "ResNet_50_1by2_nsfw.pth"
        download_file(fileurl, name, download_path)
        
    print("All selected models have been successfully downloaded to " + download_path)
    quit()
	
	
	
if __name__ == "__main__":
    main()