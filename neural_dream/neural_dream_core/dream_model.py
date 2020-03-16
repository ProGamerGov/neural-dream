import copy
import torch
import torch.nn as nn
import neural_dream.neural_dream_core.dream_utils as dream_utils
import neural_dream.neural_dream_core.loss_layers as dream_loss_layers
from neural_dream.neural_dream_core.loss_layers import DreamLoss, DreamLossHook, DreamLossPreHook
from neural_dream.neural_dream_core.CaffeLoader import Flatten


# Register DreamLoss layers as hooks on model
def add_to_incept(net, layer_name, layer_num, loss_module, capture='after'):
        if len(layer_name) == 1:
            if capture == 'after':
                getattr(net, layer_name[0]).register_forward_hook(loss_module)
            elif capture == 'before':
                getattr(net, layer_name[0]).register_forward_pre_hook(loss_module)
        elif len(layer_name) == 2:
            if isinstance(getattr(getattr(net, layer_name[0]), layer_name[1]), nn.Sequential):
                if capture == 'after':
                    getattr(getattr(getattr(net, layer_name[0]), layer_name[1]), str(layer_num)).register_forward_hook(loss_module)
                elif capture == 'before':
                    getattr(getattr(getattr(net, layer_name[0]), layer_name[1]), str(layer_num)).register_forward_pre_hook(loss_module)
                layer_num = layer_num+1 if layer_num < 1 else 0
            else:
                if capture == 'after':
                    getattr(getattr(net, layer_name[0]), layer_name[1]).register_forward_hook(loss_module)
                elif capture == 'before':
                    getattr(getattr(net, layer_name[0]), layer_name[1]).register_forward_pre_hook(loss_module)
        elif len(layer_name) == 3:
            if isinstance(getattr(getattr(net, layer_name[0]), layer_name[1]), nn.Sequential):
                if capture == 'after':
                    getattr(getattr(getattr(getattr(net, layer_name[0]), layer_name[1]), str(layer_num)), layer_name[2]).register_forward_hook(loss_module)
                elif capture == 'before':
                    getattr(getattr(getattr(getattr(net, layer_name[0]), layer_name[1]), str(layer_num)), layer_name[2]).register_forward_pre_hook(loss_module)
                layer_num = layer_num+1 if layer_num < 1 else 0
            else:
                if capture == 'after':
                    getattr(getattr(getattr(net, layer_name[0]), layer_name[1]), layer_name[2]).register_forward_hook(loss_module)
                elif capture == 'before':
                    getattr(getattr(getattr(net, layer_name[0]), layer_name[1]), layer_name[2]).register_forward_pre_hook(loss_module)
        return loss_module, layer_num


# Create DeepDream model
def build_net(cnn, dream_layers, has_inception, layerList, use_classify, start_params, primary_params, secondary_params):
    cnn = copy.deepcopy(cnn)
    dream_losses, tv_losses, l2_losses = [], [], []

    lm_layer_names, loss_module_list = [], []
    dtype = start_params[0]
    if not has_inception:
        next_dream_idx = 1
        net_base = nn.Sequential()
        c, r, p, l, d = 0, 0, 0, 0, 0
        net_base, tv_losses, l2_losses = start_network(*start_params)

        for i, layer in enumerate(list(cnn), 1):
            if next_dream_idx <= len(dream_layers):
                if isinstance(layer, nn.Conv2d):
                    net_base.add_module(str(len(net_base)), layer)

                    if layerList['C'][c] in dream_layers:
                        print("Setting up dream layer " + str(i) + ": " + str(layerList['C'][c]))
                        loss_module = DreamLoss(*primary_params, **secondary_params)
                        net_base.add_module(str(len(net_base)), loss_module)
                        dream_losses.append(loss_module)
                    c+=1

                if isinstance(layer, nn.ReLU):
                    net_base.add_module(str(len(net_base)), layer)

                    if layerList['R'][r] in dream_layers:
                        print("Setting up dream layer " + str(i) + ": " + str(layerList['R'][r]))
                        loss_module = DreamLoss(*primary_params, **secondary_params)
                        net_base.add_module(str(len(net_base)), loss_module)
                        dream_losses.append(loss_module)
                        next_dream_idx += 1
                    r+=1

                if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                    net_base.add_module(str(len(net_base)), layer)

                    if layerList['P'][p] in dream_layers:
                        print("Setting up dream layer " + str(i) + ": " + str(layerList['P'][p]))
                        loss_module = DreamLoss(*primary_params, **secondary_params)
                        net_base.add_module(str(len(net_base)), loss_module)
                        dream_losses.append(loss_module)
                        next_dream_idx += 1
                    p+=1

                if isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer, nn.AdaptiveMaxPool2d):
                    net_base.add_module(str(len(net_base)), layer)

                if isinstance(layer, Flatten):
                    flatten_mod = Flatten().type(dtype)
                    net_base.add_module(str(len(net_base)), flatten_mod)

                if isinstance(layer, nn.Linear):
                    net_base.add_module(str(len(net_base)), layer)

                    if layerList['L'][l] in dream_layers:
                        print("Setting up dream layer " + str(i) + ": " + str(layerList['L'][l]))
                        loss_module = DreamLoss(*primary_params, **secondary_params)
                        net_base.add_module(str(len(net_base)), loss_module)
                        dream_losses.append(loss_module)
                        next_dream_idx += 1
                    l+=1

                if isinstance(layer, nn.Dropout):
                    net_base.add_module(str(len(net_base)), layer)

                    if layerList['D'][d] in dream_layers:
                        print("Setting up dream layer " + str(i) + ": " + str(layerList['D'][d]))
                        loss_module = DreamLoss(*primary_params, **secondary_params)
                        net_base.add_module(str(len(net_base)), loss_module)
                        dream_losses.append(loss_module)
                        next_dream_idx += 1
                    d+=1

                if use_classify > 0 and l == len(layerList['L']):
                    next_dream_idx += 1

    elif has_inception:
        start_net, tv_losses, l2_losses = start_network(start_params)
        lm_layer_names, loss_module_list = [], []
        net_base = copy.deepcopy(cnn)
        sn=0
        for i, n in enumerate(dream_layers):
            print("Setting up dream layer " + str(i+1) + ": " + n)
            if 'before_' in n:
                n = n.split('before_')[1]
                loss_module = DreamLossPreHook(*primary_params, **secondary_params)
                module_loc = 'before'
            else:
                loss_module = DreamLossHook(*primary_params, **secondary_params)
                module_loc = 'after'
            n = n.split('/')
            lm_layer_names.append(n)
            loss_module, sn = add_to_incept(net_base, n, sn, loss_module, module_loc)
            loss_module_list.append(loss_module)
            dream_losses.append(loss_module)
        if len(start_net) > 0:
            net_base = dream_utils.ModelPlus(start_net, net_base)
    return net_base, dream_losses, tv_losses, l2_losses, lm_layer_names, loss_module_list


# Create preprocessing model / start of VGG model
def start_network(dtype, random_transforms='none', jitter_val=32, tv_weight=0, l2_weight=0, layer_sigma=0):
    tv_losses, l2_losses = [], []
    start_net = nn.Sequential()
    if random_transforms != 'none':
        rt_mod = dream_utils.RandomTransform(random_transforms).type(dtype)
        start_net.add_module(str(len(start_net)), rt_mod)
    if jitter_val > 0:
        jitter_mod = dream_utils.Jitter(jitter_val).type(dtype)
        start_net.add_module(str(len(start_net)), jitter_mod)
    if tv_weight > 0:
        tv_mod = dream_loss_layers.TVLoss(tv_weight).type(dtype)
        start_net.add_module(str(len(start_net)), tv_mod)
        tv_losses.append(tv_mod)
    if l2_weight > 0:
        l2_mod = dream_loss_layers.L2Regularizer(l2_weight).type(dtype)
        start_net.add_module(str(len(start_net)), l2_mod)
        l2_losses.append(l2_mod)
    if layer_sigma > 0:
        gauss_mod = dream_utils.GaussianBlurLayer(5, layer_sigma).type(dtype)
        start_net.add_module(str(len(start_net)), gauss_mod)
    return start_net, tv_losses, l2_losses


# Reapply DreamLoss layer hooks
def renew_net(start_params, net, loss_module_list, dream_layers):
    start_net, tv_losses, l2_losses = start_network(*start_params)
    if isinstance(net, dream_utils.ModelPlus):
       net = net.net
    new_dream_losses = []
    sn=0
    for i, layer in enumerate(dream_layers):
        n = layer
        loss_module = loss_module_list[i]
        if str(loss_module).split('(')[0] == 'DreamLossPreHook':
            module_loc = 'before'
        else:
            module_loc = 'after'
        loss_module, sn = add_to_incept(net, n, sn, loss_module, module_loc)
        new_dream_losses.append(loss_module_list[i])
    if len(start_net) > 0:
        net = dream_utils.ModelPlus(start_net, net)
    return net, new_dream_losses, tv_losses, l2_losses