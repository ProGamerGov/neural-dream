import torch
import torch.nn as nn
import torchvision
from torchvision import models
from neural_dream.models import *


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


class VGG_SOD(nn.Module):
    def __init__(self, features, num_classes=100):
        super(VGG_SOD, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 100),
        )


class VGG_FCN32S(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_FCN32S, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Conv2d(512,4096,(7, 7)),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(4096,4096,(1, 1)),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )


class VGG_PRUNED(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_PRUNED, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )


class NIN(nn.Module):
    def __init__(self, pooling):
        super(NIN, self).__init__()
        if pooling == 'max':
            pool2d = nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        elif pooling == 'avg':
            pool2d = nn.AvgPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)

        self.features = nn.Sequential(
            nn.Conv2d(3,96,(11, 11),(4, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,(1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,(1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,(1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Dropout(0.5),
            nn.Conv2d(384,1024,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1024,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1000,(1, 1)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((6, 6),(1, 1),(0, 0),ceil_mode=True),
            nn.Softmax(),
        )



class ModelParallel(nn.Module):
    def __init__(self, net, device_ids, device_splits):
        super(ModelParallel, self).__init__()
        self.device_list = self.name_devices(device_ids.split(','))
        self.chunks = self.chunks_to_devices(self.split_net(net, device_splits.split(',')))

    def name_devices(self, input_list):
        device_list = []
        for i, device in enumerate(input_list):
            if str(device).lower() != 'c':
                device_list.append("cuda:" + str(device))
            else:
                device_list.append("cpu")
        return device_list

    def split_net(self, net, device_splits):
        chunks, cur_chunk = [], nn.Sequential()
        for i, l in enumerate(net):
            cur_chunk.add_module(str(i), net[i])
            if str(i) in device_splits and device_splits != '':
                del device_splits[0]
                chunks.append(cur_chunk)
                cur_chunk = nn.Sequential()
        chunks.append(cur_chunk)
        return chunks

    def chunks_to_devices(self, chunks):
        for i, chunk in enumerate(chunks):
            chunk.to(self.device_list[i])
        return chunks

    def c(self, input, i):
        if input.type() == 'torch.FloatTensor' and 'cuda' in self.device_list[i]:
            input = input.type('torch.cuda.FloatTensor')
        elif input.type() == 'torch.cuda.FloatTensor' and 'cpu' in self.device_list[i]:
            input = input.type('torch.FloatTensor')
        return input

    def forward(self, input):
        for i, chunk in enumerate(self.chunks):
            if i < len(self.chunks) -1:
                input = self.c(chunk(self.c(input, i).to(self.device_list[i])), i+1).to(self.device_list[i+1])
            else:
                input = chunk(input)
        return input



def buildSequential(channel_list, pooling):
    layers = []
    in_channels = 3
    if pooling == 'max':
        pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    elif pooling == 'avg':
        pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
    else:
        raise ValueError("Unrecognized pooling parameter")
    for c in channel_list:
        if c == 'P':
            layers += [pool2d]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


channel_list = {
'VGG-11': [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
'VGG-13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
'VGG-16p': [24, 22, 'P', 41, 51, 'P', 108, 89, 111, 'P', 184, 276, 228, 'P', 512, 512, 512, 'P'],
'VGG-16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
'VGG-19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'],
}

nin_dict = {
'C': ['conv1', 'cccp1', 'cccp2', 'conv2', 'cccp3', 'cccp4', 'conv3', 'cccp5', 'cccp6', 'conv4-1024', 'cccp7-1024', 'cccp8-1024'],
'R': ['relu0', 'relu1', 'relu2', 'relu3', 'relu5', 'relu6', 'relu7', 'relu8', 'relu9', 'relu10', 'relu11', 'relu12'],
'P': ['pool1', 'pool2', 'pool3', 'pool4'],
'D': ['drop'],
}
vgg11_dict = {
'C': ['conv1_1', 'conv2_1', 'conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv5_1', 'conv5_2'],
'R': ['relu1_1', 'relu2_1', 'relu3_1', 'relu3_2', 'relu4_1', 'relu4_2', 'relu5_1', 'relu5_2', 'relu6', 'relu7'],
'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
'L': ['fc6', 'fc7', 'fc8'],
'D': ['drop6', 'drop7'],
}
vgg13_dict = {
'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv5_1', 'conv5_2'],
'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu4_1', 'relu4_2', 'relu5_1', 'relu5_2', 'relu6', 'relu7'],
'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
'L': ['fc6', 'fc7', 'fc8'],
'D': ['drop6', 'drop7'],
}
vgg16_dict = {
'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'],
'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3', 'relu6', 'relu7'],
'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
'L': ['fc6', 'fc7', 'fc8'],
'D': ['drop6', 'drop7'],
}
vgg19_dict = {
'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'],
'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4', 'relu6', 'relu7'],
'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
'L': ['fc6', 'fc7', 'fc8'],
'D': ['drop6', 'drop7'],
}
name_dict = {
'vgg': ['vgg'],
'vgg11': ['vgg-11', 'vgg11', 'vgg_11'],
'vgg13': ['vgg-13', 'vgg13', 'vgg_13'],
'vgg16': ['vgg-16', 'vgg16', 'vgg_16', 'fcn32s', 'pruning', 'sod'],
'vgg19': ['vgg-19', 'vgg19', 'vgg_19',],
}
ic_dict = {
'inception': ['inception'],
'googlenet': ['googlenet'],
'inceptionv3': ['inception_v3', 'inceptionv3'],
'resnet': ['resnet'],
}



def build_googlenet_list(cnn):
    main_layers = ['conv1', 'maxpool1', 'conv2', 'conv3', 'maxpool2', 'inception3a', 'inception3b', 'maxpool3', \
    'inception4a', 'inception4b', 'inception4c', 'inception4d', 'inception4e', 'maxpool4', 'inception5a', \
    'inception5b', 'aux1', 'aux2', 'avgpool', 'dropout', 'fc']
    branch_list = ['branch1', 'branch2', 'branch3', 'branch4']
    ax = ['conv', 'fc1', 'fc2']
    conv_block =['conv', 'bn']

    layer_name_list = []

    for i, layer in enumerate(list(cnn.children())):
        if 'BasicConv2d' in str(type(layer)):
            for bl, block in enumerate(list(layer.children())):
                name = main_layers[i] + '/' + conv_block[bl]
                layer_name_list.append(name)
        elif 'Inception' in str(type(layer)) and 'Aux' not in str(type(layer)):
            for br, branch in enumerate(list(layer.children())):
                for bl, block in enumerate(list(branch.children())):
                    name = main_layers[i] + '/' + branch_list[br] + '/' + conv_block[bl]
                    layer_name_list.append(name)
        elif 'Inception' in str(type(layer)) and 'Aux' in str(type(layer)):
             for bl, block in enumerate(list(layer.children())):
                 name = main_layers[i] + '/' + ax[bl]
                 layer_name_list.append(name)
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d) \
        or isinstance(layer, nn.Dropout) or isinstance(layer, nn.Linear):
            layer_name_list.append(main_layers[i])
    return layer_name_list


def build_inceptionv3_list(cnn):
    main_layers = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'Mixed_5b', \
    'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e', 'AuxLogits', 'Mixed_7a', \
    'Mixed_7b', 'Mixed_7c', 'fc']
    ba = ['branch1x1', 'branch5x5_1', 'branch5x5_2', 'branch3x3dbl_1', 'branch3x3dbl_2', 'branch3x3dbl_3', 'branch_pool']
    bb = ['branch3x3', 'branch3x3dbl_1', 'branch3x3dbl_2', 'branch3x3dbl_3']
    bc = ['branch1x1', 'branch7x7_1', 'branch7x7_2', 'branch7x7_3', 'branch7x7dbl_1', 'branch7x7dbl_2', 'branch7x7dbl_3', \
    'branch7x7dbl_4', 'branch7x7dbl_5', 'branch_pool']
    bd = ['branch3x3_1', 'branch3x3_2', 'branch7x7x3_1', 'branch7x7x3_2', 'branch7x7x3_3', 'branch7x7x3_4']
    be = ['branch1x1', 'branch3x3_1', 'branch3x3_2a', 'branch3x3_2b', 'branch3x3dbl_1', 'branch3x3dbl_2', \
    'branch3x3dbl_3a', 'branch3x3dbl_3b', 'branch_pool']
    ax = ['conv0', 'conv1', 'fc']
    conv_block =['conv', 'bn']

    layer_name_list = []

    for i, layer in enumerate(list(cnn.children())):
        if 'BasicConv2d' in str(type(layer)):
            for bl, block in enumerate(list(layer.children())):
                name = main_layers[i] + '/' + conv_block[bl]
                layer_name_list.append(name)
        elif 'Inception' in str(type(layer)) and 'Aux' not in str(type(layer)):
            if 'InceptionA' in str(type(layer)):
                branch_list = ba
            elif 'InceptionB' in str(type(layer)):
                branch_list = bb
            elif 'InceptionC' in str(type(layer)):
                branch_list = bc
            elif 'InceptionD' in str(type(layer)):
                branch_list = bd
            elif 'InceptionE' in str(type(layer)):
                branch_list = be
            for br, branch in enumerate(list(layer.children())):
                for bl, block in enumerate(list(branch.children())):
                    name = main_layers[i] + '/' + branch_list[br] + '/' + conv_block[bl]
                    layer_name_list.append(name)
        elif 'Inception' in str(type(layer)) and 'Aux' in str(type(layer)):
             for bl, block in enumerate(list(layer.children())):
                 name = main_layers[i] + '/' + ax[bl]
                 layer_name_list.append(name)
        elif isinstance(layer, nn.Linear):
            layer_name_list.append(main_layers[i])
    return layer_name_list



def modelSelector(model_file, pooling):
    if any(name in model_file for name in name_dict):
        if any(name in model_file for name in name_dict['vgg16']):
            print("VGG-16 Architecture Detected")
            if "pruning" in model_file:
                print("Using The Channel Pruning Model")
                cnn, layerList = VGG_PRUNED(buildSequential(channel_list['VGG-16p'], pooling)), vgg16_dict
            elif "fcn32s" in model_file:
                print("Using the fcn32s-heavy-pascal Model")
                cnn, layerList = VGG_FCN32S(buildSequential(channel_list['VGG-16'], pooling)), vgg16_dict
                layerList['C'] = layerList['C'] + layerList['L']
            elif "sod" in model_file:
                print("Using The SOD Fintune Model")
                cnn, layerList = VGG_SOD(buildSequential(channel_list['VGG-16'], pooling)), vgg16_dict
            elif "16" in model_file:
                cnn, layerList = VGG(buildSequential(channel_list['VGG-16'], pooling)), vgg16_dict
        elif any(name in model_file for name in name_dict['vgg19']):
            print("VGG-19 Architecture Detected")
            if "19" in model_file:
                cnn, layerList = VGG(buildSequential(channel_list['VGG-19'], pooling)), vgg19_dict
        elif any(name in model_file for name in name_dict['vgg13']):
            print("VGG-13 Architecture Detected")
            cnn, layerList = VGG(buildSequential(channel_list['VGG-13'], pooling)), vgg13_dict
        elif any(name in model_file for name in name_dict['vgg11']):
            print("VGG-11 Architecture Detected")
            cnn, layerList = VGG(buildSequential(channel_list['VGG-11'], pooling)), vgg11_dict
        else:
            raise ValueError("VGG architecture not recognized.")
    elif "googlenet" in model_file:
         print("GoogLeNet Architecture Detected")
         if '205' in model_file:
             cnn, layerList = GoogLeNetPlaces(205), googlenet_layer_names('places')
         elif '365' in model_file:
             cnn, layerList = GoogLeNetPlaces(365), googlenet_layer_names('places')
         elif 'bvlc' in model_file:
             cnn, layerList = BVLC_GOOGLENET(), googlenet_layer_names('bvlc')
         elif 'cars' in model_file:
             cnn, layerList = GOOGLENET_CARS(), googlenet_layer_names('cars')
         elif 'sos' in model_file:
             cnn, layerList = GoogleNet_SOS(), googlenet_layer_names('sos')
         else:
             cnn, layerList = models.googlenet(pretrained=False, transform_input=False), ''
    elif "inception" in model_file:
         print("Inception Architecture Detected")
         if 'inception5h' in model_file:
             cnn, layerList = Inception5h(), inception_layer_names('5h')
         elif 'keras' in model_file:
             cnn, layerList = InceptionV3Keras(), inception_layer_names('keras-wtop')
         else:
             cnn, layerList = models.inception_v3(pretrained=False, transform_input=False), ''
    elif "resnet" in model_file:
        print("ResNet Architecture Detected")
        if 'resnet_50_1by2_nsfw' in model_file:
            cnn, layerList = ResNet_50_1by2_nsfw(), resnet_layer_names
        else:
            raise ValueError("ResNet architecture not recognized.")
    elif "nin" in model_file:
        print("NIN Architecture Detected")
        cnn, layerList = NIN(pooling), nin_dict
    else:
        raise ValueError("Model architecture not recognized.")
    return cnn, layerList


# Print like Torch7/loadcaffe
def print_loadcaffe(cnn, layerList):
    c = 0
    for l in list(cnn):
         if "Conv2d" in str(l) and "Basic" not in str(l):
             in_c, out_c, ks  = str(l.in_channels), str(l.out_channels), str(l.kernel_size)
             print(layerList['C'][c] +": " +  (out_c + " " + in_c + " " + ks).replace(")",'').replace("(",'').replace(",",'') )
             c+=1
         if c == len(layerList['C']):
             break


class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input, 1)


def add_classifier_layers(cnn, pooling='avg'):
    new_cnn, cnn_classifier = cnn.features, cnn.classifier
    if 'avg' in pooling:
        adaptive_pool2d = nn.AdaptiveAvgPool2d((7, 7))
    elif 'max' in pooling:
        adaptive_pool2d = nn.AdaptiveMaxPool2d((7, 7))
    new_cnn.add_module(str(len(new_cnn)), adaptive_pool2d)

    if not isinstance(cnn, VGG_FCN32S):
       flatten_layer = Flatten()
       new_cnn.add_module(str(len(new_cnn)), flatten_layer)

    for layer in cnn_classifier:
        new_cnn.add_module(str(len(new_cnn)), layer)
    return new_cnn


# Load the model, and configure pooling layer type
def loadCaffemodel(model_file, pooling, use_gpu, disable_check, add_classifier=False):
    cnn, layerList = modelSelector(str(model_file).lower(), pooling)

    cnn.load_state_dict(torch.load(model_file), strict=(not disable_check))
    print("Successfully loaded " + str(model_file))

    # Maybe convert the model to cuda now, to avoid later issues
    if "c" not in str(use_gpu).lower() or "c" not in str(use_gpu[0]).lower():
        cnn = cnn.cuda()

    if not isinstance(cnn, NIN) and not any(name in model_file.lower() for name in ic_dict) and add_classifier:
        cnn, has_inception = add_classifier_layers(cnn, pooling), False
    elif any(name in model_file.lower() for name in ic_dict['googlenet']):
        if '205' in model_file or '365' in model_file:
            has_inception = True
        elif 'cars' in model_file.lower() or 'sos' in model_file.lower() or 'bvlc' in model_file.lower():
            has_inception = True
        else:
            layerList, has_inception = build_googlenet_list(cnn), True
    elif any(name in model_file.lower() for name in ic_dict['inceptionv3']) and 'keras' not in model_file.lower():
        layerList, has_inception = build_inceptionv3_list(cnn), True
    elif 'inception5h' in model_file.lower() or 'resnet' in model_file.lower() or 'keras' in model_file.lower():
        has_inception = True
    else:
        cnn, has_inception = cnn.features, False

    if has_inception:
        cnn.eval()
        cnn.has_inception = True
    else:
        cnn.has_inception = False
    if not any(name in model_file.lower() for name in ic_dict):
        print_loadcaffe(cnn, layerList)

    try:
        cnn.add_layers()
    except:
        pass
    return cnn, layerList