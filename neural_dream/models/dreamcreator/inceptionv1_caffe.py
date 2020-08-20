# Model from: https://github.com/ProGamerGov/dream-creator/blob/master/utils/inceptionv1_caffe.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F



# Build the model definition
def build_model(model_file='pt_places365.pth', num_classes=120, load_branches=True):
    base_list = {'pt_places365.pth': 365, 'pt_inception5h.pth': 1008, 'pt_bvlc.pth': 1000}
    base_name = os.path.basename(model_file)
    if base_name.lower() in base_list:
        load_classes = base_list[base_name.lower()]
    else:
        load_classes = num_classes

    cnn = InceptionV1_Caffe(load_classes, mode='bvlc', load_branches=load_branches)

    if base_name.lower() not in base_list:
        cnn.replace_fc(load_classes, load_branches)
    return cnn


def load_model(model_file, num_classes=120, has_branches=True):
    checkpoint = torch.load(model_file, map_location='cpu')

    # Attempt to load model class info
    try:
        loaded_classes = checkpoint['num_classes']
    except:
        loaded_classes = None
    num_classes = num_classes if loaded_classes == None else loaded_classes

    try:
        norm_vals = checkpoint['normalize_params']
    except:
        norm_vals = None
    try:
        h_branches = checkpoint['has_branches']
    except:
        h_branches = None
    has_branches = has_branches if h_branches == None else h_branches

    cnn = build_model(model_file, num_classes, load_branches=has_branches)

    if type(checkpoint) == dict:
        model_keys = checkpoint.keys()
        cnn.load_state_dict(checkpoint['model_state_dict'])
    else:
        cnn.load_state_dict(checkpoint)
    return cnn, norm_vals, num_classes


# RedirectedReLU autograd function
class RedirectedReLU(torch.autograd.Function):
    """
    https://github.com/greentfrapp/lucent/blob/master/lucent/modelzoo/inceptionv1/helper_layers.py#L61

    A workaround when there is no gradient flow from an initial random input
    See https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py
    Note: this means that the gradient is technically "wrong"
    TODO: the original Lucid library has a more sophisticated way of doing this
    """
    @staticmethod
    def forward(self, input_tensor):
        self.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)
    @staticmethod
    def backward(self, grad_output):
        input_tensor, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor < 0] = grad_input[input_tensor < 0] * 1e-1
        return grad_input


# Basic ReLU layer
class ReluLayer(nn.Module):
    def forward(self, input):
        return F.relu(input, inplace=True)


# RedirectedReLU layer
class RedirectedReluLayer(nn.Module):
    def forward(self, input):
        if F.relu(input.detach().sum()) != 0:
            return F.relu(input, inplace=True)
        else:
            return RedirectedReLU.apply(input)


# Replace all ReLU layers with RedirectedReLU
def relu_to_redirected_relu(model):
    for name, child in model.named_children():
        if isinstance(child, ReluLayer):
            setattr(model, name, RedirectedReluLayer())
        else:
            relu_to_redirected_relu(child)


# Basic Local Response Norm layer
class LocalResponseNormLayer(nn.Module):
    def __init__(self, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1):
        super(LocalResponseNormLayer, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return F.local_response_norm(input, size=self.size, alpha=self.alpha, beta=self.beta, k=self.k)


# Better version of Inception V1/GoogleNet for Inception5h and Places models
class InceptionV1_Caffe(nn.Module):

    def __init__(self, out_features=1000, load_branches=True, mode='bvlc'):
        super(InceptionV1_Caffe, self).__init__()
        self.mode = mode
        self.use_branches = load_branches

        if self.mode == 'p365' or self.mode == 'bvlc':
            lrn_vals = (5, 9.999999747378752e-05, 0.75, 1)
            diff_channels = [208, 512, 32]
        elif self.mode == '5h':
            lrn_vals = (9, 9.99999974738e-05, 0.5, 1)
            diff_channels = [204, 508, 48]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), groups=1, bias=True)
        self.conv1_relu = ReluLayer()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.localresponsenorm1 = LocalResponseNormLayer(*lrn_vals)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2_relu = ReluLayer()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), groups=1, bias=True)
        self.conv3_relu = ReluLayer()
        self.localresponsenorm2 = LocalResponseNormLayer(*lrn_vals)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.mixed3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.mixed3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.mixed4a = InceptionModule(480, 192, 96, diff_channels[0], 16, 48, 64)
        self.mixed4b = InceptionModule(diff_channels[1], 160, 112, 224, 24, 64, 64)
        self.mixed4c = InceptionModule(512,128, 128, 256, 24, 64, 64)
        self.mixed4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.mixed4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.mixed5a = InceptionModule(832, 256, 160, 320, diff_channels[2], 128, 128)
        self.mixed5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.4000000059604645)
        if self.mode == '5h':
            self.fc = nn.Linear(1024, out_features)

        if load_branches:
            self.aux1 = AuxBranch(512, out_features)
            self.aux2 = AuxBranch(528, out_features)


    # Add branches if they did not previously exist
    def add_branches(self, out_features):
        self.use_branches = True
        ch = 512 if self.mode == 'p365' or self.mode == 'bvlc' else 508
        self.aux1 = AuxBranch(ch, out_features)
        self.aux2 = AuxBranch(528, out_features)

    # Remove branches from model
    def remove_branches(self):
        self.use_branches = False
        del self.aux1
        del self.aux2

    # Initialize weights
    def initialize_layers(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0)

    # Replace FC layers for new data set
    def replace_fc(self, out_features, use_branches):
        self.fc = nn.Linear(1024, out_features)
        self.fc.apply(self.initialize_layers)
        if use_branches or self.use_branches:
            self.use_branches = True
            self.aux1.loss_classifier = nn.Linear(in_features = 1024, out_features = out_features, bias = True)
            self.aux2.loss_classifier = nn.Linear(in_features = 1024, out_features = out_features, bias = True)
            self.aux1.loss_classifier.apply(self.initialize_layers)
            self.aux2.loss_classifier.apply(self.initialize_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = F.pad(x, (0, 1, 0, 1), value=float('-inf'))
        x = self.pool1(x)
        x = self.localresponsenorm1(x)

        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv3(x)
        x = self.conv3_relu(x)
        x = self.localresponsenorm2(x)

        x = F.pad(x, (0, 1, 0, 1), value=float('-inf'))
        x = self.pool2(x)
        x = self.mixed3a(x)
        x = self.mixed3b(x)
        x = F.pad(x, (0, 1, 0, 1), value=float('-inf'))
        x = self.pool3(x)
        x = self.mixed4a(x)

        if self.use_branches:
            aux1_output = self.aux1(x)

        x = self.mixed4b(x)
        x = self.mixed4c(x)
        x = self.mixed4d(x)

        if self.use_branches:
            aux2_output = self.aux2(x)

        x = self.mixed4e(x)
        x = F.pad(x, (0, 1, 0, 1), value=float('-inf'))
        x = self.pool4(x)
        x = self.mixed5a(x)
        x = self.mixed5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)

        if not self.use_branches:
            return x
        else:
            return x, aux1_output, aux2_output


class InceptionModule(nn.Module):

    def __init__(self, in_channels, c1x1, c3x3reduce, c3x3, c5x5reduce, c5x5, pool_proj):
        super(InceptionModule, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=c1x1, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_1x1_relu = ReluLayer()

        self.conv_3x3_reduce = nn.Conv2d(in_channels=in_channels, out_channels=c3x3reduce, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_3x3_reduce_relu = ReluLayer()
        self.conv_3x3 = nn.Conv2d(in_channels=c3x3reduce, out_channels=c3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), groups=1, bias=True)
        self.conv_3x3_relu = ReluLayer()

        self.conv_5x5_reduce = nn.Conv2d(in_channels=in_channels, out_channels=c5x5reduce, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_5x5_reduce_relu = ReluLayer()
        self.conv_5x5 = nn.Conv2d(in_channels=c5x5reduce, out_channels=c5x5, kernel_size=(5, 5), stride=(1, 1), padding=(2,2), groups=1, bias=True)
        self.conv_5x5_relu = ReluLayer()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self.pool_proj = nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.pool_proj_relu = ReluLayer()

    def forward(self, x):
        c1x1 = self.conv_1x1(x)
        c1x1 = self.conv_1x1_relu(c1x1)

        c3x3 = self.conv_3x3_reduce(x)
        c3x3 = self.conv_3x3_reduce_relu(c3x3)
        c3x3 = self.conv_3x3(c3x3)
        c3x3 = self.conv_3x3_relu(c3x3)

        c5x5 = self.conv_5x5_reduce(x)
        c5x5 = self.conv_5x5_reduce_relu(c5x5)
        c5x5 = self.conv_5x5(c5x5)
        c5x5 = self.conv_5x5_relu(c5x5)

        px = self.pool_proj(x)
        px = self.pool_proj_relu(px)
        px = F.pad(px, (1, 1, 1, 1), value=float('-inf'))
        px = self.pool(px)
        return torch.cat([c1x1, c3x3, c5x5, px], dim=1)


class AuxBranch(nn.Module):

    def __init__(self, in_channels=512, out_features=1000):
        super(AuxBranch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.loss_conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.loss_conv_relu = ReluLayer()
        self.loss_fc = nn.Linear(in_features = 2048, out_features = 1024, bias = True)
        self.loss_fc_relu = ReluLayer()
        self.loss_dropout = nn.Dropout(0.699999988079071)
        self.loss_classifier = nn.Linear(in_features = 1024, out_features = out_features, bias = True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.loss_conv(x)
        x = self.loss_conv_relu(x)
        x = torch.flatten(x, 1)
        x = self.loss_fc(x)
        x = self.loss_fc_relu(x)
        x = self.loss_dropout(x)
        x = self.loss_classifier(x)
        return x
