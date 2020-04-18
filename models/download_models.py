import torch
import argparse
from os import path
from sys import version_info
from collections import OrderedDict
from torch.utils.model_zoo import load_url

if version_info[0] < 3:
    import urllib
else:
    import urllib.request


options_list = ['all', 'caffe-vgg16', 'caffe-vgg19', 'caffe-nin', 'caffe-googlenet-places205', 'caffe-googlenet-places365', 'caffe-googlenet-bvlc', 'caffe-googlenet-cars', 'caffe-googlenet-sos', \
                'caffe-resnet-opennsfw', 'pytorch-vgg16', 'pytorch-vgg19', 'pytorch-googlenet', 'pytorch-inceptionv3', 'tensorflow-inception5h', 'keras-inceptionv3', 'all-caffe', 'all-caffe-googlenet']


def main():
    params = params_list()
    if params.models == 'all':
        params.models = options_list[1:16]
    elif 'all-caffe' in params.models and 'all-caffe-googlenet' not in params.models:
        params.models = options_list[1:10] + params.models.split(',')
    elif 'all-caffe-googlenet' in params.models:
        params.models = options_list[4:9] + params.models.split(',')
    else:
        params.models = params.models.split(',')

    if 'caffe-vgg19' in params.models:
        # Download the VGG-19 ILSVRC model and fix the layer names
        print("Downloading the VGG-19 ILSVRC model")
        sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth")
        map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
        sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
        torch.save(sd, path.join(params.download_path, "vgg19-d01eb7cb.pth"))

    if 'caffe-vgg16' in params.models:
        # Download the VGG-16 ILSVRC model and fix the layer names
        print("Downloading the VGG-16 ILSVRC model")
        sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth")
        map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
        sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
        torch.save(sd, path.join(params.download_path, "vgg16-00b39a1b.pth"))

    if 'caffe-nin' in params.models:
        # Download the NIN model
        print("Downloading the NIN model")
        fileurl = "https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth"
        name = "nin_imagenet.pth"
        download_file(fileurl, name, params.download_path)

    if 'caffe-googlenet-places205' in params.models:
        # Download the Caffe GoogeLeNet Places205 model
        print("Downloading the Places205 GoogeLeNet model")
        fileurl = "https://github.com/ProGamerGov/pytorch-places/raw/master/googlenet_places205.pth"
        name = "googlenet_places205.pth"
        download_file(fileurl, name, params.download_path)

    if 'caffe-googlenet-places365' in params.models:
        # Download the Caffe GoogeLeNet Places365 model
        print("Downloading the Places365 GoogeLeNet model")
        fileurl = "https://github.com/ProGamerGov/pytorch-places/raw/master/googlenet_places365.pth"
        name = "googlenet_places365.pth"
        download_file(fileurl, name, params.download_path)

    if 'caffe-googlenet-bvlc' in params.models:
        # Download the Caffe BVLC GoogeLeNet model
        print("Downloading the BVLC GoogeLeNet model")
        fileurl = "https://github.com/ProGamerGov/pytorch-old-caffemodels/raw/master/bvlc_googlenet.pth"
        name = "bvlc_googlenet.pth"
        download_file(fileurl, name, params.download_path)

    if 'caffe-googlenet-cars' in params.models:
        # Download the Caffe GoogeLeNet Cars model
        print("Downloading the Cars GoogeLeNet model")
        fileurl = "https://github.com/ProGamerGov/pytorch-old-caffemodels/raw/master/googlenet_finetune_web_cars.pth"
        name = "googlenet_finetune_web_cars.pth"
        download_file(fileurl, name, params.download_path)

    if 'caffe-googlenet-sos' in params.models:
        # Download the Caffe GoogeLeNet SOS model
        print("Downloading the SOS GoogeLeNet model")
        fileurl = "https://github.com/ProGamerGov/pytorch-old-caffemodels/raw/master/GoogleNet_SOS.pth"
        name = "GoogleNet_SOS.pth"
        download_file(fileurl, name, params.download_path)

    if 'pytorch-vgg19' in params.models:
        # Download the PyTorch VGG19 model
        print("Downloading the PyTorch VGG 19 model")
        fileurl = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
        name = "vgg19-dcbb9e9d.pth"
        download_file(fileurl, name, params.download_path)

    if 'pytorch-vgg16' in params.models:
        # Download the PyTorch VGG16 model
        print("Downloading the PyTorch VGG 16 model")
        fileurl = "https://download.pytorch.org/models/vgg16-397923af.pth"
        name = "vgg16-397923af.pth"
        download_file(fileurl, name, params.download_path)

    if 'pytorch-googlenet' in params.models:
        # Download the PyTorch GoogLeNet model
        print("Downloading the PyTorch GoogLeNet model")
        fileurl = "https://download.pytorch.org/models/googlenet-1378be20.pth"
        name = "googlenet-1378be20.pth"
        download_file(fileurl, name, params.download_path)

    if 'pytorch-inception' in params.models:
        # Download the PyTorch Inception V3 model
        print("Downloading the PyTorch Inception V3 model")
        fileurl = "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
        name = "inception_v3_google-1a9a5a14.pth"
        download_file(fileurl, name, params.download_path)

    if 'tensorflow-inception5h' in params.models:
        # Download the Inception5h model
        print("Downloading the TensorFlow Inception5h model")
        fileurl = "https://github.com/ProGamerGov/pytorch-old-tensorflow-models/raw/master/inception5h.pth"
        name = "inception5h.pth"
        download_file(fileurl, name, params.download_path)

    if 'keras-inceptionv3' in params.models:
        # Download the Keras Inception V3 model
        print("Downloading the Keras Inception V3 model")
        fileurl = "https://github.com/ProGamerGov/pytorch-old-tensorflow-models/raw/master/inceptionv3_keras.pth"
        name = "inceptionv3_keras.pth"
        download_file(fileurl, name, params.download_path)

    if 'caffe-resnet-opennsfw' in params.models:
        # Download the ResNet Yahoo Open NSFW model
        print("Downloading the ResNet Yahoo Open NSFW model")
        fileurl = "https://github.com/ProGamerGov/pytorch-old-caffemodels/raw/master/ResNet_50_1by2_nsfw.pth"
        name = "ResNet_50_1by2_nsfw.pth"
        download_file(fileurl, name, params.download_path)

    print("All selected models have been successfully downloaded")


def params_list():
    parser = argparse.ArgumentParser()
    parser.add_argument("-models", help="Models to download", default='caffe-googlenet-bvlc,caffe-nin', action=MultipleChoice)
    parser.add_argument("-download_path", help="Download location for models", default='models')
    params = parser.parse_args()
    return params


def download_file(fileurl, name, download_path):
    if version_info[0] < 3:
        urllib.URLopener().retrieve(fileurl, path.join(download_path, name))
    else:
        urllib.request.urlretrieve(fileurl, path.join(download_path, name))


class MultipleChoice(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        self.options = options_list
        e = [o.lower() for o in values.split(',') if o.lower() not in self.options]
        if len(e) > 0:
            raise argparse.ArgumentError(self, 'invalid choices: ' + ','.join([str(v) for v in e]) +
                                         ' (choose from ' + ','.join([ "'"+str(v)+"'" for v in self.options])+')')
        setattr(namespace, self.dest, values)



if __name__ == "__main__":
    main()
