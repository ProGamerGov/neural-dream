def model_name_list():
    pytorch_names = ['vgg11-bbd30ac9.pth', 'vgg13-c768596a.pth', 'vgg16-397923af.pth' , \
    'vgg19-dcbb9e9d.pth', 'googlenet-1378be20.pth', 'inception_v3_google-1a9a5a14.pth']
    caffe_names = ['vgg16-00b39a1b.pth', 'vgg19-d01eb7cb.pth', 'nin_imagenet.pth', \
    'VGG16_SOD_finetune.pth', 'VGG16-Stylized-ImageNet.pth', 'vgg16_places365.pth', \
    'vgg16_hybrid1365.pth', 'fcn32s-heavy-pascal.pth', 'nyud-fcn32s-color-heavy.pth', \
    'pascalcontext-fcn32s-heavy.pth', 'siftflow-fcn32s-heavy.pth', 'channel_pruning.pth', \
    'googlenet_places205.pth', 'googlenet_places365.pth', 'resnet_50_1by2_nsfw.pth', \
    'bvlc_googlenet.pth', 'googlenet_finetune_web_cars.pth', 'googlenet_sos.pth', 'inception5h']
    return pytorch_names, caffe_names

# Automatically determine model type
def auto_model_mode(model_name):
    pytorch_names, caffe_names = model_name_list()
    if any(name.lower() in model_name.lower() for name in pytorch_names):
        input_mode = 'pytorch'
    elif any(name.lower() in model_name.lower() for name in caffe_names):
        input_mode = 'caffe'
    else:
        raise ValueError("Model not recognized, please manually specify the model type.")
    return input_mode


# Automatically determine preprocessing to use for model
def auto_mean(model_name, model_type):
    pytorch_names, caffe_names = model_name_list()
    if any(name.lower() in model_name.lower() for name in pytorch_names) or model_type == 'pytorch':
        input_mean = [0.485, 0.456, 0.406] # PyTorch Imagenet
    elif any(name.lower() in model_name.lower() for name in caffe_names) or model_type == 'caffe':
        input_mean = [103.939, 116.779, 123.68] # Caffe Imagenet
        if 'googlenet_places205.pth' in model_name.lower():
            input_mean = [105.417, 113.753, 116.047] # Caffe Places205
        elif 'googlenet_places365.pth' in model_name.lower():
            input_mean = [104.051, 112.514, 116.676] # Caffe Places365
        elif 'resnet_50_1by2_nsfw.pth' in model_name.lower():
            input_mean = [104, 117, 123] # Caffe Open NSFW
    else:
        raise ValueError("Model not recognized, please manually specify the model type or model mean.")
    return input_mean