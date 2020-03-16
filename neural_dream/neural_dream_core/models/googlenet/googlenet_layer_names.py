def googlenet_layer_names(model_name='places'):
    if 'places' in model_name:
        hookable_layers = ['conv1_7x7_s2', 'conv1_relu_7x7', 'pool1_3x3_s2', 'conv2_3x3_reduce', 'conv2_relu_3x3_reduce', 'conv2_3x3', 'conv2_relu_3x3', 'pool2_3x3_s2', \
        'inception_3a_pool', 'inception_3a_1x1', 'inception_3a_5x5_reduce', 'inception_3a_3x3_reduce', 'inception_3a_pool_proj', 'inception_3a_relu_1x1', \
        'inception_3a_relu_5x5_reduce', 'inception_3a_relu_3x3_reduce', 'inception_3a_relu_pool_proj', 'inception_3a_5x5', 'inception_3a_3x3', \
        'inception_3a_relu_5x5', 'inception_3a_relu_3x3', 'inception_3a_output', 'inception_3b_3x3_reduce', 'inception_3b_pool', 'inception_3b_1x1', \
        'inception_3b_5x5_reduce', 'inception_3b_relu_3x3_reduce', 'inception_3b_pool_proj', 'inception_3b_relu_1x1', 'inception_3b_relu_5x5_reduce', \
        'inception_3b_3x3', 'inception_3b_relu_pool_proj', 'inception_3b_5x5', 'inception_3b_relu_3x3', 'inception_3b_relu_5x5', 'inception_3b_output', \
        'pool3_3x3_s2', 'inception_4a_1x1', 'inception_4a_3x3_reduce', 'inception_4a_5x5_reduce', 'inception_4a_pool', 'inception_4a_relu_1x1', \
        'inception_4a_relu_3x3_reduce', 'inception_4a_relu_5x5_reduce', 'inception_4a_pool_proj', 'inception_4a_3x3', 'inception_4a_5x5', \
        'inception_4a_relu_pool_proj', 'inception_4a_relu_3x3', 'inception_4a_relu_5x5', 'inception_4a_output', 'inception_4b_pool', 'inception_4b_5x5_reduce', \
        'inception_4b_1x1', 'inception_4b_3x3_reduce', 'inception_4b_pool_proj', 'inception_4b_relu_5x5_reduce', 'inception_4b_relu_1x1', \
        'inception_4b_relu_3x3_reduce', 'inception_4b_relu_pool_proj', 'inception_4b_5x5', 'inception_4b_3x3', 'inception_4b_relu_5x5', 'inception_4b_relu_3x3', \
        'inception_4b_output', 'inception_4c_5x5_reduce', 'inception_4c_pool', 'inception_4c_1x1', 'inception_4c_3x3_reduce', 'inception_4c_relu_5x5_reduce', \
        'inception_4c_pool_proj', 'inception_4c_relu_1x1', 'inception_4c_relu_3x3_reduce', 'inception_4c_5x5', 'inception_4c_relu_pool_proj', 'inception_4c_3x3', \
        'inception_4c_relu_5x5', 'inception_4c_relu_3x3', 'inception_4c_output', 'inception_4d_pool', 'inception_4d_3x3_reduce', 'inception_4d_1x1', \
        'inception_4d_5x5_reduce', 'inception_4d_pool_proj', 'inception_4d_relu_3x3_reduce', 'inception_4d_relu_1x1', 'inception_4d_relu_5x5_reduce', \
        'inception_4d_relu_pool_proj', 'inception_4d_3x3', 'inception_4d_5x5', 'inception_4d_relu_3x3', 'inception_4d_relu_5x5', 'inception_4d_output', \
        'inception_4e_1x1', 'inception_4e_5x5_reduce', 'inception_4e_3x3_reduce', 'inception_4e_pool', 'inception_4e_relu_1x1', 'inception_4e_relu_5x5_reduce', \
        'inception_4e_relu_3x3_reduce', 'inception_4e_pool_proj', 'inception_4e_5x5', 'inception_4e_3x3', 'inception_4e_relu_pool_proj', 'inception_4e_relu_5x5', \
        'inception_4e_relu_3x3', 'inception_4e_output', 'pool4_3x3_s2', 'inception_5a_1x1', 'inception_5a_5x5_reduce', 'inception_5a_pool', 'inception_5a_3x3_reduce', \
        'inception_5a_relu_1x1', 'inception_5a_relu_5x5_reduce', 'inception_5a_pool_proj', 'inception_5a_relu_3x3_reduce', 'inception_5a_5x5', 'inception_5a_relu_pool_proj', \
        'inception_5a_3x3', 'inception_5a_relu_5x5', 'inception_5a_relu_3x3', 'inception_5a_output', 'inception_5b_3x3_reduce', 'inception_5b_pool', 'inception_5b_5x5_reduce', \
        'inception_5b_1x1', 'inception_5b_relu_3x3_reduce', 'inception_5b_pool_proj', 'inception_5b_relu_5x5_reduce', 'inception_5b_relu_1x1', 'inception_5b_3x3', \
        'inception_5b_relu_pool_proj', 'inception_5b_5x5', 'inception_5b_relu_3x3', 'inception_5b_relu_5x5', 'inception_5b_output', 'pool5_drop_7x7_s1'] 
    elif 'bvlc' in model_name:
        hookable_layers = ['conv1_7x7_s2', 'conv1_relu_7x7', 'pool1_3x3_s2', 'conv2_3x3_reduce', 'conv2_relu_3x3_reduce', 'conv2_3x3', 'conv2_relu_3x3', 'pool2_3x3_s2', \
        'inception_3a_pool', 'inception_3a_1x1', 'inception_3a_5x5_reduce', 'inception_3a_3x3_reduce', 'inception_3a_pool_proj', 'inception_3a_relu_1x1', \
        'inception_3a_relu_5x5_reduce', 'inception_3a_relu_3x3_reduce', 'inception_3a_relu_pool_proj', 'inception_3a_5x5', 'inception_3a_3x3', \
        'inception_3a_relu_5x5', 'inception_3a_relu_3x3', 'inception_3a_output', 'inception_3b_3x3_reduce', 'inception_3b_pool', 'inception_3b_1x1', \
        'inception_3b_5x5_reduce', 'inception_3b_relu_3x3_reduce', 'inception_3b_pool_proj', 'inception_3b_relu_1x1', 'inception_3b_relu_5x5_reduce', \
        'inception_3b_3x3', 'inception_3b_relu_pool_proj', 'inception_3b_5x5', 'inception_3b_relu_3x3', 'inception_3b_relu_5x5', 'inception_3b_output', \
        'pool3_3x3_s2', 'inception_4a_1x1', 'inception_4a_3x3_reduce', 'inception_4a_5x5_reduce', 'inception_4a_pool', 'inception_4a_relu_1x1', \
        'inception_4a_relu_3x3_reduce', 'inception_4a_relu_5x5_reduce', 'inception_4a_pool_proj', 'inception_4a_3x3', 'inception_4a_5x5', \
        'inception_4a_relu_pool_proj', 'inception_4a_relu_3x3', 'inception_4a_relu_5x5', 'inception_4a_output', 'inception_4b_pool', 'inception_4b_5x5_reduce', \
        'inception_4b_1x1', 'inception_4b_3x3_reduce', 'inception_4b_pool_proj', 'inception_4b_relu_5x5_reduce', 'inception_4b_relu_1x1', \
        'inception_4b_relu_3x3_reduce', 'inception_4b_relu_pool_proj', 'inception_4b_5x5', 'inception_4b_3x3', 'inception_4b_relu_5x5', 'inception_4b_relu_3x3', \
        'inception_4b_output', 'inception_4c_5x5_reduce', 'inception_4c_pool', 'inception_4c_1x1', 'inception_4c_3x3_reduce', 'inception_4c_relu_5x5_reduce', \
        'inception_4c_pool_proj', 'inception_4c_relu_1x1', 'inception_4c_relu_3x3_reduce', 'inception_4c_5x5', 'inception_4c_relu_pool_proj', 'inception_4c_3x3', \
        'inception_4c_relu_5x5', 'inception_4c_relu_3x3', 'inception_4c_output', 'inception_4d_pool', 'inception_4d_3x3_reduce', 'inception_4d_1x1', \
        'inception_4d_5x5_reduce', 'inception_4d_pool_proj', 'inception_4d_relu_3x3_reduce', 'inception_4d_relu_1x1', 'inception_4d_relu_5x5_reduce', \
        'inception_4d_relu_pool_proj', 'inception_4d_3x3', 'inception_4d_5x5', 'inception_4d_relu_3x3', 'inception_4d_relu_5x5', 'inception_4d_output', \
        'inception_4e_1x1', 'inception_4e_5x5_reduce', 'inception_4e_3x3_reduce', 'inception_4e_pool', 'inception_4e_relu_1x1', 'inception_4e_relu_5x5_reduce', \
        'inception_4e_relu_3x3_reduce', 'inception_4e_pool_proj', 'inception_4e_5x5', 'inception_4e_3x3', 'inception_4e_relu_pool_proj', 'inception_4e_relu_5x5', \
        'inception_4e_relu_3x3', 'inception_4e_output', 'pool4_3x3_s2 ', 'inception_5a_1x1', 'inception_5a_5x5_reduce', 'inception_5a_pool', \
        'inception_5a_3x3_reduce', 'inception_5a_relu_1x1', 'inception_5a_relu_5x5_reduce', 'inception_5a_pool_proj', 'inception_5a_relu_3x3_reduce', \
        'inception_5a_5x5', 'inception_5a_relu_pool_proj', 'inception_5a_3x3', 'inception_5a_relu_5x5', 'inception_5a_relu_3x3', 'inception_5a_output', \
        'inception_5b_3x3_reduce', 'inception_5b_pool', 'inception_5b_5x5_reduce', 'inception_5b_1x1', 'inception_5b_relu_3x3_reduce', 'inception_5b_pool_proj', \
        'inception_5b_relu_5x5_reduce', 'inception_5b_relu_1x1', 'inception_5b_3x3', 'inception_5b_relu_pool_proj', 'inception_5b_5x5', 'inception_5b_relu_3x3', \
        'inception_5b_relu_5x5', 'inception_5b_output', 'pool5_drop_7x7_s1']
    elif 'sos' in model_name:	
        hookable_layers = ['conv1_7x7_s2', 'conv1_relu_7x7', 'pool1_3x3_s2', 'conv2_3x3_reduce', 'conv2_relu_3x3_reduce', 'conv2_3x3', 'conv2_relu_3x3', 'pool2_3x3_s2', \
        'inception_3a_pool', 'inception_3a_1x1', 'inception_3a_5x5_reduce', 'inception_3a_3x3_reduce', 'inception_3a_pool_proj', 'inception_3a_relu_1x1', \
        'inception_3a_relu_5x5_reduce', 'inception_3a_relu_3x3_reduce', 'inception_3a_relu_pool_proj', 'inception_3a_5x5', 'inception_3a_3x3', \
        'inception_3a_relu_5x5', 'inception_3a_relu_3x3', 'inception_3a_output', 'inception_3b_3x3_reduce', 'inception_3b_pool', 'inception_3b_1x1', \
        'inception_3b_5x5_reduce', 'inception_3b_relu_3x3_reduce', 'inception_3b_pool_proj', 'inception_3b_relu_1x1', 'inception_3b_relu_5x5_reduce', \
        'inception_3b_3x3', 'inception_3b_relu_pool_proj', 'inception_3b_5x5', 'inception_3b_relu_3x3', 'inception_3b_relu_5x5', 'inception_3b_output', \
        'pool3_3x3_s2', 'inception_4a_1x1', 'inception_4a_3x3_reduce', 'inception_4a_5x5_reduce', 'inception_4a_pool', 'inception_4a_relu_1x1', \
        'inception_4a_relu_3x3_reduce', 'inception_4a_relu_5x5_reduce', 'inception_4a_pool_proj', 'inception_4a_3x3', 'inception_4a_5x5', 'inception_4a_relu_pool_proj', \
        'inception_4a_relu_3x3', 'inception_4a_relu_5x5', 'inception_4a_output', 'inception_4b_pool', 'inception_4b_5x5_reduce', 'inception_4b_1x1', \
        'inception_4b_3x3_reduce', 'inception_4b_pool_proj', 'inception_4b_relu_5x5_reduce', 'inception_4b_relu_1x1', 'inception_4b_relu_3x3_reduce', \
        'inception_4b_relu_pool_proj', 'inception_4b_5x5', 'inception_4b_3x3', 'inception_4b_relu_5x5', 'inception_4b_relu_3x3', 'inception_4b_output', \
        'inception_4c_5x5_reduce', 'inception_4c_pool', 'inception_4c_1x1', 'inception_4c_3x3_reduce', 'inception_4c_relu_5x5_reduce', 'inception_4c_pool_proj', \
        'inception_4c_relu_1x1', 'inception_4c_relu_3x3_reduce', 'inception_4c_5x5', 'inception_4c_relu_pool_proj', 'inception_4c_3x3', 'inception_4c_relu_5x5', \
        'inception_4c_relu_3x3', 'inception_4c_output', 'inception_4d_pool', 'inception_4d_3x3_reduce', 'inception_4d_1x1', 'inception_4d_5x5_reduce', \
        'inception_4d_pool_proj', 'inception_4d_relu_3x3_reduce', 'inception_4d_relu_1x1', 'inception_4d_relu_5x5_reduce', 'inception_4d_relu_pool_proj', \
        'inception_4d_3x3', 'inception_4d_5x5', 'inception_4d_relu_3x3', 'inception_4d_relu_5x5', 'inception_4d_output', 'inception_4e_5x5_reduce', 'inception_4e_1x1', \
        'inception_4e_3x3_reduce', 'inception_4e_pool', 'inception_4e_relu_5x5_reduce', 'inception_4e_relu_1x1', 'inception_4e_relu_3x3_reduce', 'inception_4e_pool_proj', \
        'inception_4e_5x5', 'inception_4e_3x3', 'inception_4e_relu_pool_proj', 'inception_4e_relu_5x5', 'inception_4e_relu_3x3', 'inception_4e_output', 'pool4_3x3_s2', \
        'inception_5a_1x1', 'inception_5a_5x5_reduce', 'inception_5a_pool', 'inception_5a_3x3_reduce', 'inception_5a_relu_1x1', 'inception_5a_relu_5x5_reduce', \
        'inception_5a_pool_proj', 'inception_5a_relu_3x3_reduce', 'inception_5a_5x5', 'inception_5a_relu_pool_proj', 'inception_5a_3x3', 'inception_5a_relu_5x5', \
        'inception_5a_relu_3x3', 'inception_5a_output', 'inception_5b_pool', 'inception_5b_3x3_reduce', 'inception_5b_5x5_reduce', 'inception_5b_1x1', \
        'inception_5b_pool_proj', 'inception_5b_relu_3x3_reduce', 'inception_5b_relu_5x5_reduce', 'inception_5b_relu_1x1', 'inception_5b_relu_pool_proj', \
        'inception_5b_3x3', 'inception_5b_5x5', 'inception_5b_relu_3x3', 'inception_5b_relu_5x5', 'inception_5b_output', 'pool5_drop_7x7_s1']
    elif 'cars' in model_name:
        hookable_layers = ['conv1', 'relu1', 'pool1', 'conv2_1x1', 'relu_conv2_1x1', 'conv2_3x3', 'relu2_3x3', 'pool2', 'inception_3a_5x5_reduce', 'inception_3a_3x3_reduce', \
        'inception_3a_1x1', 'inception_3a_pool', 'relu_inception_3a_5x5_reduce', 'reulu_inception_3a_3x3_reduce', 'relu_inception_3a_1x1', 'inception_3a_pool_proj', \
        'inception_3a_5x5', 'inception_3a_3x3', 'relu_inception_3a_pool_proj', 'relu_inception_3a_5x5', 'relu_inception_3a_3x3', 'inception_3a_output', \
        'inception_3b_1x1', 'inception_3b_3x3_reduce', 'inception_3b_5x5_reduce', 'inception_3b_pool', 'relu_inception_3b_1x1', 'relu_inception_3b_3x3_reduce', \
        'relu_inception_3b_5x5_reduce', 'inception_3b_pool_proj', 'inception_3b_3x3', 'inception_3b_5x5', 'relu_inception_3b_pool_proj', 'relu_inception_3b_3x3', \
        'relu_inception_3b_5x5', 'inception_3b_output', 'pool3', 'inception_4a_pool', 'inception_4a_3x3_reduce', 'inception_4a_5x5_reduce', 'inception_4a_1x1', \
        'inception_4a_pool_proj', 'relu_inception_4a_3x3_reduce', 'relu_inception_4a_5x5_reduce', 'relu_inception_4a_1x1', 'relu_inception_4a_pool_proj', \
        'inception_4a_3x3', 'inception_4a_5x5', 'relu_inception_4a_3x3', 'relu_inception_4a_5x5', 'inception_4a_output', 'inception_4b_3x3_reduce', 'inception_4b_1x1', \
        'inception_4b_pool', 'inception_4b_5x5_reduce', 'inception_4b_relu_3x3_reduce', 'inception_4b_relu_1x1', 'inception_4b_pool_proj', 'inception_4b_relu_5x5_reduce', \
        'inception_4b_3x3', 'inception_4b_relu_pool_proj', 'inception_4b_5x5', 'inception_4b_relu_3x3', 'inception_4b_relu_5x5', 'inception_4b_output', 'inception_4c_pool', \
        'inception_4c_1x1', 'inception_4c_5x5_reduce', 'inception_4c_3x3_reduce', 'inception_4c_pool_proj', 'inception_4c_relu_1x1', 'inception_4c_relu_5x5_reduce', \
        'inception_4c_relu_3x3_reduce', 'inception_4c_relu_pool_proj', 'inception_4c_5x5', 'inception_4c_3x3', 'inception_4c_relu_5x5', 'inception_4c_relu_3x3', \
        'inception_4c_output', 'inception_4d_1x1', 'inception_4d_3x3_reduce', 'inception_4d_5x5_reduce', 'inception_4d_pool', 'inception_4d_relu_1x1', \
        'inception_4d_relu_3x3_reduce', 'inception_4d_relu_5x5_reduce', 'inception_4d_pool_proj', 'inception_4d_3x3', 'inception_4d_5x5', 'inception_4d_relu_pool_proj', \
        'inception_4d_relu_3x3', 'inception_4d_relu_5x5', 'inception_4d_output', 'inception_4e_pool', 'inception_4e_1x1', 'inception_4e_3x3_reduce', \
        'inception_4e_5x5_reduce', 'inception_4e_pool_proj', 'inception_4e_relu_1x1', 'inception_4e_relu_3x3_reduce', 'inception_4e_relu_5x5_reduce', \
        'inception_4e_relu_pool_proj', 'inception_4e_3x3', 'inception_4e_5x5', 'inception_4e_relu_3x3', 'inception_4e_relu_5x5', 'inception_4e_output', 'pool4', \
        'inception_5a_pool', 'inception_5a_5x5_reduce', 'inception_5a_3x3_reduce', 'inception_5a_1x1', 'inception_5a_pool_proj', 'inception_5a_relu_5x5_reduce', \
        'inception_5a_relu_3x3_reduce', 'inception_5a_relu_1x1', 'inception_5a_relu_pool_proj', 'inception_5a_5x5', 'inception_5a_3x3', 'inception_5a_relu_5x5', \
        'inception_5a_relu_3x3', 'inception_5a_output', 'inception_5b_3x3_reduce', 'inception_5b_1x1', 'inception_5b_5x5_reduce', 'inception_5b_pool', \
        'inception_5b_relu_3x3_reduce', 'inception_5b_relu_1x1', 'inception_5b_relu_5x5_reduce', 'inception_5b_pool_proj', 'inception_5b_3x3', 'inception_5b_5x5', \
        'inception_5b_relu_pool_proj', 'inception_5b_relu_3x3', 'inception_5b_relu_5x5', 'inception_5b_output', 'pool5_drop']
    return hookable_layers