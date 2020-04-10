def inception_layer_names(model_name='5h'):
    if model_name == '5h':
        hookable_layers = ['conv2d0_pre_relu_conv', 'conv2d0', 'maxpool0', 'conv2d1_pre_relu_conv', 'conv2d1', 'conv2d2_pre_relu_conv', 'conv2d2', 'maxpool1', \
        'mixed3a_1x1_pre_relu_conv', 'mixed3a_3x3_bottleneck_pre_relu_conv', 'mixed3a_5x5_bottleneck_pre_relu_conv', 'mixed3a_pool', 'mixed3a_1x1', \
        'mixed3a_3x3_bottleneck', 'mixed3a_5x5_bottleneck', 'mixed3a_pool_reduce_pre_relu_conv', 'mixed3a_3x3_pre_relu_conv', 'mixed3a_5x5_pre_relu_conv', \
        'mixed3a_pool_reduce', 'mixed3a_3x3', 'mixed3a_5x5', 'mixed3a', 'mixed3b_1x1_pre_relu_conv', 'mixed3b_3x3_bottleneck_pre_relu_conv', \
        'mixed3b_5x5_bottleneck_pre_relu_conv', 'mixed3b_pool', 'mixed3b_1x1', 'mixed3b_3x3_bottleneck', 'mixed3b_5x5_bottleneck', \
        'mixed3b_pool_reduce_pre_relu_conv', 'mixed3b_3x3_pre_relu_conv', 'mixed3b_5x5_pre_relu_conv', 'mixed3b_pool_reduce', 'mixed3b_3x3', 'mixed3b_5x5', \
        'mixed3b', 'maxpool4', 'mixed4a_1x1_pre_relu_conv', 'mixed4a_3x3_bottleneck_pre_relu_conv', 'mixed4a_5x5_bottleneck_pre_relu_conv', 'mixed4a_pool', \
        'mixed4a_1x1', 'mixed4a_3x3_bottleneck', 'mixed4a_5x5_bottleneck', 'mixed4a_pool_reduce_pre_relu_conv', 'mixed4a_3x3_pre_relu_conv', \
        'mixed4a_5x5_pre_relu_conv', 'mixed4a_pool_reduce', 'mixed4a_3x3', 'mixed4a_5x5', 'mixed4a', 'mixed4b_1x1_pre_relu_conv', \
        'mixed4b_3x3_bottleneck_pre_relu_conv', 'mixed4b_5x5_bottleneck_pre_relu_conv', 'mixed4b_pool', 'mixed4b_1x1', 'mixed4b_3x3_bottleneck', \
        'mixed4b_5x5_bottleneck', 'mixed4b_pool_reduce_pre_relu_conv', 'mixed4b_3x3_pre_relu_conv', 'mixed4b_5x5_pre_relu_conv', 'mixed4b_pool_reduce', \
        'mixed4b_3x3', 'mixed4b_5x5', 'mixed4b', 'mixed4c_1x1_pre_relu_conv', 'mixed4c_3x3_bottleneck_pre_relu_conv', 'mixed4c_5x5_bottleneck_pre_relu_conv', \
        'mixed4c_pool', 'mixed4c_1x1', 'mixed4c_3x3_bottleneck', 'mixed4c_5x5_bottleneck', 'mixed4c_pool_reduce_pre_relu_conv', 'mixed4c_3x3_pre_relu_conv', \
        'mixed4c_5x5_pre_relu_conv', 'mixed4c_pool_reduce', 'mixed4c_3x3', 'mixed4c_5x5', 'mixed4c', 'mixed4d_1x1_pre_relu_conv', \
        'mixed4d_3x3_bottleneck_pre_relu_conv', 'mixed4d_5x5_bottleneck_pre_relu_conv', 'mixed4d_pool', 'mixed4d_1x1', 'mixed4d_3x3_bottleneck', \
        'mixed4d_5x5_bottleneck', 'mixed4d_pool_reduce_pre_relu_conv', 'mixed4d_3x3_pre_relu_conv', 'mixed4d_5x5_pre_relu_conv', 'mixed4d_pool_reduce', \
        'mixed4d_3x3', 'mixed4d_5x5', 'mixed4d', 'mixed4e_1x1_pre_relu_conv', 'mixed4e_3x3_bottleneck_pre_relu_conv', 'mixed4e_5x5_bottleneck_pre_relu_conv', \
        'mixed4e_pool', 'mixed4e_1x1', 'mixed4e_3x3_bottleneck', 'mixed4e_5x5_bottleneck', 'mixed4e_pool_reduce_pre_relu_conv', 'mixed4e_3x3_pre_relu_conv', \
        'mixed4e_5x5_pre_relu_conv', 'mixed4e_pool_reduce', 'mixed4e_3x3', 'mixed4e_5x5', 'mixed4e', 'maxpool10', 'mixed5a_1x1_pre_relu_conv', \
        'mixed5a_3x3_bottleneck_pre_relu_conv', 'mixed5a_5x5_bottleneck_pre_relu_conv', 'mixed5a_pool', 'mixed5a_1x1', 'mixed5a_3x3_bottleneck', \
        'mixed5a_5x5_bottleneck', 'mixed5a_pool_reduce_pre_relu_conv', 'mixed5a_3x3_pre_relu_conv', 'mixed5a_5x5_pre_relu_conv', 'mixed5a_pool_reduce', \
        'mixed5a_3x3', 'mixed5a_5x5', 'mixed5a', 'mixed5b_1x1_pre_relu_conv', 'mixed5b_3x3_bottleneck_pre_relu_conv', 'mixed5b_5x5_bottleneck_pre_relu_conv', \
        'mixed5b_pool', 'mixed5b_1x1', 'mixed5b_3x3_bottleneck', 'mixed5b_5x5_bottleneck', 'mixed5b_pool_reduce_pre_relu_conv', 'mixed5b_3x3_pre_relu_conv', \
        'mixed5b_5x5_pre_relu_conv', 'mixed5b_pool_reduce', 'mixed5b_3x3', 'mixed5b_5x5', 'mixed5b', 'softmax2_pre_activation_matmul', 'softmax2']
    return hookable_layers