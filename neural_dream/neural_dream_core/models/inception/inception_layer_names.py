def inception_layer_names(model_name='5h'):
    if model_name == '5h':
        hookable_layers = ['conv2d0_pre_relu_conv_pad', 'conv2d0_pre_relu_conv', 'conv2d0', 'maxpool0_pad', 'maxpool0', 'localresponsenorm0', 'conv2d1_pre_relu_conv', \
		'conv2d1', 'conv2d2_pre_relu_conv_pad', 'conv2d2_pre_relu_conv', 'conv2d2', 'localresponsenorm1', 'maxpool1_pad', 'maxpool1', 'mixed3a_1x1_pre_relu_conv', \
		'mixed3a_3x3_bottleneck_pre_relu_conv', 'mixed3a_5x5_bottleneck_pre_relu_conv', 'mixed3a_pool_pad', 'mixed3a_pool', 'mixed3a_1x1', 'mixed3a_3x3_bottleneck', \
		'mixed3a_5x5_bottleneck', 'mixed3a_pool_reduce_pre_relu_conv', 'mixed3a_3x3_pre_relu_conv_pad', 'mixed3a_3x3_pre_relu_conv', 'mixed3a_5x5_pre_relu_conv_pad', \
		'mixed3a_5x5_pre_relu_conv', 'mixed3a_pool_reduce', 'mixed3a_3x3', 'mixed3a_5x5', 'mixed3a', 'mixed3b_1x1_pre_relu_conv', 'mixed3b_3x3_bottleneck_pre_relu_conv', \
		'mixed3b_5x5_bottleneck_pre_relu_conv', 'mixed3b_pool_pad', 'mixed3b_pool', 'mixed3b_1x1', 'mixed3b_3x3_bottleneck', 'mixed3b_5x5_bottleneck', \
		'mixed3b_pool_reduce_pre_relu_conv', 'mixed3b_3x3_pre_relu_conv_pad', 'mixed3b_3x3_pre_relu_conv', 'mixed3b_5x5_pre_relu_conv_pad', 'mixed3b_5x5_pre_relu_conv', \
		'mixed3b_pool_reduce', 'mixed3b_3x3', 'mixed3b_5x5', 'mixed3b', 'maxpool4_pad', 'maxpool4', 'mixed4a_1x1_pre_relu_conv', 'mixed4a_3x3_bottleneck_pre_relu_conv', \
		'mixed4a_5x5_bottleneck_pre_relu_conv', 'mixed4a_pool_pad', 'mixed4a_pool', 'mixed4a_1x1', 'mixed4a_3x3_bottleneck', 'mixed4a_5x5_bottleneck', \
	    'mixed4a_pool_reduce_pre_relu_conv', 'mixed4a_3x3_pre_relu_conv_pad', 'mixed4a_3x3_pre_relu_conv', 'mixed4a_5x5_pre_relu_conv_pad', 'mixed4a_5x5_pre_relu_conv', \
		'mixed4a_pool_reduce', 'mixed4a_3x3', 'mixed4a_5x5', 'mixed4a', 'head0_bottleneck_pre_relu_conv', 'head0_bottleneck', 'nn0_pre_relu_matmul', 'nn0', \
		'softmax0_pre_activation_matmul', 'softmax0']
    return hookable_layers