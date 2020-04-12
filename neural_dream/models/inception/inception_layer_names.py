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
    elif model_name == 'keras-wtop':
        hookable_layers = ['conv2d_1', 'batch_normalization_1', 'activation_1', 'conv2d_2', 'batch_normalization_2', 'activation_2', 'conv2d_3', 'batch_normalization_3', \
        'activation_3', 'max_pooling2d_1', 'conv2d_4', 'batch_normalization_4', 'activation_4', 'conv2d_5', 'batch_normalization_5', 'activation_5', \
        'max_pooling2d_2', 'conv2d_9', 'conv2d_7', 'average_pooling2d_1', 'conv2d_6', 'batch_normalization_9', 'batch_normalization_7', 'conv2d_12', \
        'batch_normalization_6', 'activation_9', 'activation_7', 'batch_normalization_12', 'activation_6', 'conv2d_10', 'conv2d_8', 'activation_12', \
        'batch_normalization_10', 'batch_normalization_8', 'activation_10', 'activation_8', 'conv2d_11', 'batch_normalization_11', 'activation_11', \
        'mixed0', 'conv2d_16', 'conv2d_14', 'average_pooling2d_2', 'conv2d_13', 'batch_normalization_16', 'batch_normalization_14', 'conv2d_19', \
        'batch_normalization_13', 'activation_16', 'activation_14', 'batch_normalization_19', 'activation_13', 'conv2d_17', 'conv2d_15', 'activation_19', \
        'batch_normalization_17', 'batch_normalization_15', 'activation_17', 'activation_15', 'conv2d_18', 'batch_normalization_18', 'activation_18', \
        'mixed1', 'conv2d_23', 'conv2d_21', 'average_pooling2d_3', 'conv2d_20', 'batch_normalization_23', 'batch_normalization_21', 'conv2d_26', \
        'batch_normalization_20', 'activation_23', 'activation_21', 'batch_normalization_26', 'activation_20', 'conv2d_24', 'conv2d_22', 'activation_26', \
        'batch_normalization_24', 'batch_normalization_22', 'activation_24', 'activation_22', 'conv2d_25', 'batch_normalization_25', 'activation_25', \
        'mixed2', 'conv2d_28', 'conv2d_27', 'max_pooling2d_3', 'batch_normalization_28', 'batch_normalization_27', 'activation_28', 'activation_27', \
        'conv2d_29', 'batch_normalization_29', 'activation_29', 'conv2d_30', 'batch_normalization_30', 'activation_30', 'mixed3', 'conv2d_35', 'conv2d_32', \
        'average_pooling2d_4', 'conv2d_31', 'batch_normalization_35', 'batch_normalization_32', 'conv2d_40', 'batch_normalization_31', 'activation_35', \
        'activation_32', 'batch_normalization_40', 'activation_31', 'conv2d_36', 'conv2d_33', 'activation_40', 'batch_normalization_36', \
        'batch_normalization_33', 'activation_36', 'activation_33', 'conv2d_37', 'conv2d_34', 'batch_normalization_37', 'batch_normalization_34', \
        'activation_37', 'activation_34', 'conv2d_38', 'batch_normalization_38', 'activation_38', 'conv2d_39', 'batch_normalization_39', 'activation_39', \
        'mixed4', 'conv2d_45', 'conv2d_42', 'average_pooling2d_5', 'conv2d_41', 'batch_normalization_45', 'batch_normalization_42', 'conv2d_50', \
        'batch_normalization_41', 'activation_45', 'activation_42', 'batch_normalization_50', 'activation_41', 'conv2d_46', 'conv2d_43', 'activation_50', \
        'batch_normalization_46', 'batch_normalization_43', 'activation_46', 'activation_43', 'conv2d_47', 'conv2d_44', 'batch_normalization_47', \
        'batch_normalization_44', 'activation_47', 'activation_44', 'conv2d_48', 'batch_normalization_48', 'activation_48', 'conv2d_49', \
        'batch_normalization_49', 'activation_49', 'mixed5', 'conv2d_55', 'conv2d_52', 'average_pooling2d_6', 'conv2d_51', 'batch_normalization_55', \
        'batch_normalization_52', 'conv2d_60', 'batch_normalization_51', 'activation_55', 'activation_52', 'batch_normalization_60', 'activation_51', \
        'conv2d_56', 'conv2d_53', 'activation_60', 'batch_normalization_56', 'batch_normalization_53', 'activation_56', 'activation_53', 'conv2d_57', \
        'conv2d_54', 'batch_normalization_57', 'batch_normalization_54', 'activation_57', 'activation_54', 'conv2d_58', 'batch_normalization_58', \
        'activation_58', 'conv2d_59', 'batch_normalization_59', 'activation_59', 'mixed6', 'conv2d_65', 'conv2d_62', 'average_pooling2d_7', 'conv2d_61', \
        'batch_normalization_65', 'batch_normalization_62', 'conv2d_70', 'batch_normalization_61', 'activation_65', 'activation_62', \
        'batch_normalization_70', 'activation_61', 'conv2d_66', 'conv2d_63', 'activation_70', 'batch_normalization_66', 'batch_normalization_63', \
        'activation_66', 'activation_63', 'conv2d_67', 'conv2d_64', 'batch_normalization_67', 'batch_normalization_64', 'activation_67', 'activation_64', \
        'conv2d_68', 'batch_normalization_68', 'activation_68', 'conv2d_69', 'batch_normalization_69', 'activation_69', 'mixed7', 'conv2d_73', 'conv2d_71', \
        'max_pooling2d_4', 'batch_normalization_73', 'batch_normalization_71', 'activation_73', 'activation_71', 'conv2d_74', 'conv2d_72', \
        'batch_normalization_74', 'batch_normalization_72', 'activation_74', 'activation_72', 'conv2d_75', 'batch_normalization_75', 'activation_75', \
        'conv2d_76', 'batch_normalization_76', 'activation_76', 'mixed8', 'conv2d_81', 'conv2d_78', 'average_pooling2d_8', 'conv2d_77', \
        'batch_normalization_81', 'batch_normalization_78', 'conv2d_85', 'batch_normalization_77', 'activation_81', 'activation_78', \
        'batch_normalization_85', 'activation_77', 'conv2d_82', 'conv2d_79', 'conv2d_80', 'activation_85', 'batch_normalization_82', \
        'batch_normalization_79', 'batch_normalization_80', 'activation_82', 'activation_79', 'activation_80', 'conv2d_83', 'conv2d_84', 'mixed9_0', \
        'batch_normalization_83', 'batch_normalization_84', 'activation_83', 'activation_84', 'concatenate_1', 'mixed9', 'conv2d_90', 'conv2d_87', \
        'average_pooling2d_9', 'conv2d_86', 'batch_normalization_90', 'batch_normalization_87', 'conv2d_94', 'batch_normalization_86', 'activation_90', \
        'activation_87', 'batch_normalization_94', 'activation_86', 'conv2d_91', 'conv2d_88', 'conv2d_89', 'activation_94', 'batch_normalization_91', \
        'batch_normalization_88', 'batch_normalization_89', 'activation_91', 'activation_88', 'activation_89', 'conv2d_92', 'conv2d_93', 'mixed9_1', \
        'batch_normalization_92', 'batch_normalization_93', 'activation_92', 'activation_93', 'concatenate_2', 'mixed10', 'avg_pool', 'predictions', \
        'predictions_activation']
    return hookable_layers