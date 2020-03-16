def resnet_layer_names(mode):
    if mode == '50_1by2_nsfw':
        hookable_layers = ['conv_1', 'bn_1', 'conv_stage0_block0_branch2a', 'conv_stage0_block0_proj_shortcut', 'bn_stage0_block0_branch2a', 'bn_stage0_block0_proj_shortcut', \
        'conv_stage0_block0_branch2b', 'bn_stage0_block0_branch2b', 'conv_stage0_block0_branch2c', 'bn_stage0_block0_branch2c', 'conv_stage0_block1_branch2a', \
        'bn_stage0_block1_branch2a', 'conv_stage0_block1_branch2b', 'bn_stage0_block1_branch2b', 'conv_stage0_block1_branch2c', 'bn_stage0_block1_branch2c', \
        'conv_stage0_block2_branch2a', 'bn_stage0_block2_branch2a', 'conv_stage0_block2_branch2b', 'bn_stage0_block2_branch2b', 'conv_stage0_block2_branch2c', \
        'bn_stage0_block2_branch2c', 'conv_stage1_block0_proj_shortcut', 'conv_stage1_block0_branch2a', 'bn_stage1_block0_proj_shortcut', 'bn_stage1_block0_branch2a', \
        'conv_stage1_block0_branch2b', 'bn_stage1_block0_branch2b', 'conv_stage1_block0_branch2c', 'bn_stage1_block0_branch2c', 'conv_stage1_block1_branch2a', \
        'bn_stage1_block1_branch2a', 'conv_stage1_block1_branch2b', 'bn_stage1_block1_branch2b', 'conv_stage1_block1_branch2c', 'bn_stage1_block1_branch2c', \
        'conv_stage1_block2_branch2a', 'bn_stage1_block2_branch2a', 'conv_stage1_block2_branch2b', 'bn_stage1_block2_branch2b', 'conv_stage1_block2_branch2c', \
        'bn_stage1_block2_branch2c', 'conv_stage1_block3_branch2a', 'bn_stage1_block3_branch2a', 'conv_stage1_block3_branch2b', 'bn_stage1_block3_branch2b', \
        'conv_stage1_block3_branch2c', 'bn_stage1_block3_branch2c', 'conv_stage2_block0_proj_shortcut', 'conv_stage2_block0_branch2a', 'bn_stage2_block0_proj_shortcut', \
        'bn_stage2_block0_branch2a', 'conv_stage2_block0_branch2b', 'bn_stage2_block0_branch2b', 'conv_stage2_block0_branch2c', 'bn_stage2_block0_branch2c', \
        'conv_stage2_block1_branch2a', 'bn_stage2_block1_branch2a', 'conv_stage2_block1_branch2b', 'bn_stage2_block1_branch2b', 'conv_stage2_block1_branch2c', \
        'bn_stage2_block1_branch2c', 'conv_stage2_block2_branch2a', 'bn_stage2_block2_branch2a', 'conv_stage2_block2_branch2b', 'bn_stage2_block2_branch2b', \
        'conv_stage2_block2_branch2c', 'bn_stage2_block2_branch2c', 'conv_stage2_block3_branch2a', 'bn_stage2_block3_branch2a', 'conv_stage2_block3_branch2b', \
        'bn_stage2_block3_branch2b', 'conv_stage2_block3_branch2c', 'bn_stage2_block3_branch2c', 'conv_stage2_block4_branch2a', 'bn_stage2_block4_branch2a', \
        'conv_stage2_block4_branch2b', 'bn_stage2_block4_branch2b', 'conv_stage2_block4_branch2c', 'bn_stage2_block4_branch2c', 'conv_stage2_block5_branch2a', \
        'bn_stage2_block5_branch2a', 'conv_stage2_block5_branch2b', 'bn_stage2_block5_branch2b', 'conv_stage2_block5_branch2c', 'bn_stage2_block5_branch2c', \
        'conv_stage3_block0_proj_shortcut', 'conv_stage3_block0_branch2a', 'bn_stage3_block0_proj_shortcut', 'bn_stage3_block0_branch2a', 'conv_stage3_block0_branch2b', \
        'bn_stage3_block0_branch2b', 'conv_stage3_block0_branch2c', 'bn_stage3_block0_branch2c', 'conv_stage3_block1_branch2a', 'bn_stage3_block1_branch2a', \
        'conv_stage3_block1_branch2b', 'bn_stage3_block1_branch2b', 'conv_stage3_block1_branch2c', 'bn_stage3_block1_branch2c', 'conv_stage3_block2_branch2a', \
        'bn_stage3_block2_branch2a', 'conv_stage3_block2_branch2b', 'bn_stage3_block2_branch2b', 'conv_stage3_block2_branch2c', 'bn_stage3_block2_branch2c', 'fc_nsfw_1', \
		'eltwise_stage0_block0', 'eltwise_stage0_block1', 'eltwise_stage0_block2', 'eltwise_stage1_block0', 'eltwise_stage1_block1', 'eltwise_stage1_block2', \
        'eltwise_stage1_block3', 'eltwise_stage2_block0', 'eltwise_stage2_block1', 'eltwise_stage2_block2', 'eltwise_stage2_block3', 'eltwise_stage2_block4', \
        'eltwise_stage2_block5', 'eltwise_stage3_block0', 'eltwise_stage3_block1', 'eltwise_stage3_block2']
    return hookable_layers