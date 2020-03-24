# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from utilities import PReLU, spatial_dropout, max_unpool, spatial_softmax,flatten
class EnetNet():
    def __init__(self, wd, late_drop_prob_ph, early_drop_prob_ph, no_of_classes):
        self.late_drop_prob_ph=late_drop_prob_ph
        self.early_drop_prob_ph= early_drop_prob_ph
        self.wd = wd
        self.no_of_classes = no_of_classes
    def initial_block(self, x, scope):
        # convolution branch:
        W_conv = self.get_variable_weight_decay(scope + "/W",
                    shape=[3, 3, 3, 13], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b", shape=[13], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1],
                    padding="SAME") + b_conv
    
        # max pooling branch:
        pool_branch = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding="VALID")
    
        # concatenate the branches:
        concat = tf.concat([conv_branch, pool_branch], axis=3) # (3: the depth axis)
    
        # apply batch normalization and PReLU:
        output = tf.contrib.slim.batch_norm(concat)
        output = PReLU(output, scope=scope)
    
        return output
    
    def encoder_bottleneck_regular(self, x, output_depth, drop_prob, scope,
                proj_ratio=4, downsampling=False):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]
    
        internal_depth = int(output_depth/proj_ratio)
    
        # convolution branch:
        conv_branch = x
    
        # # 1x1 projection:
        if downsampling:
            W_conv = self.get_variable_weight_decay(scope + "/W_proj",
                        shape=[2, 2, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="encoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 2, 2, 1],
                        padding="VALID") # NOTE! no bias terms
        else:
            W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                        shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="encoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                        padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")
    
        # # conv:
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                    shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")
    
        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no PReLU here
    
        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)
    
    
        # main branch:
        main_branch = x
    
        if downsampling:
            # max pooling with argmax (for use in max_unpool in the decoder):
            main_branch, pooling_indices = tf.nn.max_pool_with_argmax(main_branch,
                        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # (everytime we downsample, we also increase the feature block depth)
    
            # pad with zeros so that the feature block depth matches:
            depth_to_pad = output_depth - input_depth
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth_to_pad]])
            # (paddings is an integer tensor of shape [4, 2] where 4 is the rank
            # of main_branch. For each dimension D (D = 0, 1, 2, 3) of main_branch,
            # paddings[D, 0] is the no of values to add before the contents of
            # main_branch in that dimension, and paddings[D, 0] is the no of
            # values to add after the contents of main_branch in that dimension)
            main_branch = tf.pad(main_branch, paddings=paddings, mode="CONSTANT")
    
    
        # add the branches:
        merged = conv_branch + main_branch
    
        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")
    
        if downsampling:
            return output, pooling_indices
        else:
            return output
    
    def encoder_bottleneck_dilated(self, x, output_depth, drop_prob, scope,
                dilation_rate, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]
    
        internal_depth = int(output_depth/proj_ratio)
    
        # convolution branch:
        conv_branch = x
    
        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")
    
        # # dilated conv:
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                    shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.atrous_conv2d(conv_branch, W_conv, rate=dilation_rate,
                    padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")
    
        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no PReLU here
    
        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)
    
    
        # main branch:
        main_branch = x
    
    
        # add the branches:
        merged = conv_branch + main_branch
    
        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")
    
        return output
    
    def encoder_bottleneck_asymmetric(self, x, output_depth, drop_prob, scope, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]
    
        internal_depth = int(output_depth/proj_ratio)
    
        # convolution branch:
        conv_branch = x
    
        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")
    
        # # asymmetric conv:
        # # # asymmetric conv 1:
        W_conv1 = self.get_variable_weight_decay(scope + "/W_conv1",
                    shape=[5, 1, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv1, strides=[1, 1, 1, 1],
                    padding="SAME") # NOTE! no bias terms
        # # # asymmetric conv 2:
        W_conv2 = self.get_variable_weight_decay(scope + "/W_conv2",
                    shape=[1, 5, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv2 = self.get_variable_weight_decay(scope + "/b_conv2", shape=[internal_depth], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv2, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv2
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")
    
        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no PReLU here
    
        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)
    
    
        # main branch:
        main_branch = x
    
    
        # add the branches:
        merged = conv_branch + main_branch
    
        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")
    
        return output
    
    def decoder_bottleneck(self, x, output_depth, scope, proj_ratio=4,
                upsampling=False, pooling_indices=None, output_shape=None):
        # NOTE! decoder uses ReLU instead of PReLU
    
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]
    
        internal_depth = int(output_depth/proj_ratio)
    
        # main branch:
        main_branch = x
    
        if upsampling:
            # # 1x1 projection (to decrease depth to the same value as before downsampling):
            W_upsample = self.get_variable_weight_decay(scope + "/W_upsample",
                        shape=[1, 1, input_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="decoder_wd_losses")
            main_branch = tf.nn.conv2d(main_branch, W_upsample, strides=[1, 1, 1, 1],
                        padding="VALID") # NOTE! no bias terms
            # # # batch norm:
            main_branch = tf.contrib.slim.batch_norm(main_branch)
            # NOTE! no ReLU here
    
            # # max unpooling:
            main_branch = max_unpool(main_branch, pooling_indices, output_shape)
    
        main_branch = tf.cast(main_branch, tf.float32)
    
    
        # convolution branch:
        conv_branch = x
    
        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="decoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = tf.nn.relu(conv_branch)
    
        # # conv:
        if upsampling:
            # deconvolution:
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                        shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="decoder_wd_losses")
            b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth]], one bias weight per out depth layer),
                        initializer=tf.constant_initializer(0),
                        loss_category="decoder_wd_losses")
            main_branch_shape = main_branch.get_shape().as_list()
            output_shape = tf.convert_to_tensor([main_branch_shape[0],
                        main_branch_shape[1], main_branch_shape[2], internal_depth])
            conv_branch = tf.nn.conv2d_transpose(conv_branch, W_conv, output_shape=output_shape,
                        strides=[1, 2, 2, 1], padding="SAME") + b_conv
        else:
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                        shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="decoder_wd_losses")
            b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth])
                        initializer=tf.constant_initializer(0),
                        loss_category="decoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                        padding="SAME") + b_conv
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = tf.nn.relu(conv_branch)
    
        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="decoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no ReLU here
    
        # NOTE! no regularizer
    
    
        # add the branches:
        merged = conv_branch + main_branch
    
        # apply ReLU:
        output = tf.nn.relu(merged)
    
        return output
    
        
    def line_existance_part(self, x, scope, dilation_rate=4,drop_prob=0.1, verbose=True):
        # so the output of dilated convolution should be 36 x 100 x 32, if the input was 288 x 800 x 3, encoder output will be 36 x 100 x 128
        # # dilated conv:
        # w h w c
        shape = x.get_shape().as_list()
        
        W_conv = self.get_variable_weight_decay(scope + "/W_dconv",
                    shape=[3, 3, shape[-1], 32], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="line_existance_wd_loss")
        b_conv = self.get_variable_weight_decay(scope + "/b_dconv", shape=[32], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="line_existance_wd_loss")
        conv_branch = tf.nn.atrous_conv2d(x, W_conv, rate=dilation_rate,
                    padding="SAME") + b_conv
        if(verbose):
            print(conv_branch.get_shape().as_list(), "dilated convolution")
               
                                        
                                
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        if(verbose):print(conv_branch.get_shape().as_list(), "batch normalization")
        conv_branch = tf.nn.relu(conv_branch, name=scope + "/RELU1")
        
        if(verbose):print(conv_branch.get_shape().as_list(), "relu")
        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)
        
        if(verbose):print(conv_branch.get_shape().as_list(), "dropout")
        
        # # conv:
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                    shape=[3, 3, 32, 5], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="line_existance_wd_loss")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[5], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="line_existance_wd_loss")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv
        if(verbose):print(conv_branch.get_shape().as_list(), "convolution 1, 1")                  
        # spatial softmax
        conv_branch = spatial_softmax(conv_branch)
        if(verbose):print(conv_branch.get_shape().as_list(), "spatial softmax")
        
        conv_branch = tf.nn.avg_pool(conv_branch, ksize=2, strides=2,padding="SAME")
        if(verbose):print(conv_branch.get_shape().as_list(), "average pooling")
        # size of flattened matrix should be 4500
        fc = flatten(conv_branch)
        if(verbose):print(fc.get_shape().as_list(), "flatten")
        # # fully connected network:
        W_fc = self.get_variable_weight_decay(scope + "/W_fc",
                    shape=[4500, 128], 
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="line_existance_wd_loss")
        
        b_fc = self.get_variable_weight_decay(scope + "/b_fc", shape=[128],
                    initializer=tf.constant_initializer(0),
                    loss_category="line_existance_wd_loss")
        fc = tf.matmul(fc, W_fc)+ b_fc
        if(verbose):print(fc.get_shape().as_list(), "fully connected")
        fc = tf.nn.relu(fc, name=scope + "/RELU2")
        if(verbose):print(fc.get_shape().as_list(), "relu")
        # # fully connected network:
        W_fc = self.get_variable_weight_decay(scope + "/W_fc1",
                    shape=[128, 4], 
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="line_existance_wd_loss")
        
        b_fc = self.get_variable_weight_decay(scope + "/b_fc1", shape=[4],
                    initializer=tf.constant_initializer(0),
                    loss_category="line_existance_wd_loss")
        fc = tf.matmul(fc, W_fc)+ b_fc
        if(verbose):print(fc.get_shape().as_list(), "fully connected")
        fc = tf.math.sigmoid(fc,name=scope + '/existance_logits')
        if(verbose):print(fc.get_shape().as_list(), "sigmoid")
        return fc
    
    
    
    
    def get_variable_weight_decay(self,name, shape, initializer, loss_category,
                dtype=tf.float32):
        variable = tf.get_variable(name, shape=shape, dtype=dtype,
                    initializer=initializer)
    
        # add a variable weight decay loss:
        weight_decay = self.wd*tf.nn.l2_loss(variable)
        tf.add_to_collection(loss_category, weight_decay)
    
        return variable
    
    def add_logits(self, imgs_ph):
        # encoder:
        # # initial block:
        network = self.initial_block(x=imgs_ph, scope="inital")
        print(network.get_shape().as_list())
    
        
        # # layer 1:
        # # # save the input shape to use in max_unpool in the decoder:
        inputs_shape_1 = network.get_shape().as_list()
        network, pooling_indices_1 = self.encoder_bottleneck_regular(x=network,
                    output_depth=64, drop_prob=self.early_drop_prob_ph,
                    scope="bottleneck_1_0", downsampling=True)
        print( network.get_shape().as_list())
    
        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_1")
        print( network.get_shape().as_list())
    
        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_2")
        print( network.get_shape().as_list())
    
        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_3")
        print( network.get_shape().as_list())
    
        AT_GEN1 = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_4")
        
    
        self.AT_GEN1 = tf.square(tf.reduce_sum(AT_GEN1, -1))
        print( AT_GEN1.get_shape().as_list())
        
        
        # # layer 2:
        # # # save the input shape to use in max_unpool in the decoder:
        inputs_shape_2 = AT_GEN1.get_shape().as_list()
        network, pooling_indices_2 = self.encoder_bottleneck_regular(x=AT_GEN1,
                    output_depth=128, drop_prob=self.late_drop_prob_ph,
                    scope="bottleneck_2_0", downsampling=True)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                        drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_1")
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_2",
                    dilation_rate=2)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_3")
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_4",
                    dilation_rate=4)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_5")
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_6",
                    dilation_rate=8)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_7")
        print(network.get_shape().as_list())
    
        AT_GEN2 = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_8",
                    dilation_rate=16)
        
    
        self.AT_GEN2 = tf.square(tf.reduce_sum(AT_GEN2, -1))
        print(AT_GEN2.get_shape().as_list())
        
        # layer 3:
        network = self.encoder_bottleneck_regular(x=AT_GEN2, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1")
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2",
                    dilation_rate=2)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_3")
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_4",
                    dilation_rate=4)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_5")
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_6",
                    dilation_rate=8)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_7")
        print(network.get_shape().as_list())
    
        AT_GEN3 = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_8",
                    dilation_rate=16)
        
        
        self.AT_GEN3 = tf.square(tf.reduce_sum(AT_GEN3, -1))
        print(AT_GEN3.get_shape().as_list())    
        
        # layer 4:
        network = self.encoder_bottleneck_regular(x=AT_GEN3, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_4_1")
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_4_2",
                    dilation_rate=2)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_4_3")
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_4_4",
                    dilation_rate=4)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_4_5")
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_4_6",
                    dilation_rate=8)
        print(network.get_shape().as_list())
    
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_4_7")
        print(network.get_shape().as_list())
    
        AT_GEN4 = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_4_8",
                    dilation_rate=16)
        
        
        self.AT_GEN4 = tf.square(tf.reduce_sum(AT_GEN4, -1))
        print(AT_GEN4.get_shape().as_list())
    
        print("------------------end of encoder-------------------------------")
        
        self.line_existance_logit = self.line_existance_part(AT_GEN4, scope="line_existance_model", dilation_rate=4, verbose=True)
        print(self.line_existance_logit.get_shape().as_list())
        print("------------------end of line existance model------------------")
        # decoder:
        # # layer 5:
        network = self.decoder_bottleneck(x=AT_GEN4, output_depth=64,
                    scope="bottleneck_5_0", upsampling=True,
                    pooling_indices=pooling_indices_2, output_shape=inputs_shape_2)
        print(network.get_shape().as_list())
    
        network = self.decoder_bottleneck(x=network, output_depth=64,
                    scope="bottleneck_5_1")
        print(network.get_shape().as_list())
    
        network = self.decoder_bottleneck(x=network, output_depth=64,
                    scope="bottleneck_5_2")
        print(network.get_shape().as_list())
    
    
        # # layer 6:
        network = self.decoder_bottleneck(x=network, output_depth=16,
                    scope="bottleneck_6_0", upsampling=True,
                    pooling_indices=pooling_indices_1, output_shape=inputs_shape_1)
        print(network.get_shape().as_list())
    
        network = self.decoder_bottleneck(x=network, output_depth=16,
                    scope="bottleneck_6_1")
        print(network.get_shape().as_list())
    
    
    
        # fullconv:
        network = tf.contrib.slim.conv2d_transpose(network, self.no_of_classes,
                    [2, 2], stride=2, scope="fullconv", padding="SAME")
        print(network.get_shape().as_list())
    
        self.logits = network
