import tensorflow as tf
#import cv2
import tensorflow.contrib as tf_contrib

import numpy as np

def PReLU(x, scope):
    # PReLU(x) = x if x > 0, alpha*x otherwise

    alpha = tf.get_variable(scope + "/alpha", shape=[1],
                initializer=tf.constant_initializer(0), dtype=tf.float32)

    output = tf.nn.relu(x) + alpha*(x - abs(x))*0.5

    return output

# function for 2D spatial dropout:
def spatial_dropout(x, drop_prob):
    # x is a tensor of shape [batch_size, height, width, channels]

    keep_prob = 1.0 - drop_prob
    input_shape = x.get_shape().as_list()

    batch_size = input_shape[0]
    channels = input_shape[3]

    # drop each channel with probability drop_prob:
    noise_shape = tf.constant(value=[batch_size, 1, 1, channels])
    x_drop = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape)

    output = x_drop

    return output
def flatten(x) :
    return tf.layers.flatten(x)

weight_init = tf_contrib.layers.variance_scaling_initializer()
def fully_conneted(x, units, use_bias=True, name='fully_0'):
    with tf.variable_scope(name):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, use_bias=use_bias)

        return x

# function for spatial softmax
def spatial_softmax(x):
    """
    x is N H W C shaped image, 
    
    """
    N, H, W, C = x.get_shape().as_list()
#    print(N, H, W, C, " ---- > shape in sp soft")
    features = tf.reshape(tf.transpose(x, [0,3,1,2]), [N*C, H*W])
    softmax = tf.nn.softmax(features)
    softmax = tf.reshape(softmax, [N, C, H, W])
#    print(np.shape(softmax), 'aftert sotmax reshape')
    softmax = tf.transpose(softmax, [0, 2, 3, 1])
#    print(np.shape(softmax), 'after  softmax, transpose')
    return softmax

# getting intersection of union for loss, for the only classes greater than 0
# i am not sure if this works or not )))), in my case it worked ))) 
def get_iou_loss(masks, predictions, num_classes):
    """ here all calculations will be based on the class greater than 0, except accuracy"""
    shape = masks.get_shape().as_list()
    print(shape)
    accuracy, avgRecall, avgIOU = 0.0, 0.0, 0.0

    for i in range(shape[0]):
        numClass , numUnion = 1.0, 1.0
        recall = 0.0
        IOU = 0.0
        mask = tf.argmax(masks[i], -1)
        pred = tf.argmax(predictions[i], -1)
        accuracy = accuracy + tf.reduce_sum(tf.cast(tf.equal(mask,pred), tf.float32)) /(shape[1] * shape[2])
#        mask = tf.Print(mask, [mask], "mask : ")
        for c in np.arange(1, num_classes, 1):
            
            masks_sum = tf.reduce_sum(tf.cast(tf.equal(mask, c), tf.float32))
            
#            masks_sum = tf.Print(masks_sum, [masks_sum], "mask summ : ")
            
            predictions_sum = tf.reduce_sum(tf.cast(tf.equal(pred,c), tf.float32))

            numTrue = tf.reduce_sum(tf.multiply(tf.cast(tf.equal(pred,c), tf.float32),  tf.cast(tf.equal(mask,c), tf.float32)))
            unionSize = masks_sum + predictions_sum -numTrue
#            unionSize = tf.Print(unionSize, [unionSize], "union size : ")
            
            maskhaslabel = tf.cond(tf.greater(masks_sum,0), lambda: True,lambda: False)
            predhaslabel = tf.cond(tf.greater(predictions_sum,0), lambda: True, lambda:False)
            predormaskexistlabel = tf.cond(tf.logical_or(maskhaslabel, predhaslabel), lambda:True,lambda:False)
            IOU = tf.cond( predormaskexistlabel , lambda: IOU + numTrue/ unionSize, lambda: IOU)
            numUnion = tf.cond(predormaskexistlabel, lambda: numUnion + 1, lambda:numUnion)
            recall = tf.cond(maskhaslabel  ,lambda: recall + numTrue/masks_sum, lambda:recall)
#            recall = tf.Print(recall, [recall], "recall : ")
            numClass = tf.cond(maskhaslabel ,lambda: numClass + 1, lambda:numClass)
#            numClass = tf.Print(numClass, [numClass], "numClass : ")
        recall = recall / numClass
        avgRecall = avgRecall + recall
        IOU= IOU / numUnion
        avgIOU = avgIOU + IOU
    accuracy = accuracy / shape[0]
    avgRecall = avgRecall / shape[0]
    avgIOU = avgIOU / shape[0]
    iou_loss = 1 - avgIOU    
    return accuracy, avgIOU  , avgRecall, iou_loss

#
#def get_cross_entropy_loss(masks, predictions, num_classes):
#    shape = masks.get_shape().as_list()
#    print(shape)
#    background = 
#    for i in range(shape[0]):
#        mask = tf.argmax(masks[i], -1)
#        pred = tf.argmax(predictions[i], -1)
#            
    
def l2_loss(x1, x2):
    """
    this is Ld which is L2 loss in our case
    """
    loss =  tf.reduce_mean(tf.squared_difference(x1,x2))
    return loss

def distill_loss(layer1,layer2, shape=[288, 800]):
    
    #print(layer1.get_shape().as_list(), layer2.get_shape().as_list())
    
    layer1 = tf.expand_dims(layer1, -1)
    layer2 = tf.expand_dims(layer2, -1)
    #print(layer1.get_shape().as_list(), layer2.get_shape().as_list())
    
    
    layer1 = tf.image.resize_images(layer1, shape, method=tf.image.ResizeMethod.BILINEAR)
    layer2 = tf.image.resize_images(layer2, shape, method=tf.image.ResizeMethod.BILINEAR)
    #print(layer1.get_shape().as_list(), layer2.get_shape().as_list())
    
    layer1 = spatial_softmax(layer1)
    layer2 = spatial_softmax(layer2)
    #print(layer1.get_shape().as_list(), layer2.get_shape().as_list())
    
    distill_loss = l2_loss(layer1, layer2)
    return distill_loss
        
# function for unpooling max_pool:
def max_unpool(inputs, pooling_indices, output_shape=None, k_size=[1, 2, 2, 1]):
    # NOTE! this function is based on the implementation by kwotsin in
    # https://github.com/kwotsin/TensorFlow-ENet

    # inputs has shape [batch_size, height, width, channels]

    # pooling_indices: pooling indices of the previously max_pooled layer

    # output_shape: what shape the returned tensor should have

    pooling_indices = tf.cast(pooling_indices, tf.int32)
    input_shape = tf.shape(inputs, out_type=tf.int32)

    one_like_pooling_indices = tf.ones_like(pooling_indices, dtype=tf.int32)
    batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
    batch_range = tf.reshape(tf.range(input_shape[0], dtype=tf.int32), shape=batch_shape)
    b = one_like_pooling_indices*batch_range
    y = pooling_indices//(output_shape[2]*output_shape[3])
    x = (pooling_indices//output_shape[3]) % output_shape[2]
    feature_range = tf.range(output_shape[3], dtype=tf.int32)
    f = one_like_pooling_indices*feature_range

    inputs_size = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, inputs_size]))
    values = tf.reshape(inputs, [inputs_size])

    ret = tf.scatter_nd(indices, values, output_shape)

    return ret

# function for colorizing a label image:
def label_img_to_color(img):
    label_to_color = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [ 0, 255, 0],
        3: [0, 0, 255],
        4: [255,255,0],
#        5: [153,153,153],
#        6: [250,170, 30],
#        7: [220,220,  0],
#        8: [107,142, 35],
#        9: [152,251,152],
#        10: [ 70,130,180],
#        11: [220, 20, 60],
#        12: [255,  0,  0],
#        13: [  0,  0,142],
#        14: [  0,  0, 70],
#        15: [  0, 60,100],
#        16: [  0, 80,100],
#        17: [  0,  0,230],
#        18: [119, 11, 32],
#        19: [81,  0, 81]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color
