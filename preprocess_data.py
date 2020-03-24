import cv2
import pickle
#import os
import numpy as np
#import tensorflow as tf
from collections import namedtuple
#import random

project_dir = "./"

# (NOTE! this is taken from the official Cityscapes scripts:)
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (NOTE! this is taken from the official Cityscapes scripts:)
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      0 ,    'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'left lane 1'          ,  1 ,      1 ,    'void'            , 0       , False        , True         , (  255,  0,  0) ),
    Label(  'left lane 2'          ,  2 ,      2 ,    'void'            , 0       , False        , True         , (  0,  255,  0) ),
    Label(  'right lane 1'         ,  3 ,      3 ,    'void'            , 0       , False        , True         , (  0,  0,  255) ),
    Label(  'right lane 2'         ,  4 ,      4 ,    'void'            , 0       , False        , True         , (  255,  255,  0) ),
#    Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#    Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#    Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#    Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#    Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#    Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#    Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#    Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
#    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#    Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#    Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# create a function mapping id to trainId:

new_img_height = 512 # (the height all images fed to the model will be resized to)
new_img_width = 1024 # (the width all images fed to the model will be resized to)
no_of_classes = 5 # (number of object classes (road, sidewalk, car etc.))

case = 'train' # 'train' # 'val'

train_list_file='/home/essys/datasets/culane_dataset/extracted/list/train.txt'
val_list_file='/home/essys/datasets/culane_dataset/extracted/list/val.txt'

imgs = open(train_list_file).read().split('\n')		
imgs_val = open(val_list_file).read().split('\n')

input_names = []
output_names = []
dataset_dir = '/home/essys/datasets/culane_dataset/extracted'
output_dir = '/home/essys/datasets/culane_dataset/pixel10/'
for file in imgs[:-1]:
    input_names.append(dataset_dir+ file)
for file in imgs[:-1]:
    file = file.split(".jpg")[0] + ".png"
    output_names.append(output_dir + file)		
input_names_val = []
output_names_val = []
for file in imgs_val[:-1]:
    input_names_val.append(dataset_dir+ file)
for file in imgs_val[:-1]:
    file = file.split(".jpg")[0] + ".png"
    output_names_val.append(output_dir + file)		




# # save the validation data to disk:
pickle.dump(output_names_val,
            open(project_dir + "data/val_trainId_label_paths.pkl", "wb"))
pickle.dump(input_names_val,
            open(project_dir + "data/val_img_paths.pkl", "wb"))        

pickle.dump(input_names,
            open(project_dir + "data/train_img_paths.pkl", "wb"))
pickle.dump(output_names,
            open(project_dir + "data/train_trainId_label_paths.pkl", "wb"))



# compute the mean color channels of the train imgs:
print ("computing mean color channels of the train imgs")
no_of_train_imgs = len(input_names)
mean_channels = np.zeros((3, ))
for step, img_path in enumerate(input_names):
    if step % 100 == 0:
        print (step)

    img = cv2.imread(img_path, -1)

    img_mean_channels = np.mean(img, axis=0)
    img_mean_channels = np.mean(img_mean_channels, axis=0)

    mean_channels += img_mean_channels

mean_channels = mean_channels/float(no_of_train_imgs)

# # save to disk:
pickle.dump(mean_channels, open(project_dir + "data/mean_channels.pkl", "wb"))


# compute the class weights:
print ("computing class weights")
trainId_to_count = {}
for trainId in range(no_of_classes):
    trainId_to_count[trainId] = 0

# # get the total number of pixels in all train labels that are of each
# # object class:
for step, trainId_label_path in enumerate(output_names):
    if step % 100 == 0:
        print( step)

    # read the label image:
    trainId_label = cv2.imread(trainId_label_path, -1)

    for trainId in range(no_of_classes):
        # count how many pixels in the label image are of object class trainId:
        trainId_mask = np.equal(trainId_label, trainId)
        label_trainId_count = np.sum(trainId_mask)

        # add to the total count:
        trainId_to_count[trainId] += label_trainId_count


# # compute the class weights according to the paper:
class_weights = []
total_count = sum(trainId_to_count.values())
for trainId, count in trainId_to_count.items():
    trainId_prob = float(count)/float(total_count)
    trainId_weight = 1/np.log(1.02 + trainId_prob)
    class_weights.append(trainId_weight)

# save to disk:
pickle.dump(class_weights, open(project_dir + "data/class_weights.pkl", "wb"))




# saving the existence of labels of train
train_existance_label = []
for step, trainId_label_path in enumerate(output_names):
    if step % 100 == 0:
        print( step)

    # read the label image:
    trainId_label = cv2.imread(trainId_label_path, -1)
    existance = [0, 0, 0, 0]
    for trainId in np.arange(1, no_of_classes, 1):
        if(len(trainId_label[np.where(trainId_label == trainId)]) > 0):
            existance[trainId-1] =1
    train_existance_label.append(existance)

# saving the existence of labels of val
val_existance_label = []
for step, valId_label_path in enumerate(output_names_val):
    if step % 100 == 0:
        print( step)
    # read the label image:
    valId_label = cv2.imread(valId_label_path, -1)
    existance = [0, 0, 0, 0]
    for trainId in np.arange(1, no_of_classes, 1):
        if(len(valId_label[np.where(valId_label == trainId)]) > 0):
            existance[trainId-1] =1
    val_existance_label.append(existance)

pickle.dump(train_existance_label, open(project_dir + "data/train_existance_label.pkl", "wb"))
pickle.dump(val_existance_label, open(project_dir + "data/val_existance_label.pkl", "wb"))







## augment the train data by flipping all train imgs:
#no_of_train_imgs = len(input_names)
#print ("number of train imgs before augmentation: %d " % no_of_train_imgs)
#
#augmented_train_img_paths = []
#augmented_train_trainId_label_paths = []
#for step, (img_path, label_path) in enumerate(zip(input_names, output_names)):
#    if step % 100 == 0:
#        print (step)
#
#    augmented_train_img_paths.append(img_path)
#    augmented_train_trainId_label_paths.append(label_path)
#
#    # read the image:
#    img = cv2.imread(img_path, -1)
#
#    # flip the image and save to project_dir/data:
#    img_flipped = cv2.flip(img, 1)
#    img_flipped_path = img_path.split(".png")[0] + "_flipped.png"
#    cv2.imwrite(img_flipped_path, img_flipped)
#    augmented_train_img_paths.append(img_flipped_path)
#
#    # read the corresponding label image:
#    label_img = cv2.imread(label_path, -1)
#
#    # flip the label image and save to project_dir/data:
#    label_img_flipped = cv2.flip(label_img, 1)
#    label_img_flipped_path = label_path.split(".png")[0] + "_flipped.png"
#    cv2.imwrite(label_img_flipped_path, label_img_flipped)
#    augmented_train_trainId_label_paths.append(label_img_flipped_path)
#
## # randomly shuffle the augmented train data:
#augmented_train_data = zip(augmented_train_img_paths, augmented_train_trainId_label_paths)
#random.shuffle(augmented_train_data)
#random.shuffle(augmented_train_data)
#random.shuffle(augmented_train_data)
#random.shuffle(augmented_train_data)
#
## # save the augmented train data to disk:
#train_data = augmented_train_data
#train_img_paths, train_trainId_label_paths = zip(*train_data)
#cPickle.dump(train_img_paths,
#            open(project_dir + "data/train_img_paths.pkl", "w"))
#cPickle.dump(train_trainId_label_paths,
#            open(project_dir + "data/train_trainId_label_paths.pkl", "w"))
## train_img_paths = cPickle.load(open(project_dir + "data/train_img_paths.pkl"))
## train_trainId_label_paths = cPickle.load(open(project_dir + "data/train_trainId_label_paths.pkl"))
#
#no_of_train_imgs = len(train_img_paths)
#print ("number of train imgs after augmentation: %d " % no_of_train_imgs)
