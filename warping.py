#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:00:42 2019

@author: essys
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from utils import transforms
from utils.view_utils import blend
import imgaug as ia
from imgaug import augmenters as iaa
trainlistfile = '/home/essys/datasets/culane_dataset/extracted/list/train.txt'
vallistfile = '/home/essys/datasets/culane_dataset/extracted/list/val.txt'
dataset_dir = '/home/essys/datasets/culane_dataset/extracted'
woutput_dir = '/home/essys/datasets/culane_dataset/warped_output'

IMAGE_H = 590
IMAGE_W = 1640

def getperspectivetransform_matrix():
    top_h = 280
    r1, c1 = IMAGE_W // 2 -150, top_h
    r2, c2 = IMAGE_W // 2 +150, top_h
    r3, c3 = IMAGE_W+2400, IMAGE_H + 80
    r4, c4 = -2400, IMAGE_H+ 80
    a = [r1, c1]
    b = [r2, c2]
    c = [r3, c3]
    d = [r4, c4]
    
    src = np.float32([[r1, c1], [r2, c2], [r3, c3], [r4, c4]])
    
    # dst 
    r1, c1 = 0, 0
    r2, c2 = IMAGE_W, 0
    r3, c3 = IMAGE_W, IMAGE_H
    r4, c4 = 0, IMAGE_H

    trapezoid = np.asarray([a, b, c, d])
    trapezoid = np.expand_dims(trapezoid, 1).astype(np.int32)

    trapezoid_img = cv2.fillPoly(np.zeros(shape=(IMAGE_H, IMAGE_W, 3)),
                                 [trapezoid], color=(0, 0, 255))
    
    dst = np.float32([[r1, c1], [r2, c2], [r3, c3], [r4, c4]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
#    cv2.imshow("img", trapezoid_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    return M, Minv, trapezoid_img, top_h

def getLane_CULane(prob_map, y_px_gap, pts, class_id, resize_shape=None):
    """
    Arguments 
    ---------------
    prob_map: prob map for single lane np array size (h, w)
    resize_shape: reshape size target (H, W)
    
    Return:
    -------------------
    coords : x coords buttom up every y_px_gap px, 0 for non exist in resized shape
    """
    
#    Q = 3e-2 # bu bizning estimationimizning xatoligi
#    sz = 11
#    # allocate space for arrays
#    xhat=np.zeros(sz)      # a current estimate of x
#    P=np.zeros(sz)         # a current error estimate
#    xhatminus=np.zeros(sz) # a priori estimate of x
#    Pminus=np.zeros(sz)    # a priori error estimate
#    K=np.zeros(sz)         # gain or blending factor
#    
#    R = 4e-2
#
#    xhat[0] = 0.0
#    P[0] = 1.0

    if(resize_shape is None):
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape
    coords = np.zeros(pts)
    for i in range(pts):
        y = int(h - i * y_px_gap / H * h -1)
        
        if(y < 0):
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        
        num_of_pts = len(np.where(line ==class_id)[0])
        if(num_of_pts <= 0):
            continue
        if(num_of_pts > 0):
            id = round(num_of_pts/2)
            id = np.where(line ==class_id)[0][id]
        if(line[id] == class_id):
            coords[i] = int(id/ w * W)
#            if(i == 0):
#                xhat[i] = int(id)
#                Pminus[i] = 1.
#                K[i] = 1.
#                P[i] = 1.
#                coords[i] = xhat[i]
#            else:
#                xhatminus[i] = xhat[i-1]
#                Pminus[i] = P[i-1]+Q
#                K[i] = Pminus[i]/( Pminus[i]+R )
#                xhat[i] = xhatminus[i]+K[i]*(int(id)-xhatminus[i])
#                P[i] = (1-K[i])*Pminus[i]
#                coords[i] = xhat[i]
#            print(coords[i], int(id))
    if((coords > 0).sum() < 2):
        coords = np.zeros(pts)
    return coords

def prob2lines_CULane(prob_map, resize_shape=None, y_px_gap=20, pts=None, class_id=1, no_of_classes=5):
    """
    Arguments:
        -------
        seg_pred: np array size (h, w, 5)
        resize_shape: reshape size target (H, W)
        exist: list of existance, e.g [0, 1, 1, 0]
        smooth: whether to smooth the probability or not
        y_px_gap: y pixel gap for sampling
        pts: how many points for one line
        thresh: probability threshold
    Return:
    ------------
        coordinates [x, y] list of lanes , e.g : [[[9, 568], [50, 549]], [[630, 569],[647, 549]]]
        
    """
    if(resize_shape is None):
        resize_shape = prob_map.shape# seg_pred (h,w,5)
    h, w = prob_map.shape
    H, W = resize_shape
    coordinates = []
    if(pts is None):
        pts = round(h/2/y_px_gap)
    coords = getLane_CULane(prob_map, y_px_gap, pts, class_id, resize_shape)
#    print(coords)
    coordinates = [[coords[j],np.int32( H-1 - j*y_px_gap)] for j in range(pts) if coords[j] > 0]
    return coordinates
def label_to_warped_rgb(imglabel, M, colors= None,line_width = 10, pts = 200, toph=310, y_px_gap=20):
    if(colors == None):
        colors = [(0,0,0), (0,255,0), (0,255,255), (255,255,0), (255,0,0)]
    points_cls1 = np.array(prob2lines_CULane(imglabel, class_id=1, pts = pts, y_px_gap=y_px_gap))
    points_cls2 = np.array(prob2lines_CULane(imglabel, class_id=2, pts = pts, y_px_gap=y_px_gap))
    points_cls3 = np.array(prob2lines_CULane(imglabel, class_id=3, pts = pts, y_px_gap=y_px_gap))
    points_cls4 = np.array(prob2lines_CULane(imglabel, class_id=4, pts = pts, y_px_gap=y_px_gap))
    warped_imglabel = np.zeros((IMAGE_H, IMAGE_W, 3))
    if(len(points_cls1) > 1):
        points_cls1 = fix_points_on_top_row(points_cls1, y_px_gap, toph=toph)
        wpts1 = cv2.perspectiveTransform(np.float32([points_cls1]), M)
        warped_imglabel = cv2.polylines(warped_imglabel, [np.int32(wpts1)], False, colors[1], line_width)
    if(len(points_cls2) > 1):
        points_cls2 = fix_points_on_top_row(points_cls2, y_px_gap, toph=toph)
        wpts2 = cv2.perspectiveTransform(np.float32([points_cls2]), M)
        warped_imglabel = cv2.polylines(warped_imglabel, [np.int32(wpts2)], False, colors[2], line_width)
    if(len(points_cls3) > 1):
        points_cls3 = fix_points_on_top_row(points_cls3, y_px_gap, toph=toph)
        wpts3 = cv2.perspectiveTransform(np.float32([points_cls3]), M)
        warped_imglabel = cv2.polylines(warped_imglabel, [np.int32(wpts3)], False, colors[3], line_width)
    if(len(points_cls4) > 1):
        points_cls4 = fix_points_on_top_row(points_cls4, y_px_gap, toph=toph)
        wpts4 = cv2.perspectiveTransform(np.float32([points_cls4]), M)
        warped_imglabel = cv2.polylines(warped_imglabel, [np.int32(wpts4)], False, colors[4], line_width)
    #rgb1 = label_to_warped_rgb(train_trainId_label_paths[0])
    #cv2.imshow("img", rgb1); cv2.waitKey(0); cv2.destroyAllWindows()
#    train_existance_label = np.array([0, 0, 0, 0])
    
    return warped_imglabel

def rgb_to_onehot(rgb_arr, color_dict = None, num_classes= 0):
    if(color_dict == None):
        color_dict = {
                  0: (0,0,0),
                  1: (0,255,0),
                  2: (0,255,255),
                  3: (255,255,0),
                  4: (255,0,0)
                  }
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr
def fix_points_on_top_row(points, y_px_gap, toph=310):
    first_over_index = np.where(points[:, 1] < toph)[0]
    if(len(np.where(points[:, 1] < toph)[0]) == 0): return list(points)
    first_over_index = np.where(points[:, 1] < toph)[0][0]
#    print(first_over_index)
    diffx = points[:, 0][first_over_index -1] - points[:, 0][first_over_index]  
    pr = abs((points[first_over_index-1][1] - toph) / y_px_gap)
    
    npt = points[first_over_index-1] - np.array([diffx, y_px_gap]) * pr
    points = list(points[:first_over_index+1])
    points.append(np.array(npt))
    return points


def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
	"""
	Takes an image, gradient orientation, and threshold min/max values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Return the result
	return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
	"""
	Return the magnitude of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	"""
	Return the direction of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction,
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output


def hls_thresh(img, thresh=(100, 255)):
	"""
	Convert RGB to HLS and threshold to binary image using S channel
	"""
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return binary_output
def combined_thresh(img):
	abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=255)
	mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
	dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
	hls_bin = hls_thresh(img, thresh=(170, 255))

	combined = np.zeros_like(dir_bin)
	combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1] = 1

	return combined, abs_bin, mag_bin, dir_bin, hls_bin  # DEBUG
if __name__ == '__main__':  
    train_mean_channels = pickle.load(open("data/mean_channels.pkl", 'rb'))
    input_mean = train_mean_channels#[103.939, 116.779, 123.68] # [0, 0, 0]
    input_std = [1, 1, 1]
    ignore_label = 255
    img_height = 288
    img_width = 800
    scaler = transforms.GroupRandomScale(size=(0.8, 1.1), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST))
    cropper = transforms.GroupRandomCropRatio(size=(img_width, img_height))
    rotater= transforms.GroupRandomRotation(degree=(-1, 1), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=(input_mean, (0, )))
    normalizer = transforms.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, )))
    seq = iaa.Sequential([
    # iaa.AdditiveGaussianNoise(scale=(10, 30)),
    iaa.AddToHueAndSaturation((-10, 10)),  # change their color
    iaa.ElasticTransformation(alpha=12, sigma=9),  # water-like effect
    # iaa.CoarseDropout((0.00001, 0.0001), size_percent=0.1)  # set large image areas to zero
    ], random_order=True)
    project_dir = "./"
    data_dir = project_dir + "data/"
    colors = [(0,0,0), (0,255,0), (0,255,255), (255,255,0), (255,0,0)]
    case = 'val'#'val' #'train'
    train_img_paths = np.array(pickle.load(open(data_dir + "train_img_paths.pkl", 'rb')))
    train_trainId_label_paths = np.array(pickle.load(open(data_dir + "train_trainId_label_paths.pkl", 'rb')))
    train_existance_labels = np.array(pickle.load(open(data_dir + "train_existance_label.pkl", 'rb')))
    trlist = train_img_paths
    M,Minv, trapezoid_img, toph = getperspectivetransform_matrix()
    # this is for saving warped bird eye images make it  true if you want
    # cv2.namedWindow("warped");
    # cv2.moveWindow("warped", 20,20);
    cv2.namedWindow("img");
    cv2.moveWindow("img", 20,520);
    if(True):
        i = 0
        while(True):#
            i =i + 1
            if(i > len(train_img_paths)-1): break
            directory ='driver_161_90frame' #'driver_23_30frame' # 'driver_182_30frame' # 'driver_161_90frame'
            directory1 = '06040056_1018.MP4'
            if(train_img_paths[i].split('/')[-3] == directory and train_img_paths[i].split('/')[-2] == directory1):
                
                ktmp = train_img_paths[0].split('/')[-3] + "/" + train_img_paths[i].split('/')[-2] + "/00450.jpg"
                ltmp = train_img_paths[0].split('/')[-3] + "/" + train_img_paths[i].split('/')[-2] + train_img_paths[i].split('/')[-1]
                # if(not ktmp == ltmp): continue
                # else: print(ktmp, ltmp)
                img = cv2.imread(trlist[i])
                imglabel = cv2.imread(train_trainId_label_paths[i], -1)
                img = seq.augment_images(np.expand_dims(img, 0))
                img= img[0]
                points_cls1 = np.array(prob2lines_CULane(imglabel, class_id=1, pts = 100, y_px_gap=20))
                points_cls2 = np.array(prob2lines_CULane(imglabel, class_id=2, pts = 100, y_px_gap=20))
                points_cls3 = np.array(prob2lines_CULane(imglabel, class_id=3, pts = 100, y_px_gap=20))
                points_cls4 = np.array(prob2lines_CULane(imglabel, class_id=4, pts = 100, y_px_gap=20))
                
                # warped_imglabel = np.zeros((IMAGE_H, IMAGE_W, 3))#cv2.warpPerspective(imglabel, M, (IMAGE_W, IMAGE_H),  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
                
                # warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H),  cv2.INTER_NEAREST) # Image warping
    
                # img_tmp = np.zeros((IMAGE_H, IMAGE_W, 3))
                line_width = 5
                if(len(points_cls1) > 0):
                    img = cv2.polylines(img, [np.int32(points_cls1)], False, colors[1], line_width)
                #     points_cls1 = fix_points_on_top_row(points_cls1, 20)
                #     # if(len(points_cls1) > 0):
                #         # wpts1 = cv2.perspectiveTransform(np.float32([points_cls1]), M)
                #         # warped_img = cv2.polylines(warped_img, [np.int32(wpts1)], False, colors[1], line_width)
                if(len(points_cls2) > 0):
                    img = cv2.polylines(img, [np.int32(points_cls2)], False, colors[2], line_width)
                #     points_cls2 = fix_points_on_top_row(points_cls2, 20)
                #     # if(len(points_cls2) > 0):
                #     #     wpts2 = cv2.perspectiveTransform(np.float32([points_cls2]), M)
                #     #     warped_img = cv2.polylines(warped_img, [np.int32(wpts2)], False, colors[2], line_width)
                    
                if(len(points_cls3) > 0):
                    img = cv2.polylines(img, [np.int32(points_cls3)], False, colors[3], line_width)
                #     points_cls3 = fix_points_on_top_row(points_cls3, 20)
                #     # if(len(points_cls3) > 0):
                #     #     wpts3 = cv2.perspectiveTransform(np.float32([points_cls3]), M)
                #     #     warped_img = cv2.polylines(warped_img, [np.int32(wpts3)], False, colors[3], line_width)
                if(len(points_cls4) > 0):
                    img = cv2.polylines(img, [np.int32(points_cls4)], False, colors[4], line_width)
                #     points_cls4 = fix_points_on_top_row(points_cls4, 20)
                #     # if(len(points_cls4) > 0):
                #     #     wpts4 = cv2.perspectiveTransform(np.float32([points_cls4]), M)
                #     #     warped_img = cv2.polylines(warped_img, [np.int32(wpts4)], False, colors[4], line_width)
                    
                img = cv2.resize(img, (IMAGE_W ,IMAGE_H ))
                
                
    #            img  = blend(trapezoid_img, img)
    #            warped_img  = blend(warped_img, img_tmp)
                # img_tmp = cv2.resize(img_tmp, (800, 288),cv2.INTER_NEAREST )
                img = cv2.resize(img, (800, 288),cv2.INTER_NEAREST )
    #            warped_img = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, IMAGE_H)) # Image warping
                # warped_img = cv2.resize(warped_img, (800, 288),cv2.INTER_NEAREST )
                # started from here
                # combined, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)
                # img = np.sum(img, 2)/ 765
    #                cv2.imwrite(img_beye_path, warped_img)
                if(i % 1000 == 0):print(i)
                # cv2.imshow("warped", warped_img)
                cv2.imshow("img", img)
                # cv2.imshow("combined", combined)
                # cv2.imshow("abs_bin", abs_bin)
                # cv2.imshow("mag_bin", mag_bin)
                # cv2.imshow("dir_bin", dir_bin)
                # cv2.imshow("hls_bin", hls_bin)
                # cv2.imshow("img", img)
                
#                cv2.imshow("img", img_tmp)
                k = cv2.waitKey(0)
                if(k == 27):
                    break
                else:
                    while(True):
                        if(k == 32):
                            break
                        if(k == 27):
                            break
                        k = cv2.waitKey(0)
                if(k == 27):
                    break
            else:
                continue
#                break
    cv2.destroyAllWindows()




#
#line_width = 2
#if(len(points_cls1) > 0):
#    wpts1 = cv2.perspectiveTransform(np.float32([points_cls1]), M)
#    wpts1[0].T[[0, 1]] = wpts1[0].T[[1, 0]]
#    cc1 = np.int32(np.transpose(wpts1[0], (1, 0)))
#    cc1[cc1>1640] = 1639
#    cc1[cc1<0] = 0
#    img_tmp2[cc1[0], cc1[1]] = 4
#if(len(points_cls2) > 0):
#    wpts2 = cv2.perspectiveTransform(np.float32([points_cls2]), M)
#    wpts2[0].T[[0, 1]] = wpts2[0].T[[1, 0]]
#    cc2 = np.int32(np.transpose(wpts2[0], (1, 0)))
#    cc2[cc2>1640] = 1639
#    cc2[cc2<0] = 0
#    img_tmp2[cc2[0], cc2[1]] = 4
#if(len(points_cls3) > 0):
#    wpts3 = cv2.perspectiveTransform(np.float32([points_cls3]), M)
#    wpts3[0].T[[0, 1]] = wpts3[0].T[[1, 0]]
#    cc3 = np.int32(np.transpose(wpts3[0], (1, 0)))
#    cc3[cc3>1640] = 1639
#    cc3[cc3<0] = 0
#    img_tmp2[cc3[0], cc3[1]] = 4
#if(len(points_cls4) > 0):
#    wpts4 = cv2.perspectiveTransform(np.float32([points_cls4]), M)
#    wpts4[0].T[[0, 1]] = wpts4[0].T[[1, 0]]
#    cc4 = np.int32(np.transpose(wpts4[0], (1, 0)))
#    cc4[cc4>1640] = 1639
#    cc4[cc4<0] = 0
#    img_tmp2[cc4[0], cc4[1]] = 4
#    
#    
#
#img_tmp[np.where(img_tmp2 == 1)[0], np.where(img_tmp2 == 1)[1], :] = [255,0,0]
#img_tmp[np.where(img_tmp2 == 2)[0], np.where(img_tmp2 == 2)[1], :] = [255,255,0]
#img_tmp[np.where(img_tmp2 == 3)[0], np.where(img_tmp2 == 3)[1], :] = [0,255,0]
#img_tmp[np.where(img_tmp2 == 4)[0], np.where(img_tmp2 == 4)[1], :] = [0,255,255]