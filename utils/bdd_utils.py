#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:35:53 2019

@author: essys
"""


import json
import numpy as np
import os 
import skimage
import cv2
import time 
import scipy.misc
from PIL import Image
dataset_dir = "/home/essys/datasets/bdd_dataset"
input_dir = 'labels'
mode = 'val'
images_folder = os.path.join(dataset_dir, 'images','100k',mode)
annotations = json.load(open(os.path.join(os.path.join(dataset_dir, input_dir), "bdd100k_labels_images_" + mode+".json")))
annotation_root = os.path.join(os.path.join(dataset_dir, "annotation"), input_dir)

def get_mask_from_polygons(polygon, height, width):
        mask = np.zeros([height, width,3],
                        dtype=np.uint8)
        #print ('mask shape',mask.shape)
        for i, p in enumerate(polygon):
#             print ('i',i)
#             print ('p',p)
            # Get indexes of pixels inside the polygon and set them to 1
            x=[]
            y=[]
            for d in p:
                for l in d['vertices']:  
                    x.append(l[0])
                    y.append(l[1])
#             print ('x',x)
#             print ('y',y)
#            rr, cc = skimage.draw.polygon(y,x, )
            print(x, y)
#            print(rr,cc)
#            mask[rr, cc] = [255,255,255]
#         print(mask)
        return mask
i = 0
categories = []
for a in annotations:
    i += 1
    start = time.time()
    
    categories.extend([d['category'] for d in a['labels']])
    # print(np.unique(categories))
    polygons=[d['poly2d'] for d in a['labels'] if d['category']=="lane"]

    image_path = os.path.join(images_folder, a['name'])
    height, width = [720, 1280]#image.shape[:2]
    mask = np.zeros([height, width,3],
                    dtype=np.uint8)
    mask = cv2.imread(image_path, -1)
    for i, p in enumerate(polygons):
        for d in p:
#            pt1 = d['vertices'][0]
#            pt2 = d['vertices'][1]
#            for l in d['vertices']:  
#               mask[int(l[1]), int(l[0]) ]  = [255,255,255]
            for k in np.arange(0, len(d['vertices']), 2):
                cv2.polylines(mask, np.int32([d['vertices'][k:k+2]]), isClosed=False,color=[255,255,255], thickness=1)

    # self.images.append(image_path)
#    img = get_mask_from_polygons(polygons,height, width) 
    # print(np.unique(img))
    img = np.array(mask)
    
    # print(np.unique(img))
#    mask_path = os.path.join('/media/ixtiyor/087E6CE67E6CCDCE/annotation/',input_dir, a['name'].split('.')[0] +".png" )
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(mask_path, img, [cv2.IMWRITE_JPEG_QUALITY, 255])
    # scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save(mask_path)
    # im = Image.fromarray(img,"RGB" )
    # im.save(mask_path)
    # read_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    # print(np.shape(img), type(img))
    # np.save(mask_path,img)
    # read_image =  np.load(mask_path)
    # print()
    # print(i, "qolgan vaqt " , (time.time()-start) * (10000-i) * 1./60, " min")
    # break
    cv2.imshow("img", img)
    if(cv2.waitKey(0) ==27):
        cv2.destroyAllWindows()
        break