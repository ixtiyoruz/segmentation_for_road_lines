#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:40:17 2019

@author: essys
"""

import argparse
import json


def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', help='path to the label dir', default="/home/essys/datasets/bdd_dataset/labels/bdd100k_labels_images_train.json")
    parser.add_argument('--det_path', help='path to output detection file',  default="/home/essys/datasets/bdd_dataset/labeldarknet/darknet_bdd100k_labels_images_train.json")
    args = parser.parse_args()

    return args


def label2det(frames):


    for frame in frames:
        iname = frame['name']
        fname = iname[:-4]
#Creates, opens, and adds to a txt file with the name of each image.jpg
        f = open("data/" + fname + ".txt","w+")
#For each sub label of each image, get the box2d variable
#Get the relative center point compared to the image size 1280/720
        for label in frame['labels']:
            if 'box2d' not in label:
                continue
            xy = label['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue
            X = xy['x1']/1280
            Y = xy['y1']/720
            MX = ((xy['x1'] + xy['x2']) / 2)/1280
            MY = ((xy['y1'] + xy['y2']) / 2)/720
            W = xy['x2']/1280
            H = xy['y2']/720
            if X > W or Y > H:
                continue
            lbl = -1
#provide a number corresponding to the category of sub label for darknet format.
            if(label['category'] == "bike"):
                lbl = 0
            if(label['category'] == "bus"):
                lbl = 1
            if(label['category'] == "car"):
                lbl = 2
            if(label['category'] == "motor"):
                lbl = 3
            if(label['category'] == "person"):
                lbl = 4
            if(label['category'] == "rider"):
                lbl = 5
            if(label['category'] == "traffic light"):
                lbl = 6
            if(label['category'] == "traffic sign"):
                lbl = 7
            if(label['category'] == "train"):
                lbl = 8
            if(label['category'] == "truck"):
                lbl = 9
            f.write(repr(lbl) + " " + repr(MX) + " " + repr(MY) + " " + repr(W-X) + " " + repr(H-Y) + '\n')

def convert_labels(label_path, det_path):
    frames = json.load(open(label_path, 'r'))
    det = label2det(frames)
    json.dump(det, open(det_path, 'w'), indent=4, separators=(',', ': '))


def main():
    args = parse_args()
    convert_labels(args.label_path, args.det_path)


if __name__ == '__main__':
    main()