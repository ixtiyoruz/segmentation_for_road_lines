#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:20:56 2019

@author: essys
"""

import json
class LabelFileError(Exception):
    pass

def get_shape(line_color, points, flags, fill_color, label, shape_type='linestrip'):
    shape_data = dict(
            line_color = line_color,
            shape_type = shape_type,
            points = points,
            flags = flags,
            fill_color = fill_color,
            label = label,
            )
    
    return shape_data
def change_pts_size(pts, size=None, size_change=None, crop=None):
    
    if(crop == None):
        h, w = size
        h_c, w_c = size_change
        for i in range(len(pts)):
            pts[i] = [pts[i][0]* w_c / w, pts[i][1] * h_c / h] 
    else:
        for i in range(len(pts)):
            pts[i] = [pts[i][0], pts[i][1]  +crop] 
    return pts

def save_label(filename, flags, shapes, lineColor, fillColor, imagePath, imageData, imageHeight, imageWidth):
    version = '3.16.7'
    data = dict(
            imagePath = imagePath,
            imageData = imageData,
            shapes = shapes,
            version = version,
            flags = flags,
            fillColor = fillColor,
            lineColor = lineColor,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            )
    print(data)
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise LabelFileError(e)
