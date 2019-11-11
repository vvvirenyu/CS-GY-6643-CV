#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 18:19:11 2019

@author: vvviren
"""

import numpy as np
import sys
import cv2
import math


def convolve(img, shape_tuple):

    g_kernel = (1.0/140.0)*np.array([(1,1,2,2,2,1,1),
                    (1,2,2,4,2,2,1),
                    (2,2,4,8,4,2,2),
                    (2,4,8,16,8,4,2),
                    (2,2,4,8,4,2,2),
                    (1,2,2,4,2,2,1),
                    (1,1,2,2,2,1,1)])
    
    con_output = np.zeros(shape=shape_tuple)
    offset = math.floor(g_kernel.shape[0]/2)
    for row in range(shape_tuple[0]-g_kernel.shape[0]-1):
        for col in range(shape_tuple[1]-6):
            one_sum = 0
            for i in range (g_kernel.shape[0]-1):
                for j in range (g_kernel.shape[1]-1):
                    one_sum = one_sum + img[row+i,col+j]*g_kernel[i,j]
            con_output[row+offset, col+offset] = one_sum

    return [con_output, offset]

def sobel(img, shape_tuple, offset):

    s_kernel_x = (1.0/4.0)*np.array([(-1,0,1), (-2,0,2), (-1,0,1)])
    s_kernel_y = (1.0/4.0)*np.array([(1,2,1), (0,0,0), (-1,-2,-1)])

    gradientx_normalized = np.zeros(shape=shape_tuple)
    gradienty_normalized = np.zeros(shape=shape_tuple)
    gradient = np.zeros(shape=shape_tuple)
    gradient_angle = np.zeros(shape=shape_tuple)
    
    this_offset = math.floor(s_kernel_x.shape[0]/2)
    for row in range(offset,shape_tuple[0]-(offset + s_kernel_x.shape[0]-1)):
        for col in range(offset,shape_tuple[1]-(offset + s_kernel_x.shape[1]-1)):
            tgx = 0
            tgy = 0
            for i in range (s_kernel_x.shape[0]):
                for j in range (s_kernel_x.shape[1]):
                    tgx = tgx + img[row+i,col+j]*s_kernel_x[i,j]
                    tgy = tgy + img[row+i,col+j]*s_kernel_y[i,j]
            gx = tgx
            gradientx_normalized[row+this_offset,col+this_offset] = abs(gx)
            gy = tgy 
            gradienty_normalized[row+this_offset,col+this_offset] = abs(gy)
            gradient[row+this_offset,col+this_offset]=((gx**2+gy**2)**(0.5))/(1.4142)
            angle = 0
            if(gx == 0):
                 angle = 90 if gy > 0  else -90
            else:
                angle = math.degrees(math.atan(gy/gx))
            if (angle < 0):
                angle = angle + 360
            gradient_angle[row+this_offset,col+this_offset]  = angle
    offset+= this_offset
    return [gradientx_normalized, gradienty_normalized, gradient, gradient_angle, offset]


def nonMaximaSuppression(gradient, gradient_angle, shape_tuple, offset):
    nms = np.zeros(shape=shape_tuple)
    
    for row in range(offset,shape_tuple[0]-offset):
        for col in range(offset,shape_tuple[1]-offset):
            theta = gradient_angle[row,col]
            gr = gradient[row,col]
            val = 0 
            if( 0 <= theta <= 22.5 or  157.5 < theta <= 202.5 or 337.5 < theta <= 360):
                val = gr if (gr > gradient[row,col+1] and gr > gradient[row,col-1]) else 0
            elif ( 22.5 < theta <= 67.5 or  202.5 < theta <= 247.5):
                val = gr if (gr > gradient[row+1,col-1] and gr > gradient[row-1,col+1]) else 0
            elif ( 67.5 < theta <= 112.5 or  247.5 < theta <= 292.5):
                val = gr if (gr > gradient[row+1,col] and gr > gradient[row-1,col]) else 0
            elif ( 112.5 < theta <= 157.5 or  292.5 < theta <= 337.5):
                val = gr if (gr > gradient[row+1,col+1] and gr > gradient[row-1,col-1]) else 0

            nms[row,col] = val
            
    return nms


def doubleThresholding(img, t, shape_tuple):
    t1=t
    t2=2*t
    
    res = np.zeros(shape=shape_tuple)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= t2)
    zeros_i, zeros_j = np.where(img < t1)
    weak_i, weak_j = np.where((img <= t2) & (img >= t1))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    for i in range(1, shape_tuple[0]-1):
        for j in range(1, shape_tuple[1]-1):
            if (res[i,j] == weak):
                try:
                    if ((res[i+1, j-1] == strong) or (res[i+1, j] == strong) or (res[i+1, j+1] == strong)
                        or (res[i, j-1] == strong) or (res[i, j+1] == strong)
                        or (res[i-1, j-1] == strong) or (res[i-1, j] == strong) or (res[i-1, j+1] == strong)):
                        res[i, j] = strong
                    else:
                        res[i, j] = 0
                except IndexError as e:
                    pass
    return res

    
if __name__ == "__main__":
    file=sys.argv[1]
    img = cv2.imread("./"+file+".bmp",0)
    shape_tuple = img.shape
    con = convolve(img, shape_tuple)
    cv2.imwrite(file+"_gaussian.bmp",con[0])
    gradients = sobel(con[0], shape_tuple, con[1])
    gradient_x = gradients[0]
    gradient_y = gradients[1]
    gradient = gradients[2]
    gradient_angle = gradients[3]
    offset = gradients[4]
    cv2.imwrite(file+"_gradientX.bmp",gradient_x)
    cv2.imwrite(file+"_gradientY.bmp",gradient_y)
    cv2.imwrite(file+"_sobel.bmp",gradient)
    nms = nonMaximaSuppression(gradient, gradient_angle, shape_tuple, offset)
    cv2.imwrite(file+"_nms.bmp",nms)
    cv2.imwrite(file+"_doubleT_7.bmp", doubleThresholding(np.copy(nms),7, shape_tuple))
    cv2.imwrite(file+"_doubleT_11.bmp", doubleThresholding(np.copy(nms),11, shape_tuple))
    cv2.imwrite(file+"_doubleT_15.bmp", doubleThresholding(np.copy(nms),15, shape_tuple))

