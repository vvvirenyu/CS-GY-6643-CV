#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:19:11 2019

@author: vvviren
"""

import numpy as np
import sys
import cv2
import math


def convolve(img):
    mask = (1.0/140.0)*np.array([(1,1,2,2,2,1,1),
                    (1,2,2,4,2,2,1),
                    (2,2,4,8,4,2,2),
                    (2,4,8,16,8,4,2),
                    (2,2,4,8,4,2,2),
                    (1,2,2,4,2,2,1),
                    (1,1,2,2,2,1,1)])

    con = np.zeros(shape=img.shape)

    for row in range(img.shape[0]-6):
        for col in range(img[row].size-6):
            temp = 0
            for i in range (0,7):
                for j in range (0,7):
                    temp = temp + img[row+i,col+j]*mask[i,j]
            con[row+3,col+3] = temp
    return con

def sobel(img):
    #Mask to calculate x gradient
    px = (1.0/3.0)*np.array([(-1,0,1),
                (-2,0,2),
                (-1,0,1)])
    #Mask to calculate y gradient
    py = (1.0/3.0)*np.array([(1,2,1),
                (0,0,0),
                (-1,-2,-1)])

    #initialize matrices to store the value of x,y gradient and gradient angle
    gradientx = np.zeros(shape=img.shape)
    gradientx_normalized = np.zeros(shape=img.shape)
    gradienty = np.zeros(shape=img.shape)
    gradienty_normalized = np.zeros(shape=img.shape)
    gradient = np.zeros(shape=img.shape)
    gradient_angle = np.zeros(shape=img.shape)

    #find the gradient values by perfoeming convolution
    for row in range(3,img.shape[0]-5):
        for col in range(3,img[row].size-5):
            tgx = 0
            tgy = 0
            for i in range (0,3):
                for j in range (0,3):
                    tgx = tgx + img[row+i,col+j]*px[i,j]
                    tgy = tgy + img[row+i,col+j]*py[i,j]
            gx = tgx
            gradientx[row+1,col+1] = gx
            gradientx_normalized[row+1,col+1] = abs(gx)
            gy = tgy 
            gradienty[row+1,col+1] = gy
            gradienty_normalized[row+1,col+1] = abs(gy)
            #normalize by dividing by sqrt(2)
            gradient[row+1,col+1]=((gx**2+gy**2)**(0.5))/(1.4142)
            angle = 0
            if(gx == 0):
                if( gy > 0):
                    angle = 90
                else:
                    angle = -90
            else:
                angle = math.degrees(math.atan(gy/gx))
            if (angle < 0):
                angle = angle + 360
            gradient_angle[row+1,col+1]  = angle

    return [gradientx_normalized, gradienty_normalized, gradient, gradient_angle]


def nonMaximaSuppression(gradient, gradient_angle):
    #initialize matrix with zero to store the output of non maxima suppression
    nms = np.zeros(shape=img.shape)
    #histogram is an array to store the number of pixels with a particular gray level value
    histogram = np.zeros(shape=(256))
    ep = 0 #number of edge pixels
    for row in range(5,img.shape[0]-5): #-4
        for col in range(5,img[row].size-5):
            # print(gradient_angle[row,col])
            theta = gradient_angle[row,col]
            gr = gradient[row,col] #gradient at current pixel
            val = 0 
            #sector zero
            if( 0 <= theta <= 22.5 or  157.5 < theta <= 202.5 or 337.5 < theta <= 360):
                val = gr if (gr > gradient[row,col+1] and gr > gradient[row,col-1]) else 0
            #sector one
            elif ( 22.5 < theta <= 67.5 or  202.5 < theta <= 247.5):
                val = gr if (gr > gradient[row+1,col-1] and gr > gradient[row-1,col+1]) else 0
            #sector two
            elif ( 67.5 < theta <= 112.5 or  247.5 < theta <= 292.5):
                val = gr if (gr > gradient[row+1,col] and gr > gradient[row-1,col]) else 0
            #sector three
            elif ( 112.5 < theta <= 157.5 or  292.5 < theta <= 337.5):
                val = gr if (gr > gradient[row+1,col+1] and gr > gradient[row-1,col-1]) else 0

            nms[row,col] = val
            #if value is greater than zero after non maxima suppression
            if(val>0):
                #increment total number of edge pixels
                ep = ep + 1 
                try:
                    #increment the number of pixels at that gray level value by one
                    histogram[int(val)] += 1
                except:
                    print("gray level value is out of range, ", val)
    print("total number of edge pixels is: ", ep)
    return [nms, histogram, ep]


def doubleThresholding(img, t):
    t1=t
    t2=2*t
    
    M,N =img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= t2)
    zeros_i, zeros_j = np.where(img < t1)
    
    weak_i, weak_j = np.where((img <= t2) & (img >= t1))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (res[i,j] == weak):
                try:
                    if ((res[i+1, j-1] == strong) or (res[i+1, j] == strong) or (res[i+1, j+1] == strong)
                        or (res[i, j-1] == strong) or (res[i, j+1] == strong)
                        or (res[i-1, j-1] == strong) or (res[i-1, j] == strong) or (res[i-1, j+1] == strong)):
                        res[i, j] = strong
                    else:
                        res[i, j] = 0
                except IndexError as e:
                    print ("lol\n")
                    pass
    return res

    
if __name__ == "__main__":
    file=sys.argv[1]
    img = cv2.imread("./"+file+".bmp",0)
    con = convolve(img)
    #cv2.imwrite(file+"_gaussian.bmp",con)
    gradients = sobel(con)
    gradient_x = gradients[0]
    gradient_y = gradients[1]
    gradient = gradients[2]
    gradient_angle = gradients[3]
    #cv2.imwrite(file+"_gradientX.bmp",gradient_x)
    #cv2.imwrite(file+"_gradientY.bmp",gradient_y)
    #cv2.imwrite(file+"_sobel.bmp",gradient)
    suppressed = nonMaximaSuppression(gradient, gradient_angle)
    nms = suppressed[0]
    #cv2.imwrite(file+"_nms.bmp",nms)
    histogram = suppressed[1]
    edge_pixels = suppressed[2]
    cv2.imwrite(file+"_doubleT_11.bmp", doubleThresholding(np.copy(nms),11))

