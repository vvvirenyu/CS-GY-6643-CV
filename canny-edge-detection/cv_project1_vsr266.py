#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 18:19:11 2019

@author: vvviren
"""

# importing necessary libraries
import numpy as np
import sys
import cv2
import math
    
# Convolution of original image with Gaussian smoothing kernel(7x7)
def smoothing(img, shape_tuple):
    # defining gaussian kernel
    g_kernel = (1.0/140.0)*np.array([(1,1,2,2,2,1,1),
                    (1,2,2,4,2,2,1),
                    (2,2,4,8,4,2,2),
                    (2,4,8,16,8,4,2),
                    (2,2,4,8,4,2,2),
                    (1,2,2,4,2,2,1),
                    (1,1,2,2,2,1,1)])
    
    #initializing convolution output array
    con_output = np.zeros(shape=shape_tuple)
    
    #offset to take care of boundary pixels
    offset = math.floor(g_kernel.shape[0]/2)
    
    # looping each row and each column for smoothing
    for row in range(shape_tuple[0]-g_kernel.shape[0]-1):
        for col in range(shape_tuple[1]-6):
            one_sum = 0
            for i in range (g_kernel.shape[0]-1):
                for j in range (g_kernel.shape[1]-1):
                    # convoluting 7x7 sample from original image with 7x7 kernel
                    one_sum = one_sum + img[row+i,col+j]*g_kernel[i,j]
            con_output[row+offset, col+offset] = one_sum
            
    # returning convolution output and offset
    return [con_output, offset]

# convolution of smoothing output with sobel detector for edge detection
def gradient(img, shape_tuple, offset):
    
    # defining sobel detector in horizontal and vertical direction
    s_kernel_x = (1.0/4.0)*np.array([(-1,0,1), (-2,0,2), (-1,0,1)])
    s_kernel_y = (1.0/4.0)*np.array([(1,2,1), (0,0,0), (-1,-2,-1)])
    
    # initializing output gradient values in x and y direction
    gradientx_normalized = np.zeros(shape=shape_tuple)
    gradienty_normalized = np.zeros(shape=shape_tuple)
    # initializing output gradient magnitude 
    gradient = np.zeros(shape=shape_tuple)
    # initializing output gradient angle 
    gradient_angle = np.zeros(shape=shape_tuple)
    #offset to take care of boundary pixels
    this_offset = math.floor(s_kernel_x.shape[0]/2)
    
    #looping through required rows and columns of smoothing output
    for row in range(offset,shape_tuple[0]-(offset + s_kernel_x.shape[0]-1)):
        for col in range(offset,shape_tuple[1]-(offset + s_kernel_x.shape[1]-1)):
            tgx = 0
            tgy = 0
            for i in range (s_kernel_x.shape[0]):
                for j in range (s_kernel_x.shape[1]):
                    # assigning values to output gradient x 
                    tgx = tgx + img[row+i,col+j]*s_kernel_x[i,j]
                    # assigning values to output gradient y
                    tgy = tgy + img[row+i,col+j]*s_kernel_y[i,j]
            gx = tgx
            # output gradient x normalized
            gradientx_normalized[row+this_offset,col+this_offset] = abs(gx)
            gy = tgy 
            # output gradient y normalized
            gradienty_normalized[row+this_offset,col+this_offset] = abs(gy)
            # output gradient magnitude normalized
            gradient[row+this_offset,col+this_offset]=((gx**2+gy**2)**(0.5))/(1.4142)
            angle = 0
            
            # infinite check else calculate tan of gy/gx
            if(gx == 0):
                 angle = 90 if gy > 0  else -90
            else:
                angle = math.degrees(math.atan(gy/gx))
            
            # making all angles positive 
            if (angle < 0):
                angle = angle + 360
            
            # output gradient angle 
            gradient_angle[row+this_offset,col+this_offset]  = angle
    # updating offset
    offset+= this_offset
    
    # returning normalized gradient x and y , gradient magnitude, gradient angle and offset
    return [gradientx_normalized, gradienty_normalized, gradient, gradient_angle, offset]

# non maxima suppression
def nonMaximaSuppression(gradient, gradient_angle, shape_tuple, offset):
    
    # initializing output nms array
    nms = np.zeros(shape=shape_tuple)
    
    for row in range(offset,shape_tuple[0]-offset):
        for col in range(offset,shape_tuple[1]-offset):
            # reference pixel angle
            theta = gradient_angle[row,col]
            # reference gradient magnitude
            gr = gradient[row,col]
            val = 0 
            
            # if lies in sector 0, check left and right pixel to reference pixel
            if( 0 <= theta <= 22.5 
               or  157.5 < theta <= 202.5 
               or 337.5 < theta <= 360):
                val = gr if (gr > gradient[row,col+1] and gr > gradient[row,col-1]) else 0
                
            # if lies in sector 1, check top right and bottom left to reference pixel
            elif ( 22.5 < theta <= 67.5 
                  or  202.5 < theta <= 247.5):
                val = gr if (gr > gradient[row+1,col-1] and gr > gradient[row-1,col+1]) else 0
            
            # if lies in sector 2, check upper and lower pixel to reference pixel
            elif ( 67.5 < theta <= 112.5 
                  or  247.5 < theta <= 292.5):
                val = gr if (gr > gradient[row+1,col] and gr > gradient[row-1,col]) else 0
            
            # if lies in sector 3, check upper left and lower right pixel to reference pixel
            elif ( 112.5 < theta <= 157.5 
                  or  292.5 < theta <= 337.5):
                val = gr if (gr > gradient[row+1,col+1] and gr > gradient[row-1,col-1]) else 0
            
            # output nms 
            nms[row,col] = val
    
    # returning nms and gradient angle
    return [nms, gradient_angle]

# Double thresholding method
def doubleThresholding(img, ga, t, shape_tuple):
    # defining two thresholds to divide image into three different regions
    t1=t
    t2=2*t
    
    # initializing double thresholding output array
    res = np.zeros(shape=shape_tuple)
    
    # defining weak and strong pixel value
    weak, strong = np.int32(25), np.int32(255)
    
    # dividing image into three regions 
    # fetching indexes of pixels where value is >t2 to strong
    strong_i, strong_j = np.where(img > t2)
    
    # fetching indexes of pixels where value is <t1 to zeros
    zeros_i, zeros_j = np.where(img < t1)
    
    # fetching indexes of pixels where value is between t1 and t2 to weak
    weak_i, weak_j = np.where((img <= t2) & (img >= t1))
    
    # assigning value of strong and weak (zeros were alreay initialized as zero)
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    for i in range(1, shape_tuple[0]-1):
        for j in range(1, shape_tuple[1]-1):
            # reassigning value of pixel if its in the weak region based on 8-connected neightbor and gradient angle
            if (res[i,j] == weak):
                try:
                    # if any of the 8-connected neighbor is strong(255) and difference of the angle is <=45 (or between 315 and 360 for 4th quadrant angles)
                    if ((res[i+1, j-1] == strong and (abs(ga[i,j]-ga[i+1,j-1] <=45 or 315<= abs(ga[i,j]-ga[i+1,j-1]) <=360 )))
                    or (res[i+1, j] == strong and (abs(ga[i,j]-ga[i+1,j]) <=45 or 315<= abs(ga[i,j]-ga[i+1,j]) <=360))
                    or (res[i+1, j+1] == strong and (abs(ga[i,j]-ga[i+1,j+1]) <=45 or 315<= abs(ga[i,j]-ga[i+1,j+1]) <=360))
                    or (res[i, j-1] == strong and (abs(ga[i,j]-ga[i,j-1]) <=45 or 315<= abs(ga[i,j]-ga[i,j-1]) <=360))
                    or (res[i, j+1] == strong and (abs(ga[i,j]-ga[i,j+1])<=45 or 315<= abs(ga[i,j]-ga[i,j+1]) <=360))
                    or (res[i-1, j-1] == strong and (abs(ga[i,j]-ga[i-1,j-1]) <=45 or 315<= abs(ga[i,j]-ga[i-1,j-1]) <=360))
                    or (res[i-1, j] == strong and (abs(ga[i,j]-ga[i-1,j]) <=45 or 315<= abs(ga[i,j]-ga[i-1,j]) <=360))
                    or (res[i-1, j+1] == strong and (abs(ga[i,j]-ga[i-1,j+1]) <=45 or 315<= abs(ga[i,j]-ga[i-1,j+1]) <=360))):
                        res[i, j] = strong
                    else:
                        res[i, j] = 0
                #handling IndexError exception
                except IndexError as e:
                    print ("lol")
                    pass
    # returning binary image after double thresholding
    return res


# Entry point of code
if __name__ == "__main__":
    file=sys.argv[1]                                                # taking the file name as argument
    img = cv2.imread("./"+file+".bmp",0)                            # opening and reading image using cv2 library
    shape_tuple = img.shape                                         # assigning image shape to a variable
    con = smoothing(img, shape_tuple)                               # performing smoothing operation on the image
    cv2.imwrite(file+"_gaussian.bmp",con[0])                        # writing/printing output of smoothing
    gradients = gradient(con[0], shape_tuple, con[1])               # performing gradient operation 
    gradient_x = gradients[0]                                       # horizontal normalized gradient response
    gradient_y = gradients[1]                                       # vertical normalized gradient response
    gradient = gradients[2]                                         # gradient magnitude output
    gradient_angle = gradients[3]                                   # gradient angle output
    offset = gradients[4]                                           # offset
    cv2.imwrite(file+"_gradientX.bmp",gradient_x)                   # writing/printing gradient x
    cv2.imwrite(file+"_gradientY.bmp",gradient_y)                   # writing/printing gradient y
    cv2.imwrite(file+"_gradient_magnitude.bmp",gradient)                         # writing/printing gradient magnitude
    nms = nonMaximaSuppression(gradient, gradient_angle, shape_tuple, offset) # non maxima suppresion operation
    cv2.imwrite(file+"_nms.bmp",nms[0])                             # writing/printing nms response
    # writing/printing binary image after double thresholding with threshold 7,14 
    cv2.imwrite(file+"_doubleT_7.bmp", doubleThresholding(np.copy(nms[0]), gradient_angle,7, shape_tuple))
    # writing/printing binary image after double thresholding with threshold 11,22 
    cv2.imwrite(file+"_doubleT_11.bmp", doubleThresholding(np.copy(nms[0]), gradient_angle,11, shape_tuple))
    # writing/printing binary image after double thresholding with threshold 15,30 
    cv2.imwrite(file+"_doubleT_15.bmp", doubleThresholding(np.copy(nms[0]), gradient_angle,15, shape_tuple))

