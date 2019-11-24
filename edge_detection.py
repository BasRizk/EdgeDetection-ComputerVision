# -*- coding: utf-8 -*-
#
#
# Authors: Ibram Medhat, Basem Rizk
#
# Assignment 2
# DMET901 - Computer Vision 
# The German University Of Cairo
#

import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# ------------------------------Problem 1
# =============================================================================
    
# 1. Determine the size of the LoG mask
def compute_log_kernel_size(sigma):
    if(int(3*sigma) < (3*sigma)):
        return int(2 * int((3*sigma) + 1) + 1)
    return int(2*int(3*sigma) + 1)

# 2. Form the LoG mask
# NOTE: [0,0] is the middle cell
def compute_log_kernel(kernel_size):
    log_kernel = np.zeros((kernel_size, kernel_size))
    kernel_shift = int(kernel_size/2)
    for x in range(-kernel_shift, kernel_shift + 1):
        for y in range(-kernel_shift, kernel_shift + 1):
            log_x_y = 1
            redundant_part =\
                -((x**2 + y**2)/(2*kernel_size**2))
            log_x_y = log_x_y * -1/(np.pi*(kernel_size**4))
            log_x_y = log_x_y * (1 + redundant_part)
            log_x_y = log_x_y * (np.e**redundant_part)
            log_kernel[x+kernel_shift][y+kernel_shift] =\
                log_x_y
    return log_kernel

def convolve(org_img_array, kernel_window):
    
    log_convolved_array = np.zeros(org_img_array.shape)
    padding_value = kernel_window.shape[0]//2
    for x in range(padding_value, org_img_array.shape[0] - padding_value):
        for y in range(padding_value, org_img_array.shape[1] - padding_value):
            sub_convolve_result = 0
            for i in range(-padding_value, padding_value + 1):
                for j in range(-padding_value, padding_value + 1):
                    sub_convolve_result +=\
                        kernel_window[i+padding_value][j+padding_value]*\
                        org_img_array[x+i][y+j]
            log_convolved_array[x][y] =\
                sub_convolve_result
    return log_convolved_array

def edge_detect(org_img_array, h_1, h_2, threshold):
    h_1_convolved_array = convolve(org_img_array, h_1)
    h_2_convolved_array = convolve(org_img_array, h_2)
    gradient_magnitude_array = np.sqrt(h_1_convolved_array**2 + h_2_convolved_array**2)
    
    return np.where(gradient_magnitude_array > threshold, 255, 0)

def prewitt_edge_detection(org_img_array, threshold):
    
    h_1 = np.array([[ 1,  1,  1],
                    [ 0,  0,  0],
                    [-1, -1, -1]])
    h_2 = np.array([[-1,  0,  1],
                    [-1,  0,  1],
                    [-1,  0,  1]])
    
    return edge_detect(org_img_array, h_1, h_2, threshold)

# LoG Algorithm
#    1. Determine the size of the LoG mask
#    2. Form the LoG mask
#    3. Convolve the LoG mask with the image
#    4. Search for the zero crossings in the result of the convolution
#    5. If there is sufficient edge evidence from a first derivative operator,
#    form the edge image by setting the position of zero crossing to 1
#    and other positions to 0
def log_edge_detection(org_img_array, sigma, threshold):
    #    1. Determine the size of the LoG mask
    log_kernel_size = compute_log_kernel_size(sigma)
    #    2. Form the LoG mask
    log_kernel_window = compute_log_kernel(log_kernel_size)
    #    3. Convolve the LoG mask with the image
    log_convolved_array = convolve(org_img_array, log_kernel_window)
    #    4. Search for the zero crossings in the result of the convolution
    # TODO
    
    #    5. If there is sufficient edge evidence from a first derivative operator,
    #    form the edge image by setting the position of zero crossing to 1
    #    and other positions to 0
    # TODO   
    prewitt_edge_array = prewitt_edge_detection(org_img_array, threshold)
    
    
    corner_detect_img = Image.fromarray(log_convolved_array)
    corner_detect_img = corner_detect_img.convert("L")
    return log_convolved_array, corner_detect_img
    
# =============================================================================
# ------------------------------Problem 2
# =============================================================================
def normalize(array, min_value = 0, max_value = 1):
    return np.interp(array, (array.min(), array.max()), (min_value, max_value))

def sharpen_image(org_img_array, sharpen_value = 50):
    edge_detector_kernel = np.array([[-1, -1, -1],
                                     [-1,  8, -1],
                                     [-1, -1, -1]])
                
    edge_detected_array = convolve(org_img_array, edge_detector_kernel)
    edge_detected_array = normalize(edge_detected_array, 0, sharpen_value)
    
    sharpen_img_array = np.zeros(org_img_array.shape)
    for x in range(sharpen_img_array.shape[0]):
        for y in range(sharpen_img_array.shape[1]):
            sharpen_value = org_img_array[x][y] + edge_detected_array[x][y]
            if(sharpen_value > 255):
                sharpen_value = 255
            sharpen_img_array[x][y] = sharpen_value
            
    sharpen_img = Image.fromarray(sharpen_img_array)
    sharpen_img = sharpen_img.convert("L")
    return sharpen_img_array, sharpen_img   
    
    


# =============================================================================
# Initializations
# =============================================================================
img_filepath = "Cameraman.tif"
org_img = Image.open(img_filepath)
org_img_array = np.array(org_img)
threshold = 0.1

# Output Laplacian Of Gausian edge detecteded Images
for sigma in [2, 3, 4]:    
    img_array, image = log_edge_detection(org_img_array, sigma, threshold)
    image.save(img_filepath + str(sigma) + "_log.jpg")

# Output sharpened image
sharpen_value = 50
img_array, image = sharpen_image(org_img_array, sharpen_value)
image.save(img_filepath + "_sharpen.jpg")
 








