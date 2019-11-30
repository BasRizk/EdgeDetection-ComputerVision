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

def print_progress(iteration_type, iteration_value, end_value = 0, upper_bound_exist = False):
    if(upper_bound_exist):
        iteration_value = np.around((iteration_value/end_value)*100,
                                    decimals = 1)
    print( '\r ' + iteration_type + ' %s' % (str(iteration_value)),
              end = '\r')

# =============================================================================
# ------------------------------Problem 1
# =============================================================================
    
# 1. Determine the size of the LoG mask
def compute_log_kernel_size(sigma):
    print("Computing log kernel size..")
    if(int(3*sigma) < (3*sigma)):
        return int(2 * int((3*sigma) + 1) + 1)
    return int(2*int(3*sigma) + 1)

# 2. Form the LoG mask
# NOTE: [0,0] is the middle cell
def compute_log_kernel(kernel_size, sigma):
    print("Computing log kernel..")
    log_kernel = np.zeros((kernel_size, kernel_size))
    kernel_shift = int(kernel_size/2)
    for x in range(-kernel_shift, kernel_shift + 1):
        for y in range(-kernel_shift, kernel_shift + 1):
            log_x_y = 1
            redundant_part =\
                -((x**2 + y**2)/(2*sigma**2))
            log_x_y = log_x_y * -1/(np.pi*(sigma**4))
            log_x_y = log_x_y * (1 + redundant_part)
            log_x_y = log_x_y * (np.e**redundant_part)
            log_kernel[x+kernel_shift][y+kernel_shift] =\
                log_x_y
    return log_kernel

def convolve(org_img_array, kernel_window):
    print("Convolving with kernel window..")
    log_convolved_array = np.zeros(org_img_array.shape)
    padding_value = kernel_window.shape[0]//2
    progress_end_value = (org_img_array.shape[0] - 2*padding_value)**2
    progress_count = 0
    for x in range(padding_value, org_img_array.shape[0] - padding_value):
        for y in range(padding_value, org_img_array.shape[1] - padding_value):
            progress_count += 1
            print_progress("Convolution Progress: ", progress_count,
                           progress_end_value, upper_bound_exist = True)
            sub_convolve_result = 0
            for i in range(-padding_value, padding_value + 1):
                for j in range(-padding_value, padding_value + 1):
                    sub_convolve_result +=\
                        kernel_window[i+padding_value][j+padding_value]*\
                        org_img_array[x+i][y+j]
            log_convolved_array[x][y] =\
                sub_convolve_result
                
    print()
    return log_convolved_array

def edge_detect(org_img_array, h_1, h_2,
                threshold = 0.1, thresholding = False, max_value = 1):
    print("Running edge detector..")
    h_1_convolved_array = convolve(org_img_array, h_1)
    h_2_convolved_array = convolve(org_img_array, h_2)
    gradient_magnitude_array = np.sqrt(h_1_convolved_array**2 + h_2_convolved_array**2)
    
    if thresholding:
        return np.where(gradient_magnitude_array > threshold, max_value, 0)
    else:
        return gradient_magnitude_array

def prewitt_edge_detection(org_img_array, threshold = 0.1, thresholding = False):
    print("Initializing Prewitt edge detector..")
    h_1 = np.array([[ 1,  1,  1],
                    [ 0,  0,  0],
                    [-1, -1, -1]])
    h_2 = np.array([[-1,  0,  1],
                    [-1,  0,  1],
                    [-1,  0,  1]])
    
    return edge_detect(org_img_array, h_1, h_2, threshold, thresholding)

def threshold_array(some_array, threshold = 0.5, lower_value = 0, upper_value = 1):
    print("Thresholding array with threshold %s, to get values either %s or %s" 
             % (str(threshold), str(lower_value), str(upper_value)))
    return np.where(some_array > threshold, upper_value, lower_value)


def zero_crossings_1(some_array):
    
    def has_zero_neighbor(some_array, x, y):
        for delta_x in range(-1,2):
          for delta_y in range(-1,2):
             if some_array[x+delta_x, y+delta_y] == 0:
                return True
        return False

    print("Calculating zero-crossings (TECHNIQUE 1)..")
    thresholded_array = threshold_array(some_array, threshold = 0)
    zero_crossings_array = np.zeros(some_array.shape)

    for x in range(some_array.shape[0]):
        for y in range(some_array.shape[1]):
            if thresholded_array[x][y] != 0:
                zero_crossings_array[x][y] =\
                    has_zero_neighbor(thresholded_array, x, y)
    return zero_crossings_array

def zero_crossings_2(some_array):
    print("Calculating zero-crossings (TECHNIQUE 2)..")
    some_array_t = some_array.T
    zero_crossings_array = np.zeros(some_array.shape)

    for x in range(some_array.shape[0]):
        zero_crossings_array[x][:-1] = np.where(abs(np.diff(np.sign(some_array[x]))) > 0, 1, 0)
    
    zero_crossings_array = zero_crossings_array.T
    for y in range(some_array.shape[0]):
        zero_crossings_array[y][:-1] = np.where(abs(np.diff(np.sign(some_array_t[y]))) > 0, 1, 0)
    zero_crossings_array = zero_crossings_array.T
                
    return zero_crossings_array

def two_arrays_meet(array_1, array_2, assign_value = 255):
    crossed_array = np.zeros(array_1.shape)
    for x in range(crossed_array.shape[0]):
        for y in range(crossed_array.shape[1]):
            if array_1[x][y] and array_2[x][y] :
                crossed_array[x][y] = assign_value
    return crossed_array
        
# LoG Algorithm
#    1. Determine the size of the LoG mask
#    2. Form the LoG mask
#    3. Convolve the LoG mask with the image
#    4. Search for the zero crossings in the result of the convolution
#    5. If there is sufficient edge evidence from a first derivative operator,
#    form the edge image by setting the position of zero crossing to 1
#    and other positions to 0
def log_edge_detection(org_img_array, sigma,
                       threshold = 0.1, automatic_thresholding = False):
    print("=> Log Edge Detection with sigma = %s" % str(sigma))
    #    1. Determine the size of the LoG mask
    log_kernel_size = compute_log_kernel_size(sigma)
    #    2. Form the LoG mask
    log_kernel_window = compute_log_kernel(log_kernel_size, sigma)
    #    3. Convolve the LoG mask with the image
    log_convolved_array = convolve(org_img_array, log_kernel_window)
    #    4. Search for the zero crossings in the result of the convolution
    #    5. If there is sufficient edge evidence from a first derivative operator,
    #    form the edge image by setting the position of zero crossing to 1
    #    and other positions to 0

    # METHOD 1,2: USING ZERO CROSSING METHOD
#    zero_crossings_array = zero_crossings_1(log_convolved_array)
#    first_deriv_array = prewitt_edge_detection(org_img_array,
#                                               threshold,
#                                               thresholding = True)
#    
#    edge_img_array = two_arrays_meet(first_deriv_array, zero_crossings_array)
#    edge_detect_img = Image.fromarray(edge_img_array).convert("L")


    # METHOD 3: APPLYING PREWITT DIRECTLY ON THE CONVOLVED ARRAY METHOD    
    log_deriv_array = prewitt_edge_detection(log_convolved_array,
                                             thresholding = False)
#    log_deriv_array_normalized = normalize(log_deriv_array,
#                                   min_value = 0,
#                                   max_value = 1)
    if(automatic_thresholding):
        threshold = np.mean(log_deriv_array) +\
                    np.sqrt(np.var(log_deriv_array))
    thresholded_edges = threshold_array(log_deriv_array,
                                        threshold,
                                        lower_value = 0,
                                        upper_value = 255)
    edge_detect_img = Image.fromarray(thresholded_edges).convert("L")
    edge_detect_img
    return log_convolved_array, edge_detect_img
    

# =============================================================================
# ------------------------------Problem 2
# =============================================================================
def normalize(array, min_value = 0, max_value = 1):
    print("Normalizing (Scaling) array with min_value = %s, and max_value = %s"
          % (str(min_value), str(max_value)))
    return np.interp(array, (array.min(), array.max()), (min_value, max_value))

def sharpen_image(org_img_array, sharpen_value = 50):
    print("=> Sharpening Image using edge detector," + 
          "with sharpen_value = %s" % str(sharpen_value))
    
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
            
    sharpen_img = Image.fromarray(sharpen_img_array).convert("L")
    return sharpen_img_array, sharpen_img   
    
    


# =============================================================================
# Initializations
# =============================================================================
img_filepath = "Cameraman.tif"
org_img = Image.open(img_filepath)
org_img_array = np.array(org_img)
#threshold = 0.1

# Output Laplacian Of Gausian edge detecteded Images
for sigma in [2, 3, 4]:    
    img_array, image = log_edge_detection(org_img_array, sigma,
                                          automatic_thresholding = True)
    image.save(img_filepath + str(sigma) + "_log.jpg")

# Output sharpened image
sharpen_value = 50
img_array, image = sharpen_image(org_img_array, sharpen_value)
image.save(img_filepath + "_sharpen.jpg")
 








