# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy.signal import find_peaks
import csv

#results = []
#with open("data/Laying-Accelerometer.csv") as csvfile:
#    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
#    for row in reader: # each row is a list
#        results.append(row)
#print(results)


def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    mean = np.mean(window, axis=0)
    return mean

# print("mean")
# result = np.array(results)
# print(_compute_mean_features(result[:,0]))

# TODO: define functions to compute more features

def _median_feature(window):
    median = np.median(window)
    return median

# print("median")
# print(median_feature(results))

def _fft_feature(window):
    fft_win = np.fft.rfft(window)
    refined_fft_win = fft_win.astype(float)
    return refined_fft_win

# print("fft")
# print(fft_feature(results))

def _entropy_feature(window):
    hist_win = np.histogram(window, bins=10)
    distribution = hist_win[0]
    entropy_val = distribution.sum()
    return entropy_val

# print("entropy value")
# print(entropy_feature(results))

def _peak_feature(window):
    peaks = []
    mean_signal = np.average(window)
    peak_arr = find_peaks(window)
    for p in range (len(peak_arr)):
        if peak_arr[p] >= mean_signal:
            peaks.append(peak_arr[p])
    return len(peaks)


#print("peaks")
#print(_peak_feature(results))
    

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    """
    Statistical
    These include the mean, variance and the rate of zero- or mean-crossings. The
    minimum and maximum may be useful, as might the median
    
    FFT features
    use rfft() to get Discrete Fourier Transform
    
    Entropy
    Integrating acceleration
    
    Peak Features:
    Sometimes the count or location of peaks or troughs in the accelerometer signal can be
    an indicator of the type of activity being performed. This is basically what you did in
    assignment A1 to detect steps. Use the peak count over each window as a feature. Or
    try something like the average duration between peaks in a window.
    """
    
    x = []
    feature_names = []
    win = np.array(window)
    
    x_arr = win[:,0]
    y_arr = win[:,1]
    z_arr = win[:,2]
    
#     x_data = []
#     y_data = []
#     z_data = []
#     for point in window:
#         z_data.append(point[2])
#         y_data.append(point[3])
#         x_data.append(point[4])
#     x_arr = np.array(x_data)
#     y_arr = np.array(y_data)
#     z_arr = np.array(z_data)
    mag_window = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)  
    
    x.append(_compute_mean_features(win[:,0]))
    feature_names.append("x_mean")

    x.append(_compute_mean_features(win[:,1]))
    feature_names.append("y_mean")

    x.append(_compute_mean_features(win[:,2]))
    feature_names.append("z_mean")
    
    x.append(_median_feature(mag_window))
    feature_names.append("magnitude_median")
    
    x.append(_fft_feature(win[:,0]))
    feature_names.append("x_fft")
    
    x.append(_fft_feature(win[:,1]))
    feature_names.append("y_fft")
    
    x.append(_fft_feature(win[:,2]))
    feature_names.append("z_fft")
    
    x.append(_fft_feature(mag_window))
    feature_names.append("magnitude_fft")
    
    x.append(_entropy_feature(mag_window))
    feature_names.append("magnitude_entropy")
    
    x.append(_peak_feature(mag_window))
    feature_names.append("magnitude_number of peaks")



    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    feature_vector = list(x)
    return feature_names, feature_vector

# print(extract_features(results))