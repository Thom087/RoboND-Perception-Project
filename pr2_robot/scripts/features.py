import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=False):
    # Compute histograms for the clusters
    point_colors_list = []
    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)
    
    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []
    
    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    # Compute histograms
    channel_1_hist = np.histogram(channel_1_vals, bins=32, range=(0,256))
    channel_2_hist = np.histogram(channel_2_vals, bins=32, range=(0,256))
    channel_3_hist = np.histogram(channel_3_vals, bins=32, range=(0,256))

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((channel_1_hist[0], channel_2_hist[0], channel_2_hist[0])).astype(np.float64)
    normed_features = hist_features / np.sum(hist_features)

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    #normed_features = np.random.random(96) 
    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []
    # max_x = 0
    # max_y = 0
    # max_z = 0
    # min_x = 0
    # min_y = 0
    # min_z = 0


    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])
        # Compute histograms of normal values (just like with color)
        # if max_x < max(norm_x_vals):
        #     max_x = max(norm_x_vals)
        # if min_x > min(norm_x_vals):
        #     min_x = min(norm_x_vals) 
        # if max_y < max(norm_y_vals):
        #     max_y = max(norm_y_vals)    
        # if min_y > min(norm_y_vals):
        #     min_y = min(norm_y_vals) 
        # if max_z < max(norm_z_vals): 
        #     max_z = max(norm_z_vals) 
        # if min_z > min(norm_z_vals):
        #     min_z = min(norm_z_vals)   
        
        norm_x_hist = np.histogram(norm_x_vals, bins=32, range=(0, 1)) 
        norm_y_hist = np.histogram(norm_y_vals, bins=32, range=(0, 1))
        norm_z_hist = np.histogram(norm_z_vals, bins=32, range=(0, 1))
    	
        # Concatenate and normalize the histograms
        hist_features = np.concatenate((norm_x_hist[0], norm_y_hist[0], norm_z_hist[0])).astype(np.float64)
        normed_features = hist_features / np.sum(hist_features)

        # Generate random features for demo mode.  
        # Replace normed_features with your feature vector
        #normed_features = np.random.random(96)
    # print("This is norm_x_vals max: ", max_x, ", min: ", min_x)
    # print("This is norm_y_vals: ", max_x, ", min: ", min_x)
    # print("This is norm_z_vals: ", max_x, ", min: ", min_x)
    return normed_features
