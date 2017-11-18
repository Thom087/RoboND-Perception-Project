#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    # TODO: Statistical Outlier Filtering
    outlierfilter = cloud.make_statistical_outlier_filter()
    outlierfilter.set_mean_k(10)                          		
    outlierfilter.set_std_dev_mul_thresh(0.2)             		
    outlierfilter_cloud = outlierfilter.filter()

    # TODO: Voxel Grid Downsampling
    vox = outlierfilter_cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005                            
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE) 
    cloud_vox_filtered = vox.filter()
    # TODO: PassThrough Filter
    passthrough_z = cloud_vox_filtered.make_passthrough_filter()
    passthrough_z.set_filter_field_name ('z')
    axis_min = 0.6 # all under axis_min [m] is erased
    axis_max = 1.2  # all over axis_max [m] is erased, ev. 1.3
    passthrough_z.set_filter_limits (axis_min, axis_max)
    cloud_filtered_passthrough = passthrough_z.filter()
    passthrough_y = cloud_filtered_passthrough.make_passthrough_filter()
    passthrough_y.set_filter_field_name ('y')
    axis_min = -0.46 # all under axis_min [m] is erased
    axis_max = 0.46  # all over axis_max [m] is erased
    passthrough_y.set_filter_limits (axis_min, axis_max)
    cloud_filtered = passthrough_y.filter()
    
    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.015 # [m] 0.01 max dist of point to be considered fitting the model
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    # TODO: Extract inliers and outliers
    # how close a point must be to the model in order to be considered as an inlier
    # Inliner
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    # Outliner
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    # Construct k-d tree (cloud with only spatial (raeumlich) information, colorless cloud)
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.025) # [m]     0.05 to 0.006 is possible!   
    ec.set_MinClusterSize(30)   #20-50
    ec.set_MaxClusterSize(2500) #2500-3000 
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0], 
                white_cloud[indice][1], 
                white_cloud[indice][2], 
                rgb_to_float(cluster_color[j])])
    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)            
    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        # Convert the cluster from pcl to ROS using helper function
        ros_cluster_cloud = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        chists = compute_color_histograms(ros_cluster_cloud, using_hsv=True) 
        normals = get_normals(ros_cluster_cloud)
        nhists = compute_normal_histograms(normals)
        # Compute the associated feature vector
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster_cloud
        detected_objects.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    object_list_param = []
    dropbox_param = []
    found_object_list = []

    object_name = String() 
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    # Get object / dropbox list
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')    

    # Get the test_scene
    test_scene_num = Int32()
    test_scene_num.data = 2

    #TODO: Rotate PR2 in place to capture side tables for the collision map

    #Loop through the founded object list
    for found_object in object_list:
        # Get arm and place position for found object
        for obj_param in object_list_param:
            if obj_param['name'] == found_object.label:      
                for dropbox_param in dropbox_list_param:
                    if dropbox_param['group'] == obj_param['group']: 
                        arm_name.data = dropbox_param['name']
                        place_pose.position.x = dropbox_param['position'][0]
                        place_pose.position.y = dropbox_param['position'][1]
                        place_pose.position.z = dropbox_param['position'][2] 

        # Get object_name 
        object_name.data = found_object.label  

        # Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(found_object.cloud).to_array()
        centroids = np.mean(points_arr, axis=0)[:3]
        #print("centroids item 0: ",centroids.item(0))
        pick_pose.position.x =  centroids.item(0)
        pick_pose.position.y =  centroids.item(1)
        pick_pose.position.z =  centroids.item(2)

        # Create a list of dictionaries for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        found_object_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            print ("Response: ",resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    #print(found_object_list)
    #found_object_list_1 = [{'pick_pose': {'position': {'y': -0.24160617589950562, 'x': 0.5418525338172913, 'z': 0.7078671455383301}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'biscuits', 'arm_name': 'right', 'test_scene_num': 1, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': -0.018535370007157326, 'x': 0.5450712442398071, 'z': 0.6789432168006897}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'soap', 'arm_name': 'right', 'test_scene_num': 1, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.22101663053035736, 'x': 0.4450579881668091, 'z': 0.6776176691055298}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'eraser', 'arm_name': 'right', 'test_scene_num': 1, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}]
    #found_object_list_2 = [{'pick_pose': {'position': {'y': -0.24818336963653564, 'x': 0.5714129209518433, 'z': 0.7077476978302002}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'biscuits', 'arm_name': 'right', 'test_scene_num': 2, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.2805783152580261, 'x': 0.5789003372192383, 'z': 0.7256014347076416}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'snacks', 'arm_name': 'right', 'test_scene_num': 2, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.003780464408919215, 'x': 0.5602497458457947, 'z': 0.678487241268158}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'soap', 'arm_name': 'right', 'test_scene_num': 2, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.2263399064540863, 'x': 0.4447445571422577, 'z': 0.6777781844139099}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'eraser', 'arm_name': 'right', 'test_scene_num': 2, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.13079415261745453, 'x': 0.6312018632888794, 'z': 0.6810355186462402}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'glue', 'arm_name': 'left', 'test_scene_num': 2, 'place_pose': {'position': {'y': 0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}]
    #found_object_list_3 = [{'pick_pose': {'position': {'y': -0.3333655595779419, 'x': 0.4270986318588257, 'z': 0.7544612884521484}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'snacks', 'arm_name': 'right', 'test_scene_num': 3, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': -0.21887065470218658, 'x': 0.5884429216384888, 'z': 0.7070587277412415}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'biscuits', 'arm_name': 'right', 'test_scene_num': 3, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.08382683992385864, 'x': 0.49219590425491333, 'z': 0.7281785607337952}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'book', 'arm_name': 'left', 'test_scene_num': 3, 'place_pose': {'position': {'y': 0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.004417045973241329, 'x': 0.6795628070831299, 'z': 0.6785048246383667}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'soap', 'arm_name': 'right', 'test_scene_num': 3, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.28268370032310486, 'x': 0.6086574792861938, 'z': 0.6488601565361023}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'eraser', 'arm_name': 'left', 'test_scene_num': 3, 'place_pose': {'position': {'y': 0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': -0.043442465364933014, 'x': 0.45357197523117065, 'z': 0.6774208545684814}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'soap2', 'arm_name': 'right', 'test_scene_num': 3, 'place_pose': {'position': {'y': -0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.2147252857685089, 'x': 0.4396413564682007, 'z': 0.6876749992370605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'sticky_notes', 'arm_name': 'left', 'test_scene_num': 3, 'place_pose': {'position': {'y': 0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}, {'pick_pose': {'position': {'y': 0.1389564722776413, 'x': 0.6141588687896729, 'z': 0.6881303191184998}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}, 'object_name': 'glue', 'arm_name': 'left', 'test_scene_num': 3, 'place_pose': {'position': {'y': 0.71, 'x': 0, 'z': 0.605}, 'orientation': {'y': 0.0, 'x': 0.0, 'z': 0.0, 'w': 0.0}}}]
    send_to_yaml('output_3.yaml', found_object_list_3)



if __name__ == '__main__':

    # TODO: ROS node initialization
    # rospy.init_node('object_recognition', anonymous=True)
    rospy.init_node('feature_extractor', anonymous=True)
    
    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
