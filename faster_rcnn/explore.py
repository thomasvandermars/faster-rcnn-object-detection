import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.cluster import KMeans

def get_bbox_dims(label_pathnames, classes):
    """
    Function to get all bounding box dimensions (widths and heights) for the given class labels in the dataset.
    
    :param list label_pathnames: list with all the xml annotation files in the dataset.
    :param list classes: list of class labels to be included, other existing class labels are filtered out.
    
    :return list bbox_widths: list of bounding box widths.
    :return list bbox_heights: list of bounding box heights.
    """
    
    # initialize the objects to hold the information
    bbox_widths, bbox_heights = [], []
    
    # iterate through sampled annotation files
    for xml_file in tqdm(label_pathnames):

        # extract XML file root
        tree = ET.parse('datasets/'+ xml_file)
        root = tree.getroot()

        # compute the width to height ratio
        width = float(root.find('size').find('width').text)
        height = float(root.find('size').find('height').text)
        
        # iterate through objects in annotation files
        for i in root.findall('object'):
            
            if i.find('name').text in classes: 
                
                xmin = float(i.find('bndbox').find('xmin').text)
                ymin = float(i.find('bndbox').find('ymin').text)
                xmax = float(i.find('bndbox').find('xmax').text)
                ymax = float(i.find('bndbox').find('ymax').text)
                
                bbox_widths.append((xmax - xmin) / width)
                bbox_heights.append((ymax - ymin) / height)
                
                pass
            pass
        pass
    
    return bbox_widths, bbox_heights

def mean_IoU_by_number_of_anchors(n, bbox_widths, bbox_heights):
    """
    Function to evaluate mean IoU between the anchors and the bounding boxes for up to n anchors.
    
    :param list n: number of anchors up to which we want explore the mean IoU between the anchors and the bounding boxes.
    :params list bbox_widths: list of bounding box widths.
    :params list bbox_heights: list of bounding box heights.
    
    :return mean_ious: list of mean IoU's between the anchors and the bounding boxes for each subset of n anchors.
    """
    
    # create a zip list, tying the corresponding bbox widths and heights
    bboxes = list(zip(bbox_heights, bbox_widths))
    
    # list to store the mean Intersection over Unions (IoUs)
    mean_ious = []
    
    # iterate over the number of anchors we want to explore
    for c_n in tqdm(range(1, n+1)):

        # find the cluster centerpoints for widths and heights
        kmeans = KMeans(n_clusters = c_n, n_init = 10)
        kmeans.fit(bboxes)
        
        # cluster centerpoints (widths, heights)
        clusters = kmeans.cluster_centers_
        # cluster indices assigned to each bbox
        assigned_cluster_index = kmeans.labels_

        iou = [] # list to temporarily store bbox IoU with its assigned cluster center 

        # iterate through the bounding boxes
        for i in range(len(bboxes)):

            # find the minimum height between the bbox and the assigned cluster
            intersect_height = min(clusters[assigned_cluster_index[i],0], bboxes[i][0])
            # find the minimum width between the bbox and the assigned cluster
            intersect_width = min(clusters[assigned_cluster_index[i],1], bboxes[i][1])
            
            # intersection area
            intersection = intersect_width * intersect_height
            # bounding box area
            box_area = bboxes[i][0] * bboxes[i][1]
            # assigned cluster area
            cluster_area = clusters[assigned_cluster_index[i],0] * clusters[assigned_cluster_index[i],1]

            # add Intersection over Union (IoU)
            iou.append(intersection / (box_area + cluster_area - intersection))
            pass

        # calculate and store mean Intersection over Union (IoU) for the given number of clusters
        mean_ious.append(np.mean(np.array(iou)))
        pass
    
    return mean_ious

def sample_width_to_height_ratios(xml_filenames, sample_size):
    """
    Function to calculate the image width-to-height ratios of a sample from the dataset.
    
    :param list xml_filenames: list with all the xml annotation files in the dataset.
    :param int sample_size: size of sample.
    
    :return list width_to_height_ratios: list with the sampled width-to-height ratios.
    """
    
    # cap the sample size at the size of the dataset
    if sample_size > len(xml_filenames):
        sample_size = len(xml_filenames)
        pass
    
    # list to store the extracted width to height ratios
    width_to_height_ratios = []
    
    # iterate through sampled annotation files
    for xml_file in tqdm(np.random.choice(a = xml_filenames, size = sample_size, replace = False)):
        
        # extract XML file root
        tree = ET.parse(os.getcwd() + '/datasets/' + xml_file)
        root = tree.getroot()

        # compute the width to height ratio
        width = float(root.find('size').find('width').text)
        height = float(root.find('size').find('height').text)
        width_to_height_ratios.append(width/height)
        pass
        
    return width_to_height_ratios
 
def sample_class_distribution(xml_filenames, sample_size, classes, min_obj_surface = 0.0):
    """
    Function to sample the class label distribution of the dataset.
    
    :param list xml_filenames: list with all the xml annotation files in the dataset.
    :param int sample_size: size of sample.
    :param list classes: object class labels.
    :param float min_obj_surface: minimum object surface.
    
    :return dict class_distribution: dictionary with the class label distribution.
    """
    
    # cap the sample size at the size of the dataset
    if sample_size > len(xml_filenames):
        sample_size = len(xml_filenames)
        pass
    
    class_distribution = {}
    obj_count = 0
    
    # iterate through sampled annotation files
    for xml_file in tqdm(np.random.choice(a = xml_filenames, size = sample_size, replace = False)):
        
        # extract XML file root
        tree = ET.parse(os.getcwd() + '/datasets/' + xml_file)
        root = tree.getroot()

        # iterate through objects in annotation files
        for i in root.findall('object'):
            
            # if the detected object has a class label we are interested in
            if i.find('name').text in classes:
            
                xmin = float(i.find('bndbox').find('xmin').text)
                ymin = float(i.find('bndbox').find('ymin').text)
                xmax = float(i.find('bndbox').find('xmax').text)
                ymax = float(i.find('bndbox').find('ymax').text)
                
                if ((xmax - xmin) * (ymax - ymin)) > min_obj_surface:
                    if i.find('name').text in class_distribution.keys():
                        class_distribution[i.find('name').text] += 1
                    else:
                        class_distribution[i.find('name').text] = 1
                        pass
                    
                    obj_count += 1 # increment the object count
                pass
            pass
    
    # make the values relative to sample size
    for cl in class_distribution.keys():
        class_distribution[cl] = class_distribution[cl]/obj_count
        pass
        
    return class_distribution

def get_classes(xml_filenames):
    """
    Function to get all unique object class labels in the dataset.
    
    :param list xml_filenames: list with all the xml annotation files in the dataset.
    
    :return list classes: list of unique class labels for the objects in the dataset.
    """
    
    classes = [] # list to hold all the unique class labels
    
    # iterate through sampled annotation files
    for xml_file in tqdm(xml_filenames):
        
        # extract XML file root
        tree = ET.parse(os.getcwd() + '/datasets/' + xml_file)
        root = tree.getroot()

        # iterate through objects in annotation files
        for i in root.findall('object'):
            
            if i.find('name').text not in classes:
                classes.append(i.find('name').text)
                pass
            pass
        pass
    
    # sort the class list for consistency
    classes.sort()
    
    return classes

def combine_datasets(datasets):
    """
    Function to combine the image filenames (.jpg & .png) and the annotation filenames (.xml) from different datasets within the data folder.
    The datasets folders all have an image subfolder holding the images and an annotation subfolder holding the .xml files.
    
    :param list datasets: List of dataset names within the data folder we would like to combine for model training & evaluation

    :return list X: List of pathnames to the image files.
    :return list Y: list of pathnames to the (.xml) annotation files. 
    """
    
    # combine the datasets while taking into account the different file formats (.jpg .png) of the images
    X, Y = [], []
    for i in range(len(datasets)):
        X += [datasets[i] + '/images/' + x for x in os.listdir(os.getcwd() + '/datasets/' + datasets[i] + '/images/')]
        Y += [datasets[i] + '/annotations/' + y for y in os.listdir(os.getcwd() + '/datasets/' + datasets[i] + '/annotations/')]
        pass
    
    # sort the file lists
    X.sort() 
    Y.sort()
    
    # make sure that the filenames match between the images (.jpg .png) and the annotations (.xml)
    #assert([x.split('/')[-1][:-4] for x in X] == [y.split('/')[-1][:-4] for y in Y])
    
    return X, Y