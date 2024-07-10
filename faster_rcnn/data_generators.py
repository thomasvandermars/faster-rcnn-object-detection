import tensorflow as tf
import numpy as np
from keras.utils import Sequence
from tqdm import tqdm
from .pipeline import preprocess, preprocess_with_augmentation
from .rpn_setup import anchor_mapping, rpn_mapping

class rpn_data_generator(Sequence):
    
    def __init__(self, images, labels, params, augment_params, foreground_IoU_threshold, batch_size, min_obj_surface = 0.0):
        """
        Initialize RPN data generator.
        
        :param list images: File paths to images.
        :param list labels: File paths to annotations. 
        :param dict params: Hyperparameters.
        :param dict augment_params: Dictionary with maximum degrees of translation, hue, contrast, 
                                    brightness, and saturation being applied.
        :param float foreground_IoU_threshold: IoU threshold for assigning an anchor to a ground truth.
        :param int batch_size: batch size.
        :param float min_obj_surface: minimum threshold for ground truth box surface for it to be considered large enough to detect by the model.
        
        :return None
        """
        
        anchor_map, inbounds_map = anchor_mapping(params)
        
        self.images = images
        self.labels = labels
        self.anchor_map, self.inbounds_map = anchor_mapping(params)
        self.batch_size = batch_size
        self.params = params
        self.augment_params = augment_params
        self.foreground_IoU_threshold = foreground_IoU_threshold
        self.min_obj_surface = min_obj_surface
    
    def __len__(self):
        
        return int(np.ceil(len(self.labels) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        # extract the batch
        batch_x = self.images[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # preprocess image and label
        # this is also where we can do some data augmentation
        X_img, Y_conf, Y_deltas = [], [], []
        
        # iterate through the batch
        for i in range(len(batch_y)):
            # if no augmentation parameters are passed in, we return images and labels directly
            if self.augment_params == None:
                
                img, objs, cls = preprocess(image = batch_x[i], 
                                            label = batch_y[i], 
                                            params = self.params,
                                            min_obj_surface = self.min_obj_surface)
            else: # if augmentation parameters are specified, we return images and labels which have been augmented
                
                img, objs, cls = preprocess_with_augmentation(image = batch_x[i], 
                                                              label = batch_y[i], 
                                                              params = self.params,
                                                              augment_params = self.augment_params,
                                                              min_obj_surface = self.min_obj_surface)
                pass
            
            rpn_gt_conf, rpn_gt_deltas, indices = rpn_mapping(anchor_map = self.anchor_map, 
                                                              inbounds_map = self.inbounds_map, 
                                                              gt_boxes = objs, 
                                                              foreground_IoU_threshold = self.foreground_IoU_threshold, 
                                                              params = self.params)
                
            X_img.append(img)
            Y_conf.append(rpn_gt_conf)
            Y_deltas.append(rpn_gt_deltas)
            pass
        
        return [np.array(X_img)], [np.concatenate(Y_conf, axis = 0), np.concatenate(Y_deltas, axis = 0)]

class classifier_data_generator(Sequence):
    
    def __init__(self, 
                 images, 
                 labels, 
                 RPN, 
                 extract_rpn_proposals_layer,
                 roi_resolution,
                 params, 
                 augment_params, 
                 batch_size, 
                 min_obj_surface = 0.0):
        """
        Initialize classifier data generator.
        
        :param list images: File paths to images.
        :param list labels: File paths to annotations. 
        :param tf.keras.model RPN: Pre-trained RPN model.
        :param tf.keras.layer extract_rpn_proposals_layer: layer for extracting proposals from RPN proposals.
        :param tuple roi_resolution: dimensions the ROI's are resized to (ROI height, ROI width)
        :param dict params: Hyperparameters.
        :param dict augment_params: Dictionary with maximum degrees of translation, hue, contrast, 
                                    brightness, and saturation being applied.
        :param int batch_size: batch size.
        :param float min_obj_surface: minimum threshold for ground truth box surface for it to be considered large enough to detect by the model.
        
        :return None
        """
        
        self.images = images
        self.labels = labels
        self.anchor_map, self.inbounds_map = anchor_mapping(params)
        self.RPN = RPN 
        self.extract_rpn_proposals_layer = extract_rpn_proposals_layer
        self.batch_size = batch_size
        self.roi_resolution = roi_resolution
        self.params = params
        self.augment_params = augment_params
        self.min_obj_surface = min_obj_surface
    
    def __len__(self):
        
        return int(np.ceil(len(self.labels) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        # extract the batch
        batch_x = self.images[idx * self.batch_size:]
        batch_y = self.labels[idx * self.batch_size:]
        
        # preprocess image and label
        # this is also where we can do some data augmentation
        x, y = [], []
        
        # iterate through the batch
        for i in range(len(batch_y)):
            # if no augmentation parameters are passed in, we return images and labels directly
            if self.augment_params == None:
                
                img, objs, cls = preprocess(image = batch_x[i], 
                                            label = batch_y[i], 
                                            params = self.params,
                                            min_obj_surface = self.min_obj_surface)
            else: # if augmentation parameters are specified, we return images and labels which have been augmented
                
                img, objs, cls = preprocess_with_augmentation(image = batch_x[i], 
                                                              label = batch_y[i], 
                                                              params = self.params,
                                                              augment_params = self.augment_params,
                                                              min_obj_surface = self.min_obj_surface)
                pass
            
            # run the preprocessed image through the RPN
            conf, reg = self.RPN(np.expand_dims(img, axis = 0))
    
            # decode the RPN output and extract ROI's
            prop_confs, prop_yxminmax, prop_assigned_cls = self.extract_rpn_proposals_layer([conf, 
                                                                                             reg, 
                                                                                             np.expand_dims(objs, axis = 0), 
                                                                                             np.expand_dims(cls, axis = 0)], 
                                                                                             training = True)
            
            # convert ROI coordinates to [0,1] range
            rel_roi_yxminmax = tf.divide(prop_yxminmax, [self.params['IMG_H'], self.params['IMG_W'], 
                                                         self.params['IMG_H'], self.params['IMG_W']]) # (m, 4)

            # extract ROI's from image and resize to desired resolution
            rois = tf.image.crop_and_resize(image = np.expand_dims(img, axis = 0), 
                                            boxes = rel_roi_yxminmax, 
                                            box_indices = tf.zeros(tf.shape(rel_roi_yxminmax)[0], dtype = tf.int32), 
                                            crop_size = [self.roi_resolution[0], self.roi_resolution[1]]) # (num_rois, res_height, res_width, depth)
            
            # iterate through proposed ROI's and their assigned class labels
            for j in range(len(prop_assigned_cls)):
                # if the class label is not background, then we add the ROI to the classification data
                if prop_assigned_cls[j] != len(self.params['CLASSES']):
                    y.append(prop_assigned_cls[j])
                    x.append(np.expand_dims(rois[j], axis = 0)) 
            
            # if we have reached the desired number of ROI's to make a batch, return the batch...
            if len(y) >= self.batch_size:
                return np.concatenate(x, axis = 0)[:self.batch_size], np.array(y)[:self.batch_size]
                pass
            pass

def preload_rois(i_max, X, Y, RPN, extract_rpn_proposals_layer, resolution, params):
    """
    Function to extract and save ROI's and associated labels in memory (to speed up training).
        
    :param list X: File paths to images.
    :param list Y: File paths to annotations. 
    :param tf.keras.model RPN: Pre-trained RPN model.
    :param tf.keras.layer extract_rpn_proposals_layer: layer for extracting proposals from RPN proposals.
    :param tuple roi_resolution: dimensions the ROI's are resized to (ROI height, ROI width)
    :param dict params: Hyperparameters.
    
    :return numpy.array roi_imgs: Region of Interests.
    :return numpy.array roi_cls: Region of Interests class labels.
    """

    roi_imgs, roi_cls = [], []

    # iterate through the randomly sampled indices...
    for i in tqdm(np.random.choice(a = range(len(X)), size = i_max, replace = False)):

        # preprocessed the corresponding images and labels
        img, objs, cls = preprocess(image = X[i], 
                                    label = Y[i], 
                                    params = params,
                                    min_obj_surface = 0.0)
        
        # run the preprocessed image through the RPN
        conf, reg = RPN(np.expand_dims(img, axis = 0))

        # extract the RPN encoded predictions
        prop_confs, prop_yxminmax, prop_assigned_cls = extract_rpn_proposals_layer([conf, 
                                                                                    reg, 
                                                                                    np.expand_dims(objs, axis = 0), 
                                                                                    np.expand_dims(cls, axis = 0)], 
                                                                                    training = True)

        # convert ROI coordinates to [0, 1] format
        rel_roi_yxminmax = tf.divide(prop_yxminmax, [params['IMG_H'], params['IMG_W'], 
                                                     params['IMG_H'], params['IMG_W']]) # (m, 4)

        # extract ROI's from image
        rois = tf.image.crop_and_resize(image = np.expand_dims(img, axis = 0), 
                                        boxes = rel_roi_yxminmax, 
                                        box_indices = tf.zeros(tf.shape(rel_roi_yxminmax)[0], dtype = tf.int32), 
                                        crop_size = [resolution[0], resolution[1]]) # (num_rois, res_height, res_width, depth)

        # iterate through proposed ROI's and their assigned class labels
        for j in range(len(prop_assigned_cls)):
            # if the class label is not background, then we add the ROI to the classification data
            if prop_assigned_cls[j] != len(params['CLASSES']):
                roi_cls.append(prop_assigned_cls[j])
                roi_imgs.append(np.expand_dims(rois[j], axis = 0)) 
                pass
            pass

    return np.concatenate(roi_imgs, axis = 0), np.array(roi_cls)