import tensorflow as tf
from tensorflow.keras import Model

class Classifier_Head(Model):
    
    def __init__(self, net, roi_resolution, params, **kwargs):
        
        """
        Initialize classifier head.
        
        :param tf.keras.model net: feature extractor model.
        :param tuple roi_resolution: dimensions of roi that are extracted from feature map (height, width).
        :param dict params: Hyperparameters.
        
        :return None
        """
        
        super(Classifier_Head, self).__init__(**kwargs)
        self.roi_resolution = roi_resolution
        self.net = net
        self.params = params
        pass
    
    def call(self, inputs):
        
        feature_maps = tf.cast(inputs[0], tf.float32) # (1, GRID_H, GRID_W, CHANNELS)
        roi_yxminmax = tf.cast(inputs[1], tf.float32) # (m, 4)
        
        # compute the top-left and bottom-right corners of the rois relative to the image height and width
        rel_roi_yxminmax = tf.divide(roi_yxminmax, [self.params['IMG_H'], self.params['IMG_W'], 
                                                    self.params['IMG_H'], self.params['IMG_W']]) # (m, 4)
        
        # extract the Regions of Interest (ROIs)
        rois = tf.image.crop_and_resize(image = feature_maps, 
                                        boxes = rel_roi_yxminmax, 
                                        box_indices = tf.zeros(tf.shape(rel_roi_yxminmax)[0], dtype = tf.int32), 
                                        crop_size = [self.roi_resolution[0], self.roi_resolution[1]]) # (num_rois, res_height, res_width, depth)
        
        
        x = self.net(rois)
        
        return x