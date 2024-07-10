import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from .utils import xywh_to_xyminmax, xyminmax_to_yxminmax, IoU

class RPN_box_deltas_output(Layer):
    
    def __init__(self, params, **kwargs):
        super(RPN_box_deltas_output, self).__init__(**kwargs)
        self.params = params

    def get_config(self):
        config = super().get_config()
        config.update({
            "params": self.params
        })
        return config
        
    def call(self, inputs):
        # reshape input
        x = tf.reshape(inputs, (-1, 
                                self.params['GRID_H'], 
                                self.params['GRID_W'], 
                                len(self.params['ANCHOR_DIMS'])*len(self.params['ASPECT_RATIOS']), 
                                4))
        return x   
    
class RPN_conf_output(Layer):
    
    def __init__(self, **kwargs):
        super(RPN_conf_output, self).__init__(**kwargs)
        
    def call(self, inputs):
        # add dimension to
        x = tf.expand_dims(inputs, axis = -1)
        return x

class RPN_heads(Layer):
    
    def __init__(self, regression_filters, confidence_filters, params, **kwargs):
        
        """
        Initialize regression and confidence head layers and model variables.
        
        :param list regression_filters: list with filter sizes for the regression head.
        :param list confidence_filters: list with filter sizes for the confidence score head.
        :param dict params: Hyperparameters.
        
        :return None
        """
        
        super(RPN_heads, self).__init__(**kwargs)
        self.regression_filters = regression_filters
        self.confidence_filters = confidence_filters
        
        # initialize layers within the regression head of the RPN
        self.regression_layers = []
        t = 0
        for i in range(len(regression_filters)):
            self.regression_layers.append(Conv2D(filters = regression_filters[i], kernel_size = 3, 
                                                 padding='same', 
                                                 kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.01),
                                                 activation = None,
                                                 name='Reg_Conv2D_'+str(i+1)))
            self.regression_layers.append(BatchNormalization(name='Reg_BN_'+str(i+1)))
            self.regression_layers.append(LeakyReLU(0.1, name='Reg_Lky_ReLU_'+str(i+1)))
            t += 1
            pass
        
        self.regression_layers.append(Conv2D(filters = 4 * (len(params['ANCHOR_DIMS'])*len(params['ASPECT_RATIOS'])), 
                                             kernel_size = 1, 
                                             padding='same',
                                             activation = None,
                                             name='Reg_Conv2D_'+str(t+1)))

        self.regression_layers.append(RPN_box_deltas_output(params = params, name = 'Regression_Output'))

        
        # initialize layers within the confidence score head of the RPN
        self.confidence_layers = []
        t = 0
        for i in range(len(confidence_filters)):
            self.confidence_layers.append(Conv2D(filters = confidence_filters[i], kernel_size = 3, 
                                                 padding='same', 
                                                 kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.01),
                                                 activation = None,
                                                 name='Conf_Conv2D_'+str(i+1)))
            self.confidence_layers.append(BatchNormalization(name='Conf_BN_'+str(i+1)))
            self.confidence_layers.append(LeakyReLU(0.1, name='Conf_Lky_ReLU_'+str(i+1)))
            t += 1
            pass
        
        self.confidence_layers.append(Conv2D(filters = len(params['ANCHOR_DIMS'])*len(params['ASPECT_RATIOS']), 
                                             kernel_size = 1, 
                                             padding='same',
                                             activation = "sigmoid",
                                             name='Conf_Conv2D_'+str(t+1)))
        
        self.confidence_layers.append(RPN_conf_output(name = 'Confidence_Output'))
        
        pass
    
    def call(self, inputs):
        
        # batch size can only be 1 because of proposal (m) dimension variability
        feature_maps = tf.cast(inputs[0], tf.float32) # (n, FEAT_H, FEAT_W, CHANNELS)
        
        # run feature map through regression head
        reg_x = feature_maps
        for layer in self.regression_layers:
            reg_x = layer(reg_x)
            pass
        
        # run feature map through confidence head
        conf_x = feature_maps
        for layer in self.confidence_layers:
            conf_x = layer(conf_x)
            pass
        
        return conf_x, reg_x

class Extract_RPN_Proposals(Layer):
    
    def __init__(self, anchor_map, conf_thres, nms_iou_thres, nms_max_output_size, background_iou_thres, params, **kwargs):
        
        """
        Convert RPN predictions to Region of Interest proposals.
        
        :param np.array anchor_map: (1, GRID_H, GRID_W, ANCHORS, 4) --> [xmid, ymid, w, h].
        :param float conf_thres: Predicted confidence threshold. Only proposals that exceed this threshold 
                                 are taken into account.
        :param float nms_iou_thres: Non-Max Suppression threshold. Proposals with higher than this threshold
                                    IoU with a higher confidence proposal will be suppressed.
        :param int nms_max_output_size: Maximum number of boxes outputted after non-max suppression.
        :param float background_iou_thres: IoU threshold for determining whether a proposed region overlaps "enough"
                                           with a ground truth object to be considered a foreground class.
        :param dict params: Hyperparameters.
        
        :return None
        """
        
        super(Extract_RPN_Proposals, self).__init__(**kwargs)
        self.anchor_map = anchor_map
        self.conf_thres = conf_thres # confidence threshold: pred anchors that exceed this threshold are included as region proposals
        self.nms_iou_thres = nms_iou_thres # Non-max IoU threshold
        self.nms_max_output_size = nms_max_output_size # maximum number of boxes outputted after non-max suppression
        self.background_iou_thres = background_iou_thres # background IoU threshold: region proposals that fall below this threshold get assigned background class
        self.params = params
        pass
        
    def call(self, inputs, training = False):
        
        # read in layer inputs
        pred_confs = tf.cast(inputs[0], tf.float32) # RPN predicted object confidences (n, GRID_H, GRID_W, ANCHORS, 1)
        pred_deltas = tf.cast(inputs[1], tf.float32) # RPN predicted object bounding box regression deltas (n, GRID_H, GRID_W, ANCHORS, 4) --> [tx, ty, tw, th]
        
        gt_objs, gt_cls = None, None
        if training:
            gt_objs = tf.cast(inputs[2], tf.float32) # (n, m, 4) --> [xmid, ymid, w, h] & m = number of gt objects.
            gt_cls = tf.cast(inputs[3], tf.float32) # (n, m)
            pass
        
        # locations of RPN predictions that exceed confidence threshold
        loc = tf.where(tf.greater(pred_confs, self.conf_thres))

        # extract those predictions from predictions and the anchor map
        confs = tf.cast(tf.gather_nd(pred_confs, loc), tf.float32) # Shape = (m,)
        deltas = tf.cast(tf.gather_nd(pred_deltas, loc[...,:-1]), tf.float32) # Shape = (m, 4)
        anchors = tf.cast(tf.gather_nd(self.anchor_map, loc[...,:-1]), tf.float32) # Shape = (m, 4)

        # convert from regression variables to xywh format
        xy = tf.add(tf.multiply(deltas[:,:2], anchors[:,2:]), anchors[:,:2]) # Shape = (m, 2)
        wh = tf.multiply(tf.math.exp(deltas[:,2:]), anchors[:,2:]) # Shape = (m, 2)
        xywh = tf.concat([xy, wh], axis = -1) # Shape = (m, 4)
        
        # convert shape to desired input for tf.image.non_max_suppression
        xyminmax = xywh_to_xyminmax(xywh) # Shape = (m, 4)
        yxminmax = xyminmax_to_yxminmax(xyminmax) # Shape = (m, 4)
        
        # non-max suppression
        nms_inds = tf.image.non_max_suppression(boxes = yxminmax,
                                                scores = confs,
                                                max_output_size = self.nms_max_output_size,
                                                iou_threshold = self.nms_iou_thres) # Shape = (nms_m, )
        
        # extract non-max suppressed bounding boxes, confidence scores, and assigned classes
        #nms_xywh = tf.gather(xywh, nms_inds) # Shape = (nms_m, 4)
        nms_yxminmax = tf.gather(yxminmax, nms_inds) # Shape = (nms_m, 4)
        nms_confs = tf.gather(confs, nms_inds) # Shape = (nms_m, )
        
        # assign underlying ground truth class that the predicted region proposal based on IoU
        nms_assigned_cls = None #tf.zeros_like(nms_confs, dtype = tf.float32) # Shape = (nms_m, )
        if training:
            # NOTE: we have to remove the batch dimensions for the ground truth classes and objects
            assigned_cls = self.assign_gt_class_label(gt_cls[0], gt_objs[0], xywh) # Shape = (m, )
            nms_assigned_cls = tf.gather(assigned_cls, nms_inds) # Shape = (nms_m, )
            pass
        
        # nms_confs: RPN predicted proposals confidence scores (nms_m, ).
        # nms_xywh: RPN predicted proposals (nms_m, 4).
        # nms_assigned_cls: RPN predicted proposals assigned class (nms_m, ).
        return nms_confs, nms_yxminmax, nms_assigned_cls
    
    def assign_gt_class_label(self, gt_cls, gt_objs, xywh):
        """
        Function to assign underlying ground truth objects to predicted region proposals.

        :param tf.Tensor gt_objs: ground truth object coordinates (m, 4) --> [xmid, ymid, w, h].
        :param tf.Tensor gt_cls: ground truth object classes (m,).
        :param tf.Tensor xywh: RPN predicted proposals (m, 4) --> [xmid, ymid, w, h].
        
        :return tf.Tensor assigned_cls: RPN predicted proposals assigned classes (m,).
        """
        
        # if there are ground truth objects to consider...
        pred = tf.not_equal(tf.shape(gt_objs)[0], 0)
        
        def true_cond():
            # create a grid of (proposals, gt-objects) IoU's
            ious = IoU(pred = xywh_to_xyminmax(xywh), 
                       true = xywh_to_xyminmax(gt_objs)) # (N, M)

            # extract best proposal - gt-object pairings
            best_iou_inds = tf.argmax(ious, axis = -1) # indices of each proposal's highest IoU with a gt object (N, ) 
            best_ious = tf.reduce_max(ious, axis = -1) # each proposal's highest IoU with a gt object (N, ) 
            
            # assigned class index for each proposal
            assigned_cls = tf.gather(gt_cls, best_iou_inds) # (N, )
            
            # change the assigned class to background class...
            assigned_cls = tf.where(tf.less(best_ious, self.background_iou_thres), # where its best IoU with any of the gt boxes does not exceed the threshold
                                    tf.cast(tf.shape(self.params['CLASSES'])[0], tf.float32),
                                    assigned_cls)
            
            #print('You have reached assigning gt label when training + gt objs exist')
            #print('Percentage of pred is background: ' + str(np.round((np.sum(assigned_cls == 2.0)/len(assigned_cls))*100.0,2)))
            
            return assigned_cls
        
        def false_cond():
            #print('You have reached assigning gt label when training + gt objs DO NOT exist')
            
            # all predicted region proposals get assigned the background class index
            assigned_cls = tf.add(tf.zeros_like(xywh[:,0], dtype=tf.float32), 
                                  tf.cast(tf.shape(self.params['CLASSES'])[0], tf.float32))
            return assigned_cls
        
        return tf.cond(pred, true_cond, false_cond)  