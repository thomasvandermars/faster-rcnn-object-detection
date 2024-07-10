import tensorflow as tf

def IoU(pred, true):
    """
    Function to compute Intersection over Union (IoU) of predicted boxes 
    with the corresponding ground truth boxes.
    
    :param tf.Tensor pred: predicted bounding box [M, 4].
    :param tf.Tensor true: ground truth bounding box  [N, 4].
    
    :return tf.Tensor iou: Intersection over Union values.
    """
    
    # Shapes of incoming parameters
    # pred_... --> [M, 4]
    # true_... --> [N, 4]
    
    pred = tf.expand_dims(pred, axis = 1) # [M, 1, 4]
    true = tf.expand_dims(true, axis = 0) # [1, N, 4]
    
    # order of predicted proposals is [xmin, ymin, xmax, ymax]
    pred_x_min = pred[...,0] # [M, 1]
    pred_y_min = pred[...,1] # [M, 1]
    pred_x_max = pred[...,2] # [M, 1]
    pred_y_max = pred[...,3] # [M, 1]
    
    true_x_min = true[...,0] # [1, N]
    true_y_min = true[...,1] # [1, N]
    true_x_max = true[...,2] # [1, N]
    true_y_max = true[...,3] # [1, N]
    
    intersect_x_min = tf.maximum(pred_x_min, true_x_min) # [M, N]
    intersect_y_min = tf.maximum(pred_y_min, true_y_min) # [M, N]
    intersect_x_max = tf.minimum(pred_x_max, true_x_max) # [M, N]
    intersect_y_max = tf.minimum(pred_y_max, true_y_max) # [M, N]
    
    intersect_width = tf.maximum(tf.subtract(intersect_x_max, intersect_x_min), 0.0) # [M, N]
    intersect_height = tf.maximum(tf.subtract(intersect_y_max, intersect_y_min), 0.0) # [M, N]
    
    intersect_area = tf.multiply(intersect_width, intersect_height) # [M, N]
    
    pred_width = tf.maximum(tf.subtract(pred_x_max, pred_x_min), 0.0) # [M, N]
    pred_height = tf.maximum(tf.subtract(pred_y_max, pred_y_min), 0.0) # [M, N]
    true_width = tf.maximum(tf.subtract(true_x_max, true_x_min), 0.0) # [M, N]
    true_height = tf.maximum(tf.subtract(true_y_max, true_y_min), 0.0) # [M, N]
    
    pred_area = tf.multiply(pred_width, pred_height) # [M, N]
    true_area = tf.multiply(true_width, true_height) # [M, N]
    
    union_area = tf.subtract(tf.add(pred_area, true_area), intersect_area) # [M, N]
    
    # we need a small number for numeric stability for when the union area happens to be 0
    iou = tf.divide(intersect_area, tf.maximum(union_area, 1e-10))
    
    return iou
    
def xywh_to_xyminmax(xywh):
    """
    Function to convert set of bounding box coordinates from 
    [xmid, ymid, width, height] to [xmin, ymin, xmax, ymax].
    
    :param numpy.array xywh: set of bounding box coordinates [N, 4].
    
    :return numpy.array xyminmax: set of bounding box coordinates [N, 4].
    """
    
    xmin = tf.reshape(tf.subtract(xywh[:,0], tf.divide(xywh[:,2], 2.0)), (-1,1))
    ymin = tf.reshape(tf.subtract(xywh[:,1], tf.divide(xywh[:,3], 2.0)), (-1,1))
    xmax = tf.reshape(tf.add(xywh[:,0], tf.divide(xywh[:,2], 2.0)), (-1,1))
    ymax = tf.reshape(tf.add(xywh[:,1], tf.divide(xywh[:,3], 2.0)), (-1,1))       
    return tf.cast(tf.concat([xmin, ymin, xmax, ymax], axis = -1), tf.float32)

def xywh_to_yxminmax(xywh):
    """
    Function to convert set of bounding box coordinates from 
    [xmid, ymid, width, height] to [ymin, xmin, ymax, xmax] to .
    
    :param numpy.array xywh: set of bounding box coordinates [N, 4].
    
    :return numpy.array yxminmax: set of bounding box coordinates [N, 4].
    """

    xmin = tf.reshape(tf.subtract(xywh[:,0], tf.divide(xywh[:,2], 2.0)), (-1,1))
    ymin = tf.reshape(tf.subtract(xywh[:,1], tf.divide(xywh[:,3], 2.0)), (-1,1))
    xmax = tf.reshape(tf.add(xywh[:,0], tf.divide(xywh[:,2], 2.0)), (-1,1))
    ymax = tf.reshape(tf.add(xywh[:,1], tf.divide(xywh[:,3], 2.0)), (-1,1))       
    return tf.cast(tf.concat([ymin, xmin, ymax, xmax], axis = -1), tf.float32)

def xyminmax_to_yxminmax(xyminmax):
    """
    Function to convert set of bounding box coordinates from 
    [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax] to .
    
    :param numpy.array xyminmax: set of bounding box coordinates [N, 4].
    
    :return numpy.array yxminmax: set of bounding box coordinates [N, 4].
    """

    xmin = tf.reshape(xyminmax[:,0], (-1,1))
    ymin = tf.reshape(xyminmax[:,1], (-1,1))
    xmax = tf.reshape(xyminmax[:,2], (-1,1))
    ymax = tf.reshape(xyminmax[:,3], (-1,1))       
    return tf.cast(tf.concat([ymin, xmin, ymax, xmax], axis = -1), tf.float32)
    pass