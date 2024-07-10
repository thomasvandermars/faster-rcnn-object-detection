from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, MeanSquaredError

def rpn_conf_loss(y_true, y_pred):
    """
    Function to calculate the confidence loss between predicted and target labels.
    
    :param np.array y_true:  (n, GRID_H, GRID_W, ANCHORS, 1).
    :param np.array y_pred:  (n, GRID_H, GRID_W, ANCHORS, 1).
    
    :return float bce(y_true, y_pred): binary cross entropy loss between predicted and target label confidence scores.
    """
    
    bce = BinaryCrossentropy(from_logits = False)
    
    return bce(y_true, y_pred)

def rpn_box_loss(y_true, y_pred):
    """
    Function to calculate the regression loss (mean squared error) between predicted and target bounding coordinates.
    
    :param np.array y_true:  (n, GRID_H, GRID_W, ANCHORS, 4).
    :param np.array y_pred:  (n, GRID_H, GRID_W, ANCHORS, 4).
    
    :return float mse(y_true, y_pred): mean squared error loss between predicted and target label bouding box coordinates.
    """
       
    mse = MeanSquaredError()
    
    return mse(y_true, y_pred)

def class_loss(y_true, y_pred):
    """
    Function to calculate the confidence loss between predicted and target labels.

    :param tf.Tensor y_true:  (1, n_rois, ).
    :param tf.Tensor y_pred:  (n_rois, len(classes)+1).

    :return tf.Tensor bce(y_true, y_pred): binary cross entropy loss between predicted and target label confidence scores.
    """
        
    scc = SparseCategoricalCrossentropy(from_logits = False)

    return scc(y_true = y_true[0], y_pred = y_pred)