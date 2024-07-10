from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from .losses import class_loss

def init_RPN_model(backbone, rpn_heads, params):
    """
    Initialize Region Proposal Network (RPN).
        
    :param tf.keras.model backbone: backbone feature extractor model.
    :param tf.keras.model rpn_heads: rpn model heads for box regression variables and confidence scores.
    :param dict params: Hyperparameters.
        
    :return tf.keras.model RPN: Region Proposal Network.
    """
    
    imgs = Input(shape = (params['IMG_H'], params['IMG_W'], params['CHANNELS']), name = 'Input')

    feat_map = backbone(imgs)

    conf, reg = rpn_heads([feat_map])

    return Model(inputs = [imgs], outputs = [conf, reg], name = 'RPN')
	
def init_Faster_RCNN(RPN, extract_rpn_proposals_layer, classifier_head, params):
    """
    Initialize Faster R-CNN Model.
        
    :param tf.keras.model RPN: Region Proposal Network (RPN).
    :param tf.keras.layer extract_rpn_proposals_layer: layer for decoding RPN prediction output.
    :param tf.keras.model classifier_head: Classifier Head for the Faster R-CNN model to assign predicted class labels to ROI's.
    :param dict params: Hyperparameters.
        
    :return tf.keras.model FasterRCNN: Faster R-CNN Model.
    """
    
    imgs = Input(shape = (params['IMG_H'], params['IMG_W'], params['CHANNELS']), name = 'Input_Images')
    
    conf, reg = RPN(imgs)
    
    prop_confs, prop_yxminmax, prop_assigned_cls = extract_rpn_proposals_layer([conf, reg, None, None], training = False)
    
    prop_pred_cls = classifier_head([imgs, prop_yxminmax])
    
    FasterRCNN = Model(inputs = [imgs], 
                       outputs = [prop_confs, prop_yxminmax, prop_pred_cls], 
                       name = 'FasterRCNN')
    
    return FasterRCNN