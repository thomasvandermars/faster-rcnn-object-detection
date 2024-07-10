import os
import numpy as np
import cv2 as cv
import tensorflow as tf
import xml.etree.ElementTree as ET
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from sklearn.utils.extmath import softmax
from matplotlib.patches import Rectangle
from .pipeline import preprocess, preprocess_with_augmentation

def draw_box(xmin, 
             ymin, 
             width, 
             height, 
             color, 
             alpha, 
             borderwidth, 
             borderstyle = '-', 
             fill = False):
    """
    Function to draw a bounding box on top of an image.
    
    :param float/int xmin: Top left x coordinate of bounding box.
    :param float/int ymin: Top left y coordinate of bounding box.
    :param float/int width: Width of bounding box.
    :param float/int height: Height of bounding box.
    :param str color: Name of color to be used for the bounding box.
    :param str alpha: transparancy level of bounding box fill.
    :param str borderwidth: Width of the bounding box border.
    :param str borderstyle: Bounding box border style. Default is '-'.
    :param boolean fill: True is we want the bounding box filled in with color.
    
    :return: None
    """
	# assertions
    assert(color in list(mcolors.CSS4_COLORS.keys()))
	
    # depending on if we want the bounding box filled in...
    if fill:
        plt.gca().add_patch(Rectangle((xmin, ymin), 
                                       width, height, 
                                       linewidth = borderwidth,
                                       linestyle = borderstyle,
                                       color = color, 
                                       alpha = alpha))
    else:
        plt.gca().add_patch(Rectangle((xmin, ymin), 
                                       width, height, 
                                       linewidth = borderwidth,
                                       linestyle = borderstyle,
                                       facecolor = 'none',                                       
                                       edgecolor = color))
    pass
    
def draw_midpoint(xmid, ymid, color):
    """
    Function to draw a bounding box midpoint coordinate on top of an image.
    
    :param float/int xmid: Middle x coordinate of bounding box.
    :param float/int ymid: Middle y coordinate of bounding box.
    :param str color: Name of color to be used for the midpoint coordinate.
    
    :return: None
    """
	# assertions
    assert(color in list(mcolors.CSS4_COLORS.keys()))
	
    plt.scatter(xmid, ymid, c = color)
    pass

def draw_box_label(xmin, 
                   ymin, 
                   label, 
                   fontsize = 10.0, 
                   color = 'blue', 
                   backgroundcolor = 'none', 
                   bordercolor = 'none', 
                   borderwidth = 2, 
                   borderstyle = '-',
                   padding = 5.0):
    """
    Function to draw a bounding box class label on top of an image.
    
    :param float/int xmin: Top left x coordinate of bounding box.
    :param float/int ymin: Top left y coordinate of bounding box.
    :param str label: Name of class label.
    :param float fontsize: Fontsize label. Default is 10.0.
    :param str color: Name of color to be used for the class label. Default is 'blue'.
    :param str backgroundcolor: Name of color to be used for the background of the class label. Default is 'none'.
    :param str bordercolor: Name of color to be used for the border of the class label. Default is 'none'.
    :param int borderwidth: Borderwidth of the class label. Default is 2.
    :param str borderstyle: Bounding box border style. Default is '-'.
    :param float padding: Amount of padding between label string and border. Default is 5.0.
	
    :return: None
    """
	# assertions
    assert(color in list(mcolors.CSS4_COLORS.keys()))
    
    plt.text(x = xmin + 5, 
             y = ymin - 5, 
             s = label, 
             color = color, 
             size = fontsize,
             verticalalignment = 'bottom',
             bbox = dict(facecolor = backgroundcolor, 
                         edgecolor = bordercolor, 
                         pad = padding, 
                         linewidth = borderwidth, 
                         linestyle = borderstyle))   
    pass

def show_xml_annotation(xml_file, 
                        boxcolor = 'blue',
                        boxborderwidth = 1,
                        boxborderstyle = '-',                        
                        fontsize = 10.0, 
                        labelcolor = 'blue', 
                        labelbackgroundcolor = 'none', 
                        labelbordercolor = 'none',
                        labelborderwidth = 2,
                        labelborderstyle = '-',
                        labelpadding = 5.0,
                        alpha = 0.8,
                        box_filled = False):
    """
    Function to show image and the corresponding bounding boxes described in the 
	.xml annotation file.
    
    :param str xml_file: Path to xml filename of the annotation file to be shown.
    :param str boxcolor: Bounding box color. The name has to be in the list of Matplotlib's named colors.
    :param int boxborderwidth: Bounding box borderwidth. Default is 1.
    :param str boxborderstyle: Bounding box border style. Default is '-'.
    :param float fontsize: Fontsize label. Default is 10.0.
    :param str labelcolor: Class label color. Default is 'blue'.
    :param str labelbackgroundcolor: Backgroundcolor of the class label. Default is 'none'. 
    :param str labelbordercolor: Bordercolor of the class label. Default is 'none'.
    :param str labelborderwidth: Borderwidth of the class label. Default is 2.
    :param str labelborderstyle: Class label border style. Default is '-'.
    :param str labelpadding: Padding between string label and borders. Default is 5.0. 
    :param float alpha: Transparancy level. 1.0 is fully transparant. 0.0 is non-transparant. Default is 0.8.
    :param boolean box_filled: True if we want to fill in bounding box with boxcolor.
    
    :return: None
    """
    
	# make some assertions about the parameters
    assert(boxcolor in list(mcolors.CSS4_COLORS.keys()))
    if labelcolor != 'none' and labelcolor != None:
        assert(labelcolor in list(mcolors.CSS4_COLORS.keys()))
    if labelbackgroundcolor != 'none' and labelbackgroundcolor != None:
        assert(labelbackgroundcolor in list(mcolors.CSS4_COLORS.keys()))
    if labelbordercolor != 'none' and labelbordercolor != None:
        assert(labelbordercolor in list(mcolors.CSS4_COLORS.keys()))
    
    # extract XML file root
    tree = ET.parse(os.getcwd() + '/datasets/' + xml_file)
    root = tree.getroot()

    # read in image
    filename = root.find('filename').text
    img = cv.imread(os.getcwd() + '/datasets/' + xml_file.split('/')[0] + '/images/' + filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    
    # iterate through objects in annotation files
    for i in root.findall('object'):

        xmin = float(i.find('bndbox').find('xmin').text)
        ymin = float(i.find('bndbox').find('ymin').text)
        xmax = float(i.find('bndbox').find('xmax').text)
        ymax = float(i.find('bndbox').find('ymax').text)

        # draw bounding box
        draw_box(xmin = xmin, 
                 ymin = ymin, 
                 width = xmax-xmin, 
                 height = ymax-ymin, 
                 color = boxcolor,
                 borderwidth = boxborderwidth,
                 borderstyle = boxborderstyle,
                 alpha = alpha,
                 fill = box_filled)
        
        # draw object label
        draw_box_label(xmin = xmin, 
                       ymin = ymin, 
                       label = i.find('name').text,
                       fontsize = fontsize,
                       color = labelcolor, 
                       backgroundcolor = labelbackgroundcolor,
                       bordercolor = labelbordercolor,
                       borderwidth = labelborderwidth,
                       borderstyle = labelborderstyle,
                       padding = labelpadding)
        pass
    pass

def show_preprocessed_img(ind, 
                          X, 
                          Y, 
                          params, 
                          augment_params = None,
                          fontsize = 10, 
                          boxborderwidth = 2,
                          boxborderstyle = '-',
                          box_filled = False,
                          labelcolor = 'blue', 
                          labelbackgroundcolor = 'none',
                          labelborderwidth = 2,
                          labelborderstyle = '-',
                          labelpadding = 5.0,
                          alpha = 0.3):
    """
    Function to show a preprocessed (with augmentation) image (indicated by index) from the dataset.
    
    :param int ind: image index within the provided dataset.
    :param list X: List that holds the pathnames to the images (.jpg).
    :param list Y: List that holds the pathnames to the annotations (.xml).
    :param dict params: Dictionary that holds the hyperparameters.
    :param dict augment_params: Dictionary that holds the augmentation parameters.
    :param float fontsize: Fontsize label. Default is 10.0.
    :param int boxborderwidth: Bounding box borderwidth. Default is 2.
    :param str boxborderstyle: Bounding box border style. Default is '-'.
    :param boolean box_filled: True if we want to fill in bounding box with boxcolor. Default is False.
    :param str labelcolor: Class label color. Default is 'blue'.
    :param str labelbackgroundcolor: Backgroundcolor of the class label. Default is 'none'.
    :param str labelborderwidth: Borderwidth of the class label. Default is 2.
    :param str labelborderstyle: Class label border style. Default is '-'.
    :param str labelpadding: Padding between string label and borders. Default is 5.0. 
    :param float alpha: Transparancy level. 1.0 is fully transparant. 0.0 is non-transparant. Default is 0.8.
    
    :return None
    """
    
    # check if the index is within the dataset
    if ind >= len(Y):
        print('Please select an image index that is within the range of the training dataset: between 0 and ' + str(len(Y)-1))
    else:
        
        #  preprocess based on whether we want to augment or not...
        if augment_params == None:
            # preprocess image and label
            img, objs, cls = preprocess(image = X[ind], 
                                        label = Y[ind], 
                                        params = params)
            pass
        else:
            # preprocess image and label (with augmentation)
            img, objs, cls = preprocess_with_augmentation(image = X[ind], 
                                                          label = Y[ind], 
                                                          params = params,
                                                          augment_params = augment_params)
            pass

        # show image and bounding boxes
        for i in range(objs.shape[0]):
        
            # draw bounding box
            draw_box(xmin = objs[i][0]-objs[i][2]/2, 
                     ymin = objs[i][1]-objs[i][3]/2, 
                     width = objs[i][2], 
                     height = objs[i][3], 
                     color = params['CLASS_COLORS'][cls[i]],
                     borderwidth = boxborderwidth,
                     borderstyle = boxborderstyle,
                     alpha = alpha,
                     fill = box_filled)
            
            # draw object label
            draw_box_label(xmin = int(objs[i][0]-objs[i][2]/2), 
                           ymin = int(objs[i][1]-objs[i][3]/2), 
                           label = params['CLASSES'][cls[i]],
                           fontsize = fontsize,
                           color = labelcolor, 
                           backgroundcolor = labelbackgroundcolor,
                           bordercolor = params['CLASS_COLORS'][cls[i]],
                           borderwidth = labelborderwidth,
                           borderstyle = labelborderstyle,
                           padding = labelpadding)
            pass
    
    # show preprocessed image
    plt.imshow(img)
    pass

def show_RPN_performance(RPN,
                         extract_rpn_proposals_layer, 
                         params,
                         image,
                         label = None,
                         folder = '/test/image/',
                         pred_color = 'green', 
                         gt_color = 'red',
                         return_original_dimensions = True,
                         boxborderwidth = 2,
                         boxborderstyle = '-',
                         fontsize = 10.0,
                         labelcolor = 'blue', 
                         labelbackgroundcolor = 'none',
                         labelborderwidth = 2,
                         labelborderstyle = '-',
                         labelpadding = 5.0,
                         min_obj_surface = 0.0,
                         axisdrawn = True):

    """
    Function to evaluate RPN performance.
    
    :param tf.keras.model RPN: Pre-trained RPN model.
    :param tf.keras.layer extract_rpn_proposals_layer: layer for extracting proposals from RPN proposals.
    :param dict params: Hyperparameters.
    :param string image: File path to image.
    :param string label: File path to annotation. Default is None.
    :param str folder: Relative path to directory containing the image and/or label. Default is '/test/image/'.
    :param str pred_color: Predicted object bounding box color. Default is 'green'.
    :param str gt_color: Ground truth object bounding box color. Default is 'red'.
    :param bool return_original_dimensions: If True, we return original image dimensions. Default is True.
    :param int boxborderwidth: Bounding box borderwidth. Default is 1.
    :param str boxborderstyle: Bounding box border style. Default is '-'.
    :param float fontsize: Fontsize label. Default is 10.0.
    :param str labelcolor: Class label color. Default is 'blue'.
    :param str labelbackgroundcolor: Backgroundcolor of the class label. Default is 'none'.
    :param str labelborderwidth: Borderwidth of the class label. Default is 2.
    :param str labelborderstyle: Class label border style. Default is '-'.
    :param str labelpadding: Padding between string label and borders. Default is 5.0. 
    :param float min_obj_surface: Minimum object surface. Detected objects with bounding boxes that have surface dimensions 
                                  smaller than this minimum threshold are excluded from consideration. 
                                  Default is 0.0 (all detected bounding are included).
    :param boolean axisdrawn: True if we want to display axis values. Default is True.
    
    :return None
    """
    
    # turn off axis if specified
    if axisdrawn == False:
        plt.axis('off')
        pass
    
    # read in original image and its dimensions
    img_org = cv.imread(os.getcwd() + folder + image)
    h, w = img_org.shape[0], img_org.shape[1]
    h_rescale, w_rescale = h / params['IMG_H'], w / params['IMG_W'] # calculate the 
    
    # depending on whether we have a label annotations prepare the image (and target labels)
    if label is None:
        img = img_org
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (params['IMG_W'], params['IMG_H']))
        img_array = np.asarray(img)
        pass
    else:
        img, objs, cls = preprocess(image = image,
                                    label = label,
                                    params = params,
                                    folder = folder,
                                    min_obj_surface = min_obj_surface)
        pass

    # run the prepared image through the Region Proposal Network
    rpn_pred_conf, rpn_pred_reg = RPN(tf.expand_dims(img, axis = 0))

    # extract proposals from RPN encoded prediction
    rpn_roi_conf, rpn_roi_yxminmax, rpn_roi_assigned_cls = extract_rpn_proposals_layer([rpn_pred_conf, 
                                                                                        rpn_pred_reg, 
                                                                                        None, 
                                                                                        None], 
                                                                                        training = False)

    # showing the image 
    if return_original_dimensions: # if we are returning original dimensions...
        
        # show ground truth objects if we are provided ground truth labels
        # NOTE that we have to rescale everything back to original dimensions
        if label != None:
            # draw bounding boxes for the ground truth objects
            for i in range(objs.shape[0]):
                
                # draw bounding box
                draw_box(xmin = (objs[i,0]-objs[i,2]/2)*w_rescale, 
                         ymin = (objs[i,1]-objs[i,3]/2)*h_rescale, 
                         width = objs[i,2]*w_rescale, 
                         height = objs[i,3]*h_rescale, 
                         color = gt_color,
                         borderwidth = boxborderwidth,
                         borderstyle = '-',
                         alpha = 0.4,
                         fill = True)
                pass
            pass

        # if we are able to extract at least 1 predicted bounding box...
        if len(rpn_roi_yxminmax) > 0:

            # draw predicted bounding boxes
            for i in range(rpn_roi_yxminmax.shape[0]):

                # draw bounding box
                draw_box(xmin = rpn_roi_yxminmax[i][1]*w_rescale, 
                         ymin = rpn_roi_yxminmax[i][0]*h_rescale, 
                         width = (rpn_roi_yxminmax[i][3] - rpn_roi_yxminmax[i][1])*w_rescale, 
                         height = (rpn_roi_yxminmax[i][2] - rpn_roi_yxminmax[i][0])*h_rescale, 
                         color = pred_color,
                         borderwidth = boxborderwidth,
                         borderstyle = boxborderstyle,
                         alpha = 'none',
                         fill = False)
                
                # draw object label
                draw_box_label(xmin = int(rpn_roi_yxminmax[i][1])*w_rescale, 
                               ymin = int(rpn_roi_yxminmax[i][0])*h_rescale, 
                               label = 'Confidence: ' + str(np.round(rpn_roi_conf[i]*100.0,0)) + '%)',
                               fontsize = fontsize,
                               color = labelcolor, 
                               backgroundcolor = labelbackgroundcolor,
                               bordercolor = pred_color,
                               borderwidth = labelborderwidth,
                               borderstyle = labelborderstyle,
                               padding = labelpadding)
                
                pass
            pass
        
        # show image
        img_org = cv.cvtColor(img_org, cv.COLOR_BGR2RGB)
        plt.imshow(img_org)
        
    else: # if we are showing the preprocessed dimensions (standardized as model input)
    
        # show ground truth objects if we are provided ground truth labels
        if label != None:
            # draw bounding boxes for the ground truth objects
            for i in range(objs.shape[0]):
            
                # draw bounding box
                draw_box(xmin = objs[i,0]-objs[i,2]/2, 
                         ymin = objs[i,1]-objs[i,3]/2, 
                         width = objs[i,2], 
                         height = objs[i,3], 
                         color = gt_color,
                         borderwidth = boxborderwidth,
                         borderstyle = '-',
                         alpha = 0.4,
                         fill = True)
                pass
            pass

        # if we are able to extract at least 1 predicted bounding box...
        if len(rpn_roi_yxminmax) > 0:

            # draw predicted bounding boxes
            for i in range(rpn_roi_yxminmax.shape[0]):

                # draw bounding box
                draw_box(xmin = rpn_roi_yxminmax[i][1], 
                         ymin = rpn_roi_yxminmax[i][0], 
                         width = (rpn_roi_yxminmax[i][3] - rpn_roi_yxminmax[i][1]), 
                         height = (rpn_roi_yxminmax[i][2] - rpn_roi_yxminmax[i][0]), 
                         color = pred_color,
                         borderwidth = boxborderwidth,
                         borderstyle = boxborderstyle,
                         alpha = 'none',
                         fill = False)
                
                # draw object label
                draw_box_label(xmin = int(rpn_roi_yxminmax[i][1]), 
                               ymin = int(rpn_roi_yxminmax[i][0]), 
                               label = 'Confidence: ' + str(np.round(rpn_roi_conf[i]*100.0,0)) + '%)',
                               fontsize = fontsize,
                               color = labelcolor, 
                               backgroundcolor = labelbackgroundcolor,
                               bordercolor = pred_color,
                               borderwidth = labelborderwidth,
                               borderstyle = labelborderstyle,
                               padding = labelpadding)
                pass
            pass
            
        # show image
        plt.imshow(img)
        pass
    pass

def show_FasterRCNN_prediction(FasterRCNN, 
                               params,
                               image, 
                               label = None,
                               folder = '/test/image/',
                               pred_color = 'green', 
                               gt_color = 'red',
                               return_original_dimensions = True,
                               boxborderwidth = 2,
                               boxborderstyle = '-',
                               fontsize = 10,
                               labelcolor = 'blue', 
                               labelbackgroundcolor = 'none',
                               labelborderwidth = 2,
                               labelborderstyle = '-',
                               labelpadding = 5.0,
                               min_obj_surface = 0.0,
                               axisdrawn = True):

    """
    Function to evaluate FasterRCNN performance.
    
    :param tf.keras.model FasterRCNN: Pre-trained FasterRCNN model.
    :param dict params: Hyperparameters.
    :param string image: File path to image.
    :param string label: File path to annotation. Default is None.
    :param str folder: Relative path to directory containing the image and/or label. Default is '/test/image/'.
    :param str pred_color: Predicted object bounding box color. Default is 'green'.
    :param str gt_color: Ground truth object bounding box color. Default is 'red'.
    :param bool return_original_dimensions: If True, we return original image dimensions. Default is True.
    :param int boxborderwidth: Bounding box borderwidth. Default is 1.
    :param str boxborderstyle: Bounding box border style. Default is '-'.
    :param float fontsize: Fontsize label. Default is 10.0.
    :param str labelcolor: Class label color. Default is 'blue'.
    :param str labelbackgroundcolor: Backgroundcolor of the class label. Default is 'none'.
    :param str labelborderwidth: Borderwidth of the class label. Default is 2.
    :param str labelborderstyle: Class label border style. Default is '-'.
    :param str labelpadding: Padding between string label and borders. Default is 5.0.
    :param float min_obj_surface: Minimum object surface. Detected objects with bounding boxes that have surface dimensions 
                                  smaller than this minimum threshold are excluded from consideration. 
                                  Default is 0.0 (all detected bounding are included).
    :param boolean axisdrawn: True if we want to display axis values. Default is True.
    
    :return None
    """
    
    # turn off axis if specified
    if axisdrawn == False:
        plt.axis('off')
        pass
    
    # the real Faster R-CNN has a class label for "background"...
    cls_lbls = params['CLASSES']#.copy()
    #cls_lbls.append('background')
    
    # read in original image and its dimensions
    img_org = cv.imread(os.getcwd() + folder + image)
    h, w = img_org.shape[0], img_org.shape[1]
    h_rescale, w_rescale = h / params['IMG_H'], w / params['IMG_W'] # calculate the 
    
    # read in image (and label) depending on whether a ground truth label was given
    if label is None: # if no ground truth label (target) was given, read in and resize image...
        img = img_org
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (params['IMG_W'], params['IMG_H']))
        img_array = np.asarray(img)
        pass
    else: # if both image and target label were given preprocess using our preprocess function...
        img, objs, cls = preprocess(image = image,
                                    label = label,
                                    params = params,
                                    folder = folder,
                                    min_obj_surface = min_obj_surface)
        pass

    # run image through Faster R-CNN model
    rpn_roi_conf, rpn_roi_yxminmax, pred_cls = FasterRCNN([tf.expand_dims(img, axis = 0)])

    # If we want to return images with their original dimensions...
    if return_original_dimensions:
        
        # If we have provided ground truth bounding boxes... 
        if label != None:
            
            # draw bounding boxes for the ground truth objects
            for i in range(objs.shape[0]):
                
                # draw bounding box
                draw_box(xmin = (objs[i,0]-objs[i,2]/2)*w_rescale, 
                         ymin = (objs[i,1]-objs[i,3]/2)*h_rescale, 
                         width = objs[i,2]*w_rescale, 
                         height = objs[i,3]*h_rescale, 
                         color = gt_color,
                         borderwidth = boxborderwidth,
                         borderstyle = '-',
                         alpha = 0.4,
                         fill = True)
                pass
            pass

        # If the model predicts at least 1 bounding box...
        if len(rpn_roi_yxminmax) > 0:

            # draw predicted bounding boxes
            for i in range(rpn_roi_yxminmax.shape[0]):

                # draw bounding box                
                draw_box(xmin = rpn_roi_yxminmax[i][1]*w_rescale, 
                         ymin = rpn_roi_yxminmax[i][0]*h_rescale, 
                         width = (rpn_roi_yxminmax[i][3] - rpn_roi_yxminmax[i][1])*w_rescale, 
                         height = (rpn_roi_yxminmax[i][2] - rpn_roi_yxminmax[i][0])*h_rescale, 
                         color = pred_color,
                         borderwidth = boxborderwidth,
                         borderstyle = boxborderstyle,
                         alpha = 'none',
                         fill = False)
                
                # draw object label
                draw_box_label(xmin = int(rpn_roi_yxminmax[i][1])*w_rescale, 
                               ymin = int(rpn_roi_yxminmax[i][0])*h_rescale, 
                               label = cls_lbls[np.argmax(pred_cls[i])] + ' [' + str(np.round(np.max(pred_cls[i])*100.0,0)) + '%]', # '(conf: ' + str(np.round(rpn_roi_conf[i]*100.0,2)) + '%)'
                               fontsize = fontsize,
                               color = labelcolor, 
                               backgroundcolor = labelbackgroundcolor,
                               bordercolor = pred_color,
                               borderwidth = labelborderwidth,
                               borderstyle = labelborderstyle,
                               padding = labelpadding)
                pass
            pass
        
        # show image
        img_org = cv.cvtColor(img_org, cv.COLOR_BGR2RGB) # change color channels to RGB if necessary
        plt.imshow(img_org)
    
    # If we want to return images with their preprocessed dimensions (model input dimensions that is)...
    else:
    
        # If we have provided ground truth boxes...
        if label != None:
        
            # draw bounding boxes for the ground truth objects
            for i in range(objs.shape[0]):
                
                # draw bounding box
                draw_box(xmin = (objs[i,0]-objs[i,2]/2), 
                         ymin = (objs[i,1]-objs[i,3]/2), 
                         width = objs[i,2], 
                         height = objs[i,3], 
                         color = gt_color,
                         borderwidth = boxborderwidth,
                         borderstyle = '-',
                         alpha = 0.4,
                         fill = True)
                pass
            pass

        # if we are able to extract at least 1 predicted bounding box...
        if len(rpn_roi_yxminmax) > 0:

            # draw predicted bounding boxes
            for i in range(rpn_roi_yxminmax.shape[0]):

                # draw bounding box                
                draw_box(xmin = rpn_roi_yxminmax[i][1], 
                         ymin = rpn_roi_yxminmax[i][0], 
                         width = (rpn_roi_yxminmax[i][3] - rpn_roi_yxminmax[i][1]), 
                         height = (rpn_roi_yxminmax[i][2] - rpn_roi_yxminmax[i][0]), 
                         color = pred_color,
                         borderwidth = boxborderwidth,
                         borderstyle = boxborderstyle,
                         alpha = 'none',
                         fill = False)
                
                # draw object label
                draw_box_label(xmin = int(rpn_roi_yxminmax[i][1]), 
                               ymin = int(rpn_roi_yxminmax[i][0]), 
                               label = cls_lbls[np.argmax(pred_cls[i])] + ' [' + str(np.round(np.max(pred_cls[i])*100.0,0)) + '%]', # '(conf: ' + str(np.round(rpn_roi_conf[i]*100.0,2)) + '%)'
                               fontsize = fontsize,
                               color = labelcolor, 
                               backgroundcolor = labelbackgroundcolor,
                               bordercolor = pred_color,
                               borderwidth = labelborderwidth,
                               borderstyle = labelborderstyle,
                               padding = labelpadding)
                pass
            pass
        
        # show image
        plt.imshow(img)
        pass
    pass