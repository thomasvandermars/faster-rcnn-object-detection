import os
import numpy as np
import random
import cv2 as cv
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def preprocess(image, label, params, folder = '/datasets/', min_obj_surface = 0.0):
    """
    Function to load and resize images. Also, the corresponding annotation file is read in 
    and the bounding box coordinates encoded.
    
    :param str image: Path to image file.
    :param str label: Path to annotation file.
    :param dict params: Dictionary with hyperparameters.
    :param str folder: folder name where image and label are located.
    :param float min_obj_surface: minimum object surface. Filters out bounding boxes that does not 
                                  have a sq surface exceeding this threshold.
    
    :return numpy.array img_array: Numpy array with preprocess image.
    :return numpy.array objects: Numpy array with encoded box coordinates. Shape (n, 4). [xmid, ymid, width, height].
    :return numpy.array classes: Numpy array with class indices.
    """
    # assertions
    assert(isinstance(params, dict))

    # assert valid parameter values
    assert(image.split('/')[-1][:-4] == label.split('/')[-1][:-4])
    
    # load in image and convert to image array
    img = cv.imread(os.getcwd() + folder + image) # read in image + '/datasets/'
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # convert to RGB
    img = cv.resize(img, (params['IMG_W'], params['IMG_H']))
    img_array = np.asarray(img) # convert to numpy array
    #img_array = img_array / 255. # normalize image
    
    # extract XML file root
    tree = ET.parse(os.getcwd() + folder + label)
    root = tree.getroot()

    # extract original image dimensions
    img_h, img_w = int(root.find('size').find('height').text), int(root.find('size').find('width').text)
    
    objects, classes = [], []
    
    # iterate through objects in annotation files
    for i in root.findall('object'):

        # if the detected object has a class label we are interested in
        if i.find('name').text in params['CLASSES']:
                
            # extract label & coordinates
            class_index = params['CLASSES'].index(i.find('name').text)
            xmin = float(i.find('bndbox').find('xmin').text)
            ymin = float(i.find('bndbox').find('ymin').text)
            xmax = float(i.find('bndbox').find('xmax').text)
            ymax = float(i.find('bndbox').find('ymax').text)
            
            if ((xmax - xmin) * (ymax - ymin)) > min_obj_surface:

                # image relative width, height, and midpoint coordinates (x,y)
                w = (xmax - xmin) / img_w # width relative to picture width [0,1]
                h = (ymax - ymin) / img_h # height relative to picture height [0,1]
                x = (xmin + xmax) / 2 / img_w # mid-point x-coordinate relative to picture width [0,1]
                y = (ymin + ymax) / 2 / img_h # mid-point y-coordinate relative to picture height [0,1]

                # grid relative midpoint coordinates
                x_box = x * params['IMG_W'] # mid-point x-coordinate relative to picture grid width [0, GRID_W]
                y_box = y * params['IMG_H'] # mid-point y-coordinate relative to picture grid height [0, GRID_H]
                w_box = w * params['IMG_W'] # bbox width relative to picture grid width [0, GRID_W]
                h_box = h * params['IMG_H'] # bbox height relative to picture grid height [0, GRID_H]
                
                objects.append([x_box, y_box, w_box, h_box])
                classes.append(class_index)
                pass
            pass
        pass
    
    return img_array, np.array(objects), np.array(classes)

# # function for the pre-built augmentation functions
# def augment_image(image, hue, contrast, brightness, saturation):
    # """
    # Function to augment image (by adjusting color, exposure, saturation, etc.).
    
    # :param numpy.array image: Numpy array with pixel values of incoming image.
    # :param float hue: Maximum degree of hue variation.
    # :param float contrast: Maximum degree of contrast variation.
    # :param float brightness: Maximum degree of brightness variation.
    # :param float saturation: Maximum degree of saturation variation.
    
    # :return numpy.array img_augmented: numpy array with pixel values of the augmented image
    # """
    
    # img_augmented = tf.image.stateless_random_hue(image = image, 
                                                  # max_delta = hue, 
                                                  # seed = (tf.random.uniform(shape=[], minval = 0, maxval = 1000, dtype = tf.int64),
                                                          # tf.random.uniform(shape=[], minval = 0, maxval = 1000, dtype = tf.int64)))
    # if contrast > 0.0:
        # img_augmented = tf.image.stateless_random_contrast(image = img_augmented, 
                                                           # lower = 1.0 - contrast, 
                                                           # upper = 1.0 + contrast, 
                                                           # seed = (tf.random.uniform(shape=[], minval = 0, maxval = 1000, dtype = tf.int64),
                                                                   # tf.random.uniform(shape=[], minval = 0, maxval = 1000, dtype = tf.int64)))
    
    # img_augmented = tf.image.stateless_random_brightness(image = img_augmented, 
                                                         # max_delta = brightness, 
                                                         # seed = (tf.random.uniform(shape=[], minval = 0, maxval = 1000, dtype = tf.int64),
                                                                 # tf.random.uniform(shape=[], minval = 0, maxval = 1000, dtype = tf.int64)))
    # if saturation > 0.0:
        # img_augmented = tf.image.stateless_random_saturation(image = img_augmented, 
                                                           # lower = 1.0 - saturation, 
                                                           # upper = 1.0 + saturation, 
                                                           # seed = (tf.random.uniform(shape=[], minval = 0, maxval = 1000, dtype = tf.int64),
                                                                   # tf.random.uniform(shape=[], minval = 0, maxval = 1000, dtype = tf.int64)))

    # return img_augmented

# def preprocess_with_augmentation(image, 
                                 # label, 
                                 # params,
                                 # augment_params):
    # """
    # Function to load and augment image (by adjusting color, exposure, saturation, etc.). The image is also randomly translated 
    # (randomly cropped) so that the objects appear in different sections (grids) of the image. These augmentations will help 
    # prevent overfitting on the training images. The annotation file is read in, the bounding box coordinates adjusted based
    # on the translation and then encoded into YOLO (v2) target format.
    
    # :param str image: Path to image file.
    # :param str label: Path to annotation file.
    # :param dict params: Dictionary with hyperparameters.
    # :param dict augment_params: Dictionary with maximum degrees of translation, hue, contrast, 
                                # brightness, and saturation being applied
    
    # :return numpy.array img_array: Numpy array with pixel values of the augmented image.
    # :return numpy.array lbl_array: Numpy array with encoded label (adjusted for augmentation).
    # :return numpy.array true_boxes: Numpy array with the ground truth coordinates for all the image.
    # """
    
    # # assert valid parameter values
    # assert(image.split('/')[-1][:-4] == label.split('/')[-1][:-4])
    # assert(isinstance(params, dict))
    # assert(isinstance(augment_params, dict))
    # assert(augment_params['translation'] >= 0.0)
    # assert(augment_params['translation'] <= 0.5)
    # assert(augment_params['hue'] >= 0.0)
    # assert(augment_params['hue'] <= 0.5)
    # assert(augment_params['contrast'] >= 0.0)
    # assert(augment_params['contrast'] <= 0.5)
    # assert(augment_params['brightness'] >= 0.0)
    # assert(augment_params['brightness'] <= 0.5)
    # assert(augment_params['saturation'] >= 0.0)
    # assert(augment_params['saturation'] <= 0.5)
    
    # # load in image and convert to image array
    # img = cv.imread(os.getcwd() + '/datasets/' + image) # read in image
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # convert to RGB
    # img_array = np.asarray(img) # convert to numpy array
    
    # # extract XML file root
    # tree = ET.parse(os.getcwd() + '/datasets/' + label)
    # root = tree.getroot()

    # # extract original image dimensions
    # img_h, img_w = int(root.find('size').find('height').text), int(root.find('size').find('width').text)
    
    # objects, classes = [], []
    
    # # if we are looking to translate the image
    # if augment_params['translation'] > 0.0:
        
        # # generate vertical and horizontal offsets
        # y_offset = np.random.randint(1, int(augment_params['translation'] * img_h)) 
        # x_offset = np.random.randint(1, int(augment_params['translation'] * img_w))
        
        # # boolean for where offsets are applied
        # left_offset, up_offset = random.choice([True, False]), random.choice([True, False])

        # # iterate through objects in annotation files
        # for i in root.findall('object'):

            # # if the detected object has a class label we are interested in
            # if i.find('name').text in params['CLASSES']:
            
                # # extract label & coordinates
                # class_index = params['CLASSES'].index(i.find('name').text)
                # xmin = float(i.find('bndbox').find('xmin').text)
                # ymin = float(i.find('bndbox').find('ymin').text)
                # xmax = float(i.find('bndbox').find('xmax').text)
                # ymax = float(i.find('bndbox').find('ymax').text)

                # if (left_offset == True) and (up_offset == True): # upperleft offset

                    # # left offset
                    # xmin_new = max(xmin - x_offset, 0)
                    # xmax_new = max(xmax - x_offset, 0)
                    # # upper offset
                    # ymin_new = max(ymin - y_offset, 0)
                    # ymax_new = max(ymax - y_offset, 0)
                    # # crop out array
                    # img_array_new = img_array[y_offset:, x_offset:, :]
                    # pass

                # elif (left_offset == True) and (up_offset == False): # bottomleft offset

                    # # left offset
                    # xmin_new = max(xmin - x_offset, 0) 
                    # xmax_new = max(xmax - x_offset, 0)
                    # # bottom offset
                    # ymin_new = min(ymin, img_h - y_offset - 1)
                    # ymax_new = min(ymax, img_h - y_offset - 1)
                    # # crop out array
                    # img_array_new = img_array[:img_h-y_offset, x_offset:, :]
                    # pass

                # elif (left_offset == False) and (up_offset == True): # upperright offset

                    # # upper offset
                    # ymin_new = max(ymin - y_offset, 0)
                    # ymax_new = max(ymax - y_offset, 0)
                    # # right offset
                    # xmin_new = min(xmin, img_w - x_offset - 1)
                    # xmax_new = min(xmax, img_w - x_offset - 1)
                    # # crop out array
                    # img_array_new = img_array[y_offset:, :img_w-x_offset, :]
                    # pass

                # elif (left_offset == False) and (up_offset == False): # bottomright offset

                    # # right offset
                    # xmin_new = min(xmin, img_w - x_offset - 1)
                    # xmax_new = min(xmax, img_w - x_offset - 1)
                    # # bottom offset
                    # ymin_new = min(ymin, img_h - y_offset - 1)
                    # ymax_new = min(ymax, img_h - y_offset - 1)
                    # # crop out array
                    # img_array_new = img_array[:img_h-y_offset, :img_w-x_offset, :]
                    # pass

                # # extract original image dimensions
                # img_h_new, img_w_new, _ = img_array_new.shape

                # # check if the bbox surfaces are not positive (object is not within the image)
                # if ((xmax_new - xmin_new) * (ymax_new - ymin_new)) > 0.0:

                    # # image relative width, height, and midpoint coordinates (x,y)
                    # w = (xmax_new - xmin_new) / img_w_new # width relative to picture width [0,1]
                    # h = (ymax_new - ymin_new) / img_h_new # height relative to picture height [0,1]
                    # x = (xmin_new + xmax_new) / 2 / img_w_new # mid-point x-coordinate relative to picture width [0,1]
                    # y = (ymin_new + ymax_new) / 2 / img_h_new # mid-point y-coordinate relative to picture height [0,1]

                    # x_box = x * params['IMG_W'] # mid-point x-coordinate relative to picture grid width [0, GRID_W]
                    # y_box = y * params['IMG_H'] # mid-point y-coordinate relative to picture grid height [0, GRID_H]
                    # w_box = w * params['IMG_W'] # bbox width relative to picture grid width [0, GRID_W]
                    # h_box = h * params['IMG_H'] # bbox height relative to picture grid height [0, GRID_H]
                    
                    # objects.append([x_box, y_box, w_box, h_box])
                    # classes.append(class_index)
                # pass
            # pass
        # # need to repeat this until we have at least one gt object still within the image
        # # or adjust loss target to be able to handle no objects --> better if we want to add backgrnd images
        # pass
    
    # else: # if no translation (translation == 0.0)

        # # iterate through objects in annotation files
        # for i in root.findall('object'):

            # # if the detected object has a class label we are interested in
            # if i.find('name').text in params['CLASSES']:
                
                # # extract label & coordinates
                # class_index = params['CLASSES'].index(i.find('name').text)
                # xmin = float(i.find('bndbox').find('xmin').text)
                # ymin = float(i.find('bndbox').find('ymin').text)
                # xmax = float(i.find('bndbox').find('xmax').text)
                # ymax = float(i.find('bndbox').find('ymax').text)

                # # image relative width, height, and midpoint coordinates (x,y)
                # w = (xmax - xmin) / img_w # width relative to picture width [0,1]
                # h = (ymax - ymin) / img_h # height relative to picture height [0,1]
                # x = (xmin + xmax) / 2 / img_w # mid-point x-coordinate relative to picture width [0,1]
                # y = (ymin + ymax) / 2 / img_h # mid-point y-coordinate relative to picture height [0,1]

                # x_box = x * params['IMG_W'] # mid-point x-coordinate relative to picture grid width [0, GRID_W]
                # y_box = y * params['IMG_H'] # mid-point y-coordinate relative to picture grid height [0, GRID_H]
                # w_box = w * params['IMG_W'] # bbox width relative to picture grid width [0, GRID_W]
                # h_box = h * params['IMG_H'] # bbox height relative to picture grid height [0, GRID_H]

                # objects.append([x_box, y_box, w_box, h_box])
                # classes.append(class_index)
                # pass
            # pass
        
        # # do not change the underlying image
        # img_array_new = img_array
        # pass
    
    # # resize the new image array to fit the target dimensions and then augment exposure, saturation and contrast
    # img_array = Image.fromarray(img_array_new, 'RGB')
    # img_array = img_array.resize((params['IMG_H'], params['IMG_W'])) # resize to target size
    # img_array = np.asarray(img_array) # convert to numpy array
    # img_array = augment_image(img_array, 
                              # augment_params['hue'], 
                              # augment_params['contrast'], 
                              # augment_params['brightness'],
                              # augment_params['saturation']).numpy() # augment image
    # #img_array = img_array / 255. # normalize image
    
    # return img_array, np.array(objects), np.array(classes)

def preprocess_with_augmentation(image, label, params, augment_params, folder = '/datasets/', min_obj_surface = 0.0):
    """
    Function to load and preprocess images (and their corresponding annotated bounding boxes).
    The images and their ground truth boxes are augmented by changing things such as random translations, 
    adjusting color, exposure, and saturation.
    
    :param str image: Path to image file.
    :param str label: Path to annotation file.
    :param dict params: Dictionary with hyperparameters.
    :param dict augment_params: Dictionary with maximum degrees of translation, hue, contrast, 
                                brightness, and saturation being applied.
    :param str folder: folder name where image and label are located.
    :param float min_obj_surface: minimum object surface.
    
    :return numpy.array image_aug: Numpy array with pixel values of the augmented image.
    :return numpy.array objects: Numpy array with encoded box coordinates. Shape (n, 4). [xmid, ymid, width, height].
    :return numpy.array classes: Numpy array with class indices.
    """
    
    # assert valid parameter values
    assert(image.split('/')[-1][:-4] == label.split('/')[-1][:-4])
    assert(isinstance(params, dict))
    assert(isinstance(augment_params, dict))
    
    # load in image and convert to image array
    img = cv.imread(os.getcwd() + folder + image) # read in image
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # convert to RGB
    img_array = np.asarray(img) # convert to numpy array
    
    # infer class list length & number of anchors
    CLASS = len(params['CLASSES'])
    
    # iterate through objects in annotation files and store them as bounding box objects
    bboxes = []
    tree = ET.parse(os.getcwd() + folder + label)
    root = tree.getroot()
    
    for i in root.findall('object'):

        # if the detected object has a class label we are interested in
        if i.find('name').text in params['CLASSES']:
            
            bboxes.append(BoundingBox(x1 = float(i.find('bndbox').find('xmin').text), 
                                      y1 = float(i.find('bndbox').find('ymin').text), 
                                      x2 = float(i.find('bndbox').find('xmax').text), 
                                      y2 = float(i.find('bndbox').find('ymax').text),
                                      label = int(params['CLASSES'].index(i.find('name').text))))
            pass
        pass
    
    # rescale image and the corresponding bounding boxes
    bbs = BoundingBoxesOnImage(bboxes, shape=img_array.shape)
    image_rescaled = ia.imresize_single_image(img_array, (params['IMG_H'], params['IMG_W']))
    bbs_rescaled = bbs.on(image_rescaled)
    
    # create an augmentation pipeline
    seq = iaa.Sequential([iaa.Fliplr(0.5),
                          iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, augment_params['blur']))),        
                          iaa.LinearContrast((1.0-augment_params['contrast'], 1.0+augment_params['contrast'])),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, augment_params['noise'] * 255), per_channel=0.5),
                          iaa.Multiply((1.0-augment_params['brightness'], 1.0+augment_params['brightness']), per_channel=0.2),
                          iaa.Affine(scale={"x": (1.0-augment_params['zoom'], 1.0+augment_params['zoom']),
                                            "y": (1.0-augment_params['zoom'], 1.0+augment_params['zoom'])},
                                     translate_percent={"x": (-1*augment_params['translate'], augment_params['translate']),
                                                        "y": (-1*augment_params['translate'], augment_params['translate'])})
                          # rotate=(-25, 25),
                          # shear=(-8, 8))
                           ],random_order=True)

    # augment the resized image en the corresponding bounding boxes
    image_aug, bbs_aug = seq(image = image_rescaled, bounding_boxes = bbs_rescaled)
        
    # clip the bounding box values if they exceed the image boundaries
    bbs_aug = bbs_aug.clip_out_of_image()
    
    objects, classes = [], []
    
    # iterate through the augmented bounding boxes and encode the YOLO label
    for i in range(len(bbs_aug.bounding_boxes)):
        
        # extract label & coordinates
        class_index = bbs_aug.bounding_boxes[i].label
        xmin = float(bbs_aug.bounding_boxes[i].x1)
        ymin = float(bbs_aug.bounding_boxes[i].y1)
        xmax = float(bbs_aug.bounding_boxes[i].x2)
        ymax = float(bbs_aug.bounding_boxes[i].y2)
        
        if ((xmax - xmin) * (ymax - ymin)) > min_obj_surface:
            # image relative width, height, and midpoint coordinates (x,y)
            w = (xmax - xmin) / params['IMG_W'] # width relative to picture width [0,1]
            h = (ymax - ymin) / params['IMG_H'] # height relative to picture height [0,1]
            x = (xmin + xmax) / 2 / params['IMG_W'] # mid-point x-coordinate relative to picture width [0,1]
            y = (ymin + ymax) / 2 / params['IMG_H'] # mid-point y-coordinate relative to picture height [0,1]
            
            # grid relative midpoint coordinates
            x_box = x * params['IMG_W'] # mid-point x-coordinate relative to picture grid width [0, GRID_W]
            y_box = y * params['IMG_H'] # mid-point y-coordinate relative to picture grid height [0, GRID_H]
            w_box = w * params['IMG_W'] # bbox width relative to picture grid width [0, GRID_W]
            h_box = h * params['IMG_H'] # bbox height relative to picture grid height [0, GRID_H]
            
            objects.append([x_box, y_box, w_box, h_box])
            classes.append(class_index)
            pass
    
    return image_aug, np.array(objects), np.array(classes)