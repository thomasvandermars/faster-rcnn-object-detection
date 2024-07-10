import cv2 as cv
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def predict_video(filename,
                  FasterRCNN,
                  params,
                  folder = 'test/video/',
                  fps = 25,
                  box_border_color = (255, 0, 0),
                  box_border_thickness = 2,
                  label_font = cv.FONT_HERSHEY_SIMPLEX,
                  label_font_size = 1,
                  label_font_color = (255, 0, 0),
                  label_font_thickness = 2,
                  label_border_color = (255, 0, 0),
                  label_border_thickness = 2,
                  label_background_color = (255, 255, 255)):
    
    """
    Function to use the provided model to do object detection on the frames of a video.
    And then return a reconstructed video with the bounding boxes.
    
    :param str filename: Image filename.
    :param tf.keras.model FasterRCNN: Faster R-CNN model.
    :param dict params: Hyperparameters.
    :param str folder: Relative path to location of file with filename. Default is 'test/video/'.
    :param int fps: Frames per second for the outputted video. Default is 25.
    :param tuple box_border_color: Tuple with BGR color channel values [0-255] for bounding box border. Default is (255,0,0), which is equivalent to blue.
    :param int box_border_thickness: Thickness of the bounding box border. Default is 2.
    :param cv2.fontFace label_font: Label font. Default is cv2.FONT_HERSHEY_SIMPLEX.
    :param int label_font_size: Label fontsize. Default is 1.
    :param tuple label_font_color: Tuple with BGR color channel values [0-255] for label font. Default is (255,0,0), which is equivalent to blue.
    :param int label_font_thickness: Label font thickness. Default is 2.
    :param tuple label_border_color: Tuple with BGR color channel values [0-255] for label border. Default is (255,0,0), which is equivalent to blue.
    :param int label_border_thickness: Thickness of label border. Default value is 2.
    :param tuple label_background_color: Tuple with BGR color channel values [0-255] for label background. Default is (255,0,0), which is equivalent to blue.
    
    :return list frame_inference_times: List with the model inference times for each frame.
    """
    
    # make the directory for the results if it does not exist already
    if not os.path.exists(folder + 'results'):
        os.makedirs(folder + 'results')
        pass
    
    # open video that we want to do object detection on and get the number of frames
    cap = cv.VideoCapture(folder + filename)
    amount_of_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    h = int(cap.get(cv.CAP_PROP_FOURCC))
    codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
    
    # establish video dimensions
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    res, frame = cap.read()
    height, width, _ = frame.shape
    
    # open video file to write to
    fourcc = cv.VideoWriter_fourcc(*'mp4v') # codec for .mp4 format
    video = cv.VideoWriter(folder + 'results/' + filename.split('.')[-2] + '_(Faster_RCNN).mp4', fourcc, fps, (width, height))

    # list to store frames and frame inference times
    frames_list, frame_inference_times = [], []
    
    print('Creating video (.mp4)...')
    
    # iterate through frames
    for i in tqdm(range(int(amount_of_frames))):
        
        # read in frame
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        res, frame = cap.read()
        
        # use object detector model to make prediction on frame
        start = time.time() # start time frame inference
        #------------
        
        # read in image
        img_org = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_org_h, img_org_w, _ = img_org.shape
        width_rescale_factor = img_org_w / params['IMG_W']
        height_rescale_factor = img_org_h / params['IMG_H']

        # resize image to require input size
        img_input = np.array(cv.resize(img_org, (params['IMG_W'], params['IMG_H'])))

        ############## predict using RPN model and classify ROIs ###############
        
        # run frame through Faster RCNN
        rpn_roi_conf, rpn_roi_yxminmax, pred_cls = FasterRCNN([tf.expand_dims(img_input, axis = 0)])

        # convert a copy of the image back to bgr color channels
        img_org_bgr = cv.cvtColor(img_org.copy(), cv.COLOR_RGB2BGR)
        
        # if we the network predicted some Regions Of Interest (ROIs).
        if rpn_roi_yxminmax.shape[0] > 0:
            
            ###### Drawing Boxes ######
            
            # iterate through the ROIs...
            for i in range(rpn_roi_yxminmax.shape[0]):
            
                # extract and rescale bounding box coordinates
                xmin = int(rpn_roi_yxminmax[i][1] * width_rescale_factor)
                ymin = int(rpn_roi_yxminmax[i][0] * height_rescale_factor)
                xmax = int(rpn_roi_yxminmax[i][3] * width_rescale_factor)
                ymax = int(rpn_roi_yxminmax[i][2] * height_rescale_factor)
                
                # set text for label
                txt = str(params['CLASSES'][np.argmax(pred_cls[i])]) + ' [' + str(np.round(np.max(pred_cls[i])*100.0,0)) + '%]'
                
                # retrieve text dimensions
                text_size, _ = cv.getTextSize(text = txt, 
                                              fontFace = label_font, 
                                              fontScale = label_font_size, 
                                              thickness = label_font_thickness)
                text_width, text_height = text_size
                
                # draw bounding box around object
                cv.rectangle(img = img_org_bgr,
                             pt1 = (xmin, ymin),
                             pt2 = (xmax, ymax),
                             color = box_border_color, 
                             thickness = box_border_thickness)

                # draw box for background of class label text
                cv.rectangle(img = img_org_bgr,
                             pt1 = (xmin, (ymin - text_height - 20)),
                             pt2 = ((xmin + text_width + 5), ymin),
                             color = label_background_color, 
                             thickness = -1)
                
                # draw box border for class label 
                cv.rectangle(img = img_org_bgr,
                             pt1 = (xmin, (ymin - text_height - 20)),
                             pt2 = ((xmin + text_width + 5), ymin),
                             color = label_border_color, 
                             thickness = label_border_thickness)
                
                # plot text for class label
                cv.putText(img = img_org_bgr, 
                           text = txt, 
                           org = (xmin + 5, ymin - 10), 
                           fontFace = label_font, 
                           fontScale = label_font_size, 
                           color = label_font_color,
                           thickness = label_font_thickness)
                pass
            pass
        
        # write frame to video
        video.write(img_org_bgr)
        
        # append frame
        frames_list.append(img_org_bgr)
        pass
        
        #------------
        end = time.time() # end time frame inference
        
        frame_inference_times.append(end - start) # store frame inference time
    
    # release the video objects
    cap.release()
    video.release()
    #cv.destroyAllWindows()
    
    print('Done.')
    
    return frame_inference_times