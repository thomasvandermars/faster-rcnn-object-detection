import numpy as np

def get_anchor_sizes(params):
    """
    Function to find all possible anchor box width-height combinations given the 
    aspect ratios.

    :param dict params: Dictionary with hyperparameters.
    
    :return numpy.array anchor_box_sizes: Anchor Box dimensions of shape (n, 2).
                                          First column has the widths & second column the heights.
    """
    
    # repeat the widths vector (row) enough times for every aspect ratio
    w = np.repeat(np.expand_dims(params['ANCHOR_DIMS'], axis = -1).reshape(1, -1), len(params['ASPECT_RATIOS']), axis = 0)
    # repeat aspect ratio vector (column) enough times for every width
    a = np.repeat(params['ASPECT_RATIOS'], len(params['ANCHOR_DIMS'])).reshape(len(params['ASPECT_RATIOS']), -1)
    
    # multiply to find all corresponding heights
    h = np.multiply(w, a)
    
    # flatten and concatenate (widths in column 1, heights in column 2)
    anchor_box_sizes = np.concatenate((w.reshape(-1,1), h.reshape(-1,1)), axis = -1)
    
    return anchor_box_sizes

def anchor_mapping(params):
    """
    Function to map the anchor coordinates on the image and determine which anchors 
    coordinates are all inbounds and thus can be considered valid.

    :param dict params: Dictionary with hyperparameters.
    
    :return numpy.array anchor_map: Array with anchors coordinates [xmid, ymid, width, height] 
                                    of shape (GRID_H, GRID_W, ANCHORS, 4).
    :return numpy.array inbounds_map: Boolean array indicating which anchors are inbound of 
                                      shape (GRID_H, GRID_W, ANCHORS, 1).                     
    """
    
    # calculate anchor dimensions (widths, heights) & number of anchors
    anchor_dims = get_anchor_sizes(params) # (number of anchors, 2)
    anchors_n = anchor_dims.shape[0] # (1,)

    # calculate the offsets from the anchor center points to get to the top-left and
    # bottom-right coordinates
    anchor_offsets = np.zeros((anchors_n, 4)) # [ANCHORS, 4]
    anchor_offsets[:,0:2] -= 0.5 * anchor_dims  # top-left offsets
    anchor_offsets[:,2:4] += 0.5 * anchor_dims  # bottom-right offsets

    # calculate the grid dimensions
    grid_width  = params['IMG_W'] // params['GRID_W']
    grid_height = params['IMG_H'] // params['GRID_H'] 

    # create grid with x mid-point coordinates for the anchors
    cell_w_i = np.repeat([np.arange(0, params['GRID_W'], 1)], params['GRID_H'], axis = 0)
    cell_w_i = np.reshape(cell_w_i, (params['GRID_H'], params['GRID_W'])) 
    cell_w_i = cell_w_i * grid_width
    cell_w_i = cell_w_i + (grid_width / 2)
    cell_w_i = np.expand_dims(cell_w_i, axis = -1) # [GRID_H, GRID_W, 1]

    # create grid with y mid-point coordinates for the anchors
    # cell_h_i = np.reshape(np.arange(0, params['GRID_H'], 1), [params['GRID_W'], 1]) 
    # cell_h_i = np.repeat(cell_h_i, params['GRID_W'], axis = 1) 
    # cell_h_i = np.reshape(cell_h_i, (params['GRID_H'], params['GRID_W'])) 
    cell_h_i = np.repeat([np.arange(0, params['GRID_H'], 1)], params['GRID_W'], axis = 0).T 
    cell_h_i = cell_h_i * grid_height
    cell_h_i = cell_h_i + (grid_height / 2)
    cell_h_i = np.expand_dims(cell_h_i, axis = -1) # [GRID_H, GRID_W, 1]

    # expand and reshape the anchor offsets such that we relate them to the anchor center coordinates
    # all are of shape [GRID_H, GRID_W, ANCHORS]
    x_min_offsets = np.repeat(np.expand_dims(anchor_offsets[:,0], axis = 0), cell_w_i.shape[1], 0)
    x_min_offsets = np.repeat(np.expand_dims(x_min_offsets, axis = 0), cell_w_i.shape[0], 0)
    y_min_offsets = np.repeat(np.expand_dims(anchor_offsets[:,1], axis = 0), cell_h_i.shape[1], 0)
    y_min_offsets = np.repeat(np.expand_dims(y_min_offsets, axis = 0), cell_h_i.shape[0], 0)

    x_max_offsets = np.repeat(np.expand_dims(anchor_offsets[:,2], axis = 0), cell_w_i.shape[1], 0)
    x_max_offsets = np.repeat(np.expand_dims(x_max_offsets, axis = 0), cell_w_i.shape[0], 0)
    y_max_offsets = np.repeat(np.expand_dims(anchor_offsets[:,3], axis = 0), cell_h_i.shape[1], 0)
    y_max_offsets = np.repeat(np.expand_dims(y_max_offsets, axis = 0), cell_h_i.shape[0], 0)

    # calculate anchor top-left and bottom-right coordinates
    x_mins = np.expand_dims((cell_w_i + x_min_offsets), axis = -1) # [GRID_H, GRID_W, ANCHORS, 1]
    y_mins = np.expand_dims((cell_h_i + y_min_offsets), axis = -1) # [GRID_H, GRID_W, ANCHORS, 1]
    x_maxs = np.expand_dims((cell_w_i + x_max_offsets), axis = -1) # [GRID_H, GRID_W, ANCHORS, 1]
    y_maxs = np.expand_dims((cell_h_i + y_max_offsets), axis = -1) # [GRID_H, GRID_W, ANCHORS, 1]

    # convert to mid-point coordinates heights and widths
    xmids = np.expand_dims(np.repeat(cell_w_i, anchors_n, -1), axis = -1) # [GRID_H, GRID_W, ANCHORS, 1]
    ymids = np.expand_dims(np.repeat(cell_h_i, anchors_n, -1), axis = -1) # [GRID_H, GRID_W, ANCHORS, 1]
    heights = y_maxs - y_mins # [GRID_H, GRID_W, ANCHORS, 1]
    widths = x_maxs - x_mins # [GRID_H, GRID_W, ANCHORS, 1]

    # concatenate the coordinates and the dimensions
    anchor_map = np.concatenate([xmids, ymids, widths, heights], axis = -1) # [GRID_H, GRID_W, ANCHORS, 4]

    if params['ALLOW_OUT_OF_BOUNDS_ANCHORS'] == False:
        # establish a boolean mapping of valid and invalid anchor boxes
        x_mins_inbounds = (x_mins >= 0.) # [GRID_H, GRID_W, ANCHORS, 1]
        y_mins_inbounds = (y_mins >= 0.) # [GRID_H, GRID_W, ANCHORS, 1]
        x_maxs_inbounds = (x_maxs < params['IMG_W']) # [GRID_H, GRID_W, ANCHORS, 1]
        y_maxs_inbounds = (y_maxs < params['IMG_H']) # [GRID_H, GRID_W, ANCHORS, 1]

        inbounds_map = np.concatenate([x_mins_inbounds, y_mins_inbounds, x_maxs_inbounds, y_maxs_inbounds], axis = -1) # [GRID_H, GRID_W, ANCHORS, 4]
        inbounds_map = np.expand_dims(np.all(inbounds_map == True, axis = -1), axis = -1) # [GRID_H, GRID_W, ANCHORS, 1]
        pass
    else:
        inbounds_map = np.reshape(np.repeat(True, params['GRID_H'] * params['GRID_W'] * (len(params['ASPECT_RATIOS']) * len(params['ANCHOR_DIMS']))), 
                                  (params['GRID_H'], params['GRID_W'], (len(params['ASPECT_RATIOS']) * len(params['ANCHOR_DIMS'])), 1))
        pass
        
    anchor_map = np.expand_dims(anchor_map, axis = 0) # [1, GRID_H, GRID_W, ANCHORS, 4]
    inbounds_map = np.expand_dims(inbounds_map, axis = 0) # [1, GRID_H, GRID_W, ANCHORS, 1]
    
    return anchor_map, inbounds_map

def iou(box1, box2):
    """
    Function to compute Intersection over Union (IoU) between two boxes.
    
    :param tuple box1: Box 1 coordinates (xmin, ymin, xmax, ymax).
    :param tuple box2: Box 2 coordinates (xmin, ymin, xmax, ymax).
    
    :return float iou: Intersection over Union (IoU)
    """
    box1_x_min, box1_y_min, box1_x_max, box1_y_max = box1
    box2_x_min, box2_y_min, box2_x_max, box2_y_max = box2
    
    box1_width = np.maximum(box1_x_max - box1_x_min, 0.0)
    box1_height = np.maximum(box1_y_max - box1_y_min, 0.0)
    box2_width = np.maximum(box2_x_max - box2_x_min, 0.0)
    box2_height = np.maximum(box2_y_max - box2_y_min, 0.0)
    
    box1_area = box1_width * box1_height
    box2_area = box2_width * box2_height
    
    intersect_x_min = np.maximum(box1_x_min, box2_x_min)
    intersect_y_min = np.maximum(box1_y_min, box2_y_min)
    intersect_x_max = np.minimum(box1_x_max, box2_x_max)
    intersect_y_max = np.minimum(box1_y_max, box2_y_max)
    
    intersect_width = np.maximum(intersect_x_max - intersect_x_min, 0.0)
    intersect_height = np.maximum(intersect_y_max - intersect_y_min, 0.0)
    
    intersect = intersect_width * intersect_height
    
    union = box1_area + box2_area - intersect
    
    iou = intersect / union
    
    return iou

def rpn_mapping(anchor_map, inbounds_map, gt_boxes, foreground_IoU_threshold, params):
    """
    Function to extract the ground truth target for training the Region Proposal Network (RPN).
    
    :param np.array anchor_map: Array with anchors coordinates [xmid, ymid, width, height] 
                                of shape (1, GRID_H, GRID_W, ANCHORS, 4).
    :param np.array inbounds_map: Boolean array indicating which anchors are inbound of 
                                  shape (1, GRID_H, GRID_W, ANCHORS, 1). 
    :param list gt_boxes: list with the image's ground truth box coordinates [xmid, ymid, width, height]
    :param float foreground_IoU_threshold: IoU threshold for an anchor with a ground truth to be 
                                           considered a foreground anchor
    :param dict params: Dictionary with hyperparameters.
    
    :return np.array conf: RPN confidence values [1, GRID_H, GRID_W, ANCHORS, 1]
    :return np.array coor: RPN regression deltas [1, GRID_H, GRID_W, ANCHORS, 4]
    :return np.array valid_foreground_indices: Numpy array with indices of the valid 
                                               foreground anchors.
    """
    
    coor = np.zeros_like(anchor_map) # [1, GRID_H, GRID_W, ANCHORS, 4]

    # it could be that no ground truth objects are passed in (because augmentation moves objects out of bounds or background images)
    # return zero arrays for coordinates and confidence and empty foreground indices array
    if gt_boxes.shape[0] == 0:
        conf = np.zeros_like(inbounds_map).astype('float32')
        valid_foreground_indices = np.array([]).astype('int32')
        return conf, coor, valid_foreground_indices
    
    # extract top-left en bottom-right coordinates for anchors
    xy = anchor_map[...,0:2] # [1, GRID_H, GRID_W, ANCHORS, 2]
    wh = anchor_map[...,2:4] # [1, GRID_H, GRID_W, ANCHORS, 2]

    xy_min = xy - 0.5 * wh # [1, GRID_H, GRID_W, ANCHORS, 2]
    xy_max = xy + 0.5 * wh # [1, GRID_H, GRID_W, ANCHORS, 2]

    x_min = np.expand_dims(xy_min[...,0], axis = -1) # [1, GRID_H, GRID_W, ANCHORS, 1]
    y_min = np.expand_dims(xy_min[...,1], axis = -1) # [1, GRID_H, GRID_W, ANCHORS, 1]
    x_max = np.expand_dims(xy_max[...,0], axis = -1) # [1, GRID_H, GRID_W, ANCHORS, 1]
    y_max = np.expand_dims(xy_max[...,1], axis = -1) # [1, GRID_H, GRID_W, ANCHORS, 1]

    # go through the image's annotated objects to determine which anchors fall within 
    # IoU threshold and thus are considered foreground anchors
    valid_foregrnd_indices = [] # list for valid foreground anchor indices
    valid_foregrnd_masks = [] # list for valid foreground masks (object confidence)

    # iterate through the image's annotated objects
    for b in range(len(gt_boxes)):

        gt_xy = np.array(gt_boxes[b][0:2]) # ground truth boxes xy mid-point coordinates
        gt_wh = np.array(gt_boxes[b][2:4]) # ground truth boxes wh dimensiona 

        gt_xy_min = gt_xy - 0.5 * gt_wh # top-left xy coordinates
        gt_xy_max = gt_xy + 0.5 * gt_wh # bottom-right xy coordinates
        
        gt_x_min = np.expand_dims(gt_xy_min[...,0], axis = (0,1,2,3)) # [1, 1, 1, 1, 1]
        gt_y_min = np.expand_dims(gt_xy_min[...,1], axis = (0,1,2,3)) # [1, 1, 1, 1, 1]
        gt_x_max = np.expand_dims(gt_xy_max[...,0], axis = (0,1,2,3)) # [1, 1, 1, 1, 1]
        gt_y_max = np.expand_dims(gt_xy_max[...,1], axis = (0,1,2,3)) # [1, 1, 1, 1, 1]

        # find the current ground truth box's IoU with all the anchors
        IoUs = iou(box1 = (x_min, y_min, x_max, y_max), 
                   box2 = (gt_x_min, gt_y_min, gt_x_max, gt_y_max)) # [1, GRID_H, GRID_W, ANCHORS, 1]

        # mask for foreground and valid anchors (a.k.a anchors that are inbounds)
        # target confidence == IoU
        valid_foregrnd_mask = (IoUs >= foreground_IoU_threshold) * inbounds_map # [1, GRID_H, GRID_W, ANCHORS, 1]
        valid_foregrnd_masks.append(valid_foregrnd_mask)

        # extract [grid_h_i, grid_w_i, anchor_i] indices for the valid foreground anchors
        # > 0.0 used to be == True
        ind = np.array(np.where(valid_foregrnd_mask > 0.0)).T # [1, grid_h_ind, grid_w_ind, anchor_i, 1]
        valid_foregrnd_indices.append(ind)

        # populate the target label with the bounding box regression values [tx, ty, tw, th]
        for i in range(ind.shape[0]):
            coor[ind[i,0], ind[i,1], ind[i,2], ind[i,3], :2] = (np.array(gt_boxes[b])[:2] - anchor_map[ind[i,0], ind[i,1], ind[i,2], ind[i,3], :2]) / anchor_map[ind[i,0], ind[i,1], ind[i,2], ind[i,3], 2:]
            coor[ind[i,0], ind[i,1], ind[i,2], ind[i,3], 2:] = np.log((np.array(gt_boxes[b])[2:] / anchor_map[ind[i,0], ind[i,1], ind[i,2], ind[i,3], 2:]))
            pass
        pass

    # concatenate all the valid foreground indices
    valid_foreground_indices = np.concatenate(valid_foregrnd_indices, axis = 0)
    
    # reduce sum the foreground masks for each object to get a complete mask
    conf = np.add.reduce(valid_foregrnd_masks) > 0.0 # [1, GRID_H, GRID_W, ANCHORS, 1]
    
    # assert statements to ensure that we did everything correctly
    #assert(np.sum(conf) == valid_foreground_indices.shape[0])
    assert(np.all([x in valid_foreground_indices for x in np.array(np.where(conf == 1)).T]))
    
    # concatenate object confidence and regression values
    #rpn_ground_truth = np.concatenate([conf, coor], axis = -1)
    
    conf = conf.astype('float32')
    coor = coor.astype('float32')
    
    return conf, coor, valid_foreground_indices