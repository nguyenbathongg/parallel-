import sys
import math
import argparse
import cv2 as cv
import numpy as np
import numba as nb
from numba import jit, cuda
from numba.typed import List
import xml.etree.ElementTree as ET
from timeit import default_timer as timer


BLOCK_SIZE = (32, 32)


def load_model(file_name):
    '''
    Load Opencv's Haar Cascade pre-trained model.
    
    Parameter
    ---------
    file_name: Name of the file from which the classifier is loaded.
 
    Returns
    -------
    A tuple contains below numpy arrays:

    window_size : numpy.ndarray with shape=(2)
        Base width and height of detection window.

    stage_thresholds : numpy.ndarray with shape=(num_stages)
        num_stages is number of stage used in the classifier.
        threshold of each stage to check if we proceed to the next stage.
 
    tree_counts : numpy.ndarray with shape=(num_stages + 1) 
        `tree_counts[i]` is the number of tree/feature before stage `i` i.e. 
        index of the first tree of stage `i`. Therefore, 
        `range(tree_counts[i], tree_counts[i + 1])` will give range of trees' 
        index (in `feature_vals` array) of stage `i`.
 
    feature_vals : numpy.ndarray with shape(num_features, 3)
        num_features is the total number of feature used in the classifier.
        3 is (threshold, left_val, right_val) of each tree.
        Each feature correspond to a tree.
    
    rect_counts : numpy.ndarray with shape(num_features + 1)
        A feature consists of 2 or 3 rectangles. `rect_counts[i]` is the index 
        of the first rectangle of feature `i`. Therefore, 
        `range(rect_counts[i], rect_counts[i + 1])` give all rectangle's index 
        (in `rectangles` array) of feature `i`.
 
    rectangles : numpy.ndarray with shape(num_rectangles, 5)
        num_rectangles is the total number of rectangle of all features in the 
        classifier.
        5 is (x_topleft, y_topleft, width, height, weight) of each rectangle.
    '''
    xmlr = ET.parse(file_name).getroot()
    cascade = xmlr.find('cascade')
    stages = cascade.find('stages')
    features = cascade.find('features')
 
    window_size = np.array([int(cascade.find('width').text), 
                            int(cascade.find('height').text)])
 
    num_stages = len(stages)
    num_features = len(features)
 
    stage_thresholds = np.empty(num_stages, dtype=np.float64)
    tree_counts = np.empty(num_stages + 1, dtype=np.int16)
    feature_vals = np.empty((num_features, 3), dtype=np.float64)
 
    ft_cnt = 0
    tree_counts[0] = 0
    for stage_idx, stage in enumerate(stages):
        num_trees = stage.find('maxWeakCount').text
        stage_threshold = stage.find('stageThreshold').text
        weak_classifiers = stage.find('weakClassifiers')
        tree_counts[stage_idx + 1] = tree_counts[stage_idx] + np.int16(num_trees)
        stage_thresholds[stage_idx] = np.float64(stage_threshold)
        for tree in weak_classifiers:
            node = tree.find('internalNodes').text.split()
            leaf = tree.find('leafValues').text.split()
            feature_vals[ft_cnt][0] = np.float64(node[3]) # threshold
            feature_vals[ft_cnt][1] = np.float64(leaf[0]) # left value
            feature_vals[ft_cnt][2] = np.float64(leaf[1]) # right vale
            ft_cnt += 1
 
    rect_counts = np.empty(num_features + 1, dtype=np.int16)
 
    rect_counts[0] = 0
    for ft_idx, feature in enumerate(features):
        rect_count = len(feature.find('rects'))
        rect_counts[ft_idx + 1] = rect_counts[ft_idx] + np.int16(rect_count)
 
    rectangles = np.empty((rect_counts[-1], 5), np.int32)
 
    rect_cnt = 0
    for feature in features:
        rects = feature.find('rects')
        for rect in rects:
            rect_vals = rect.text.split()
            rectangles[rect_cnt][0] = np.int8(rect_vals[0])
            rectangles[rect_cnt][1] = np.int8(rect_vals[1])
            rectangles[rect_cnt][2] = np.int8(rect_vals[2])
            rectangles[rect_cnt][3] = np.int8(rect_vals[3])
            rectangles[rect_cnt][4] = np.int8(rect_vals[4][:-1])
            rect_cnt += 1
 
    return (window_size, stage_thresholds, tree_counts, 
            feature_vals, rect_counts, rectangles)


@jit(nb.void(nb.uint8[:, :, ::1], nb.uint8[:, ::1]), nopython=True)
def convert_rgb2gray(in_pixels, out_pixels):
    '''
    Convert color image to grayscale image.
 
    in_pixels : numpy.ndarray with shape=(h, w, 3)
                h, w is height, width of image
                3 is colors with BGR (blue, green, red) order
        Input RGB image
    
    out_pixels : numpy.ndarray with shape=(h, w)
        Output image in grayscale
    '''
    for r in range(len(in_pixels)):
        for c in range(len(in_pixels[0])):
            out_pixels[r, c] = (in_pixels[r, c, 0] * 0.114 + 
                                in_pixels[r, c, 1] * 0.587 + 
                                in_pixels[r, c, 2] * 0.299)


@cuda.jit(nb.void(nb.uint8[:, :, ::1], nb.uint8[:, ::1]))
def convert_rgb2gray_kernel(in_pixels, out_pixels):
    c, r = cuda.grid(2)
    if r < out_pixels.shape[0] and c < out_pixels.shape[1]:
        out_pixels[r, c] = (in_pixels[r, c, 0] * 0.114 + 
                            in_pixels[r, c, 1] * 0.587 + 
                            in_pixels[r, c, 2] * 0.299)


@cuda.jit(nb.void(nb.uint8[:, ::1], nb.int64[:, ::1], nb.int64[:, ::1]))
def cal_rows_cumsum(in_pixels, sat, sqsat):
    '''
    Calculate row-wise cummulative sum of in_pixels.
    '''
    r = cuda.grid(1)
    if r < in_pixels.shape[0]:
        for c in range(in_pixels.shape[1]):
            sat[r + 1, c + 1] = sat[r + 1, c] + in_pixels[r, c]
            sqsat[r + 1, c + 1] = sqsat[r + 1, c] + in_pixels[r, c] ** 2


@cuda.jit(nb.void(nb.uint8[:, ::1], nb.int64[:, ::1], nb.int64[:, ::1]))
def cal_cols_cumsum(in_pixels, sat, sqsat):
    '''
    Calculate column-wise cummulative sum of sat and sqsat.
    '''
    c = cuda.grid(1)
    if c < in_pixels.shape[1]:
        for r in range(in_pixels.shape[0]):
            sat[r + 1, c + 1] += sat[r, c + 1]
            sqsat[r + 1, c + 1] += sqsat[r, c + 1]


@jit(nb.void(nb.uint8[:, :, ::1], nb.types.ListType(nb.int64[::1]), nb.uint8[::1], nb.int64), nopython=True, cache=True)
def draw_rect(img, final_detections, color=0, thick=1):
    '''
    Draw bounding box on image.
    '''
    for rect in final_detections:
        (x, y, w, h), t = rect, thick
        img[y:y+h, x-t:x+t+1] = color
        img[y:y+h, x+w-t:x+w+t+1] = color
        img[y-t:y+t+1, x:x+w] = color
        img[y+h-t:y+h+t+1, x:x+w] = color


@cuda.jit(device=True)
def calc_sum_rect(sat, loc, rect):
    '''
    Evaluate feature.
    '''
    tlx = loc[0] + rect[0]
    tly = loc[1] + rect[1]
    brx = loc[0] + rect[0] + rect[2]
    bry = loc[1] + rect[1] + rect[3]
    return (sat[tly, tlx] + sat[bry, brx] - sat[bry, tlx] - sat[tly, brx])


@cuda.jit()
def detect_kernel(base_win_sz, stage_thresholds, tree_counts, 
                  feature_vals, rect_counts, rectangles, weights, sat, sqsat, 
                  scale, scale_idx, sld_w, sld_h, step, candidates_w, num_candidates, is_pass):
    x, y = cuda.grid(2)
    x_sat = x * step
    y_sat = y * step

    w, h = np.int32(base_win_sz[0] * scale), np.int32(base_win_sz[1] * scale)
    inv_area = 1 / (w * h)

    if x_sat < sld_w and y_sat < sld_h:
        win_sum = sat[y_sat][x_sat] + sat[y_sat+h][x_sat+w] - sat[y_sat][x_sat+w] - sat[y_sat+h][x_sat]
        win_sqsum = sqsat[y_sat][x_sat] + sqsat[y_sat+h][x_sat+w] - sqsat[y_sat][x_sat+w] - sqsat[y_sat+h][x_sat]
        variance = win_sqsum * inv_area - (win_sum * inv_area) ** 2

        # Reject low-variance intensity region, also take care of negative variance
        # value due to inaccuracy floating point operation.
        if variance < 100:
            return 

        std = math.sqrt(variance)

        num_stages = len(stage_thresholds)
        for stg_idx in range(num_stages):
            stg_sum = 0.0
            for tr_idx in range(tree_counts[stg_idx], tree_counts[stg_idx+1]): 
                # Implement stump-base decision tree (tree has one feature) for now.
                # Each feature consists of 2 or 3 rectangle.
                rect_idx = rect_counts[tr_idx]
                ft_sum = (calc_sum_rect(sat, (x_sat, y_sat), rectangles[rect_idx]) * weights[tr_idx] + 
                        calc_sum_rect(sat, (x_sat, y_sat), rectangles[rect_idx+1]) * rectangles[rect_idx+1][4])
                if rect_idx + 2 < rect_counts[tr_idx+1]:
                    ft_sum += calc_sum_rect(sat, (x_sat, y_sat), rectangles[rect_idx+2]) * rectangles[rect_idx+2][4]
                
                # Compare ft_sum/(area*std) with threshold to choose return value.
                stg_sum += (feature_vals[tr_idx][1] 
                            if ft_sum * inv_area < feature_vals[tr_idx][0] * std 
                            else feature_vals[tr_idx][2])

            if stg_sum < stage_thresholds[stg_idx]:
                return 

        is_pass[x + y * candidates_w + num_candidates[scale_idx]] = True


@jit(nopython=True, cache=True)
def compile_features(org_rects, rects, weights, rect_counts, scale):
    '''
    Scale features and weight correction.
    '''
    for ft_idx in range(0, len(rect_counts) - 1):
        sum = 0
        for i in range(rect_counts[ft_idx], rect_counts[ft_idx+1]):
            rects[i][0] = org_rects[i][0] * scale
            rects[i][1] = org_rects[i][1] * scale
            rects[i][2] = org_rects[i][2] * scale
            rects[i][3] = org_rects[i][3] * scale
            if i == rect_counts[ft_idx]:
                area0 = rects[i][2] * rects[i][3]
            else:
                sum += rects[i][2] * rects[i][3] * rects[i][4]
        weights[ft_idx] = -sum / area0


def detect_multi_scale(model, sats, out_img, scale_factor=1.1):
    win_size = model[0]
    height, width = out_img.shape[:2]
    max_scale = min(width / win_size[0], height / win_size[1])

    # Number of candidates before each scale
    num_candidates = []
    total_candidates = 0
    
    scale = 1.0
    while scale < max_scale:
        num_candidates.append(total_candidates)
        cur_win_size = (win_size * scale).astype(np.int32)
        step = np.int(max(2, scale))
        sld_h = height - cur_win_size[1]
        sld_w = width - cur_win_size[0]
        total_candidates += math.ceil(sld_w / step) * math.ceil(sld_h / step)
        scale *= scale_factor

    d_is_pass = cuda.device_array(total_candidates, dtype=np.bool)
    d_num_candidates = nb.cuda.to_device(num_candidates)

    base_win_sz, stage_thresholds, tree_counts, \
        feature_vals, rect_counts, rectangles = model
    org_rects = model[5].copy()
    weights = np.empty(len(feature_vals)) # modified weight of first rect of each feature'

    d_base_win_size = cuda.to_device(base_win_sz)
    d_stage_thresholds = cuda.to_device(stage_thresholds)
    d_tree_counts = cuda.to_device(tree_counts)
    d_feature_vals = cuda.to_device(feature_vals)
    d_rect_count = cuda.to_device(rect_counts)

    scale = 1.0
    scale_idx = 0
    while scale < max_scale:
        compile_features(org_rects, rectangles, weights, rect_counts, scale)       
        d_rectangles = cuda.to_device(rectangles)  
        d_weights = cuda.to_device(weights)     
        cur_win_size = (win_size * scale).astype(np.int32)
        step = np.int(max(2, scale))
        sld_w = width - cur_win_size[0]
        sld_h = height - cur_win_size[1]
        grid_size = (math.ceil((sld_w) / BLOCK_SIZE[0]), 
                     math.ceil((sld_h) / BLOCK_SIZE[1]))        
        detect_kernel[grid_size, BLOCK_SIZE](d_base_win_size, d_stage_thresholds, d_tree_counts,
                                             d_feature_vals, d_rect_count, d_rectangles, d_weights,
                                             sats[0], sats[1], scale, scale_idx, 
                                             sld_w, sld_h, step, math.ceil(sld_w / step), d_num_candidates, d_is_pass)
        scale *= scale_factor
        scale_idx += 1    

    cuda.synchronize()
    is_pass = d_is_pass.copy_to_host().astype(np.bool)
    return is_pass


@jit(nb.types.ListType(nb.int64[::1])(nb.int64[::1], nb.int64, nb.int64, nb.float64, nb.boolean[::1]), nopython=True)
def delete_zero(win_size, height, width, scale_factor, is_pass):
    scale = 1.0
    max_scale = min(width / win_size[0], height / win_size[1])    
    rec_list = List()
    scale_idx = 0
    idx = 0
    while scale < max_scale:
        cur_win_size = (win_size * scale).astype(np.int32)
        step = np.int(max(2, scale))
        sld_h = height - cur_win_size[1]
        sld_w = width - cur_win_size[0]
        for y in range(0, sld_h, step):
            for x in range(0, sld_w, step):
                if is_pass[idx] == True:
                    rec_list.append(np.array([x, y, cur_win_size[0], cur_win_size[1]]))
                idx += 1
        scale *= scale_factor
        scale_idx += 1
    return rec_list


@jit(nb.types.ListType(nb.int64[::1])(nb.types.ListType(nb.int64[::1]), nb.int64, nb.float64), nopython=True)
def group_rectangles(rectangles, min_neighbors=3, eps=0.2):
    '''
    Group object candidate rectangles.

    Parameters
    ----------
    rectangles: list(np.array(4))
        List of rectangles

    min_neighbors: int
        Minimum neighbors each candidate rectangle should have to retain it.

    eps: float
        Relative difference between sides of the rectangles to merge them into 
    a group.

    Return
    ------
    A list of grouped rectangles.
    '''
    # For testing purpose, no grouping is done.
    if min_neighbors == 0:
        return rectangles

    num_rects = len(rectangles)
    num_classes = 0

    groups = List()
    num_members = List()
    labels = np.empty(num_rects, dtype=np.int32)
    for i in range(num_rects):
        r1 = rectangles[i]
        new_group = True
        for j in range(i):
            r2 = rectangles[j]
            delta = eps * (min(r1[2], r2[2]) + min(r1[3], r2[3])) * 0.5
            if (abs(r1[0] - r2[0]) <= delta and 
                abs(r1[1] - r2[1]) <= delta and 
                abs(r1[0] + r1[2] - r2[0] - r2[2]) <= delta and 
                abs(r1[1] + r1[3] - r2[1] - r2[3]) <= delta):
                new_group = False
                labels[i] = labels[j]
                groups[labels[j]] += r1
                num_members[labels[j]] += 1
                break
        if new_group:
            groups.append(r1)
            num_members.append(1)
            labels[i] = num_classes
            num_classes += 1

    # Filter out groups which don't have enough rectangles
    i = 0
    while i < num_classes:
        while num_members[i] <= min_neighbors and i < num_classes:
            num_classes -= 1
            groups[i] = groups[num_classes]
            num_members[i] = num_members[num_classes]
        groups[i] //= num_members[i]
        i += 1

    # Filter out small rectangles inside large rectangles
    final_list = List()
    for i in range(num_classes):
        r1 = groups[i]
        m1 = max(3, num_members[i])
        is_good = True
        for j in range(num_classes):
            if i == j:
                continue
            r2 = groups[j]
            dx, dy = r2[2] * 0.2, r2[3] * 0.2
            if (r1[0] >= r2[0] - dx and 
                r1[1] >= r2[1] - dy and 
                r1[0] + r1[2] <= r2[0] + r2[2] + dx and 
                r1[1] + r1[3] <= r2[1] + r2[3] + dy and 
                num_members[j] > m1):
                is_good = False
                break
        if is_good:
            final_list.append(r1)

    return final_list


def run(model, in_img, out_img, 
        scale_factor=1.1, min_neighbors=3, eps=0.2, debug=True, test=False):
    '''
    Implement object detection workflow.
    '''
    height, width = in_img.shape[:2]
    print(f'Image size (width x height): {width} x {height}\n')

    grid_size = (math.ceil(width / BLOCK_SIZE[0]), 
                 math.ceil(height / BLOCK_SIZE[1]))

    start = timer()

    # Convert image to grayscale
    if test:
        gray_img = np.empty((height, width), dtype=in_img.dtype)
        convert_rgb2gray(in_img, gray_img)
        d_gray_img = cuda.to_device(gray_img)
    else:
        d_in_img = cuda.to_device(in_img)
        d_gray_img = cuda.device_array((height, width), dtype=np.uint8)
        convert_rgb2gray_kernel[grid_size, BLOCK_SIZE](d_in_img, d_gray_img)
 
    # Calculate Summed Area Table (SAT) and squared SAT
    d_sat = cuda.device_array((height + 1, width + 1), dtype=np.int64)
    d_sqsat = cuda.device_array((height + 1, width + 1), dtype=np.int64)
    cal_rows_cumsum[grid_size[1], BLOCK_SIZE[0]](d_gray_img, d_sat, d_sqsat)
    cal_cols_cumsum[grid_size[0], BLOCK_SIZE[0]](d_gray_img, d_sat, d_sqsat)

    # Detect object
    candidates = detect_multi_scale(model, (d_sat, d_sqsat), out_img, scale_factor = scale_factor)
    print('Number of candidates:', len(candidates))

    # Filter windows that didn't pass all stages
    result = delete_zero(model[0], height, width, scale_factor, candidates)
    print('Number of detections:', len(result))

    # Group candidates
    final_detections = group_rectangles(result, min_neighbors, eps)

    # Draw bounding box on output image
    draw_rect(out_img, final_detections, color=np.array([0,0,255], dtype=np.uint8), thick=2)

    end = timer()
    total_time = (end - start) * 1000
    print(f'Total time on host: {total_time:0.6f} ms\n')

    # Test
    if debug:
        gray_img = d_gray_img.copy_to_host()
        sat = d_sat.copy_to_host()
        sqsat = d_sqsat.copy_to_host()
        test_convert_rgb2gray(in_img, gray_img)
        test_calculate_sat(gray_img, sat, sqsat)        


def test_convert_rgb2gray(img, gray_img):
    '''
    Test convert_rgb2gray function
    '''
    gray_img_np = (img @ [0.114, 0.587, 0.299]).astype(np.uint8)
    gray_img_cv = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
 
    print('Convert rgb to grayscale error:');
    print(' - Jitted vs Numpy: ', 
          np.mean(np.abs(gray_img.astype(np.int16) - gray_img_np)))
    print(' - Jitted vs Opencv:', 
          np.mean(np.abs(gray_img.astype(np.int16) - gray_img_cv)))
 
 
def test_calculate_sat(img, sat, sqsat):
    '''
    Test calculate_sat function
    '''
    sat_np = np.array(img, dtype=np.int64)
    sat_np.cumsum(axis=0, out=sat_np).cumsum(axis=1, out=sat_np)
    sqsat_np = np.power(img, 2, dtype=np.int64)
    sqsat_np.cumsum(axis=0, out=sqsat_np).cumsum(axis=1, out=sqsat_np)
 
    total = np.sum(img)
    assert(total == sat[-1, -1])
    assert(total == sat_np[-1, -1])
    assert(np.array_equal(sat[1:, 1:], sat_np))
    assert(np.array_equal(sqsat[1:, 1:], sqsat_np))
    print('Calculate SAT: Pass')


def main(_argv=None):
    argv = _argv.split() if _argv else sys.argv[1:]

    # Parse arguments
    parser = argparse.ArgumentParser(description='Face detection using Cascaded-classifiers.', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help="Opencv's Haar Cascade pre-trained model")
    parser.add_argument('input', help='Input image')
    parser.add_argument('output', help='Output image')
    parser.add_argument('-s', default=1.1, type=float, help='Scale factor')
    parser.add_argument('-m', default=3, type=int, help='Min neighbors')
    parser.add_argument('-e', default=0.2, type=float, help='Epsilon')
    parser.add_argument('--test', default=False, action='store_true', help='Grayscale conversion on host')
    params = parser.parse_args(argv)
 
    #Load Haar Cascade model
    model = load_model(params.model)
 
    # Read input image
    in_img = cv.imread(params.input)

    # Allocate memory for output image
    out_img = in_img.copy()
 
    # Run object detection workflow
    scale_factor = params.s
    min_neighbors = params.m
    eps = params.e
    run(model, in_img, out_img, scale_factor, min_neighbors, eps, debug=True, test=params.test)
 
    # Write output image
    cv.imwrite(params.output, out_img)


# Execute
main()
