"""
This file implements some image preprocessing functions used during dataset
preparation, training and inferring.

Most functions are meant to be run only once or at the beginning of training
and hence are not really optimized for performance.
"""

"""
Notes:
    1. All output images have white foreground and black background
    2. All output images are normalized (except for DATASET_PREPARATION task)
"""

import numpy as np
import cv2
import time
import os
import collections

VERBOSE = False

# Set using trial-and-error
HORIZONTAL_MIN_AREA = 10000
VERTICAL_MIN_AREA = 2500

Params = collections.namedtuple("Params", ["extract_red_component",
                                           "horizontal_segmentation",
                                           "vertical_segmentation",
                                           "write_images",
                                           "save_image_type",
                                           "correct_skew",
                                           "remove_white_columns",
                                           "remove_white_rows"])

DATASET_PREPARATION = Params(False, True,  False, True,  "jpg", False, True, False)
TRAINING            = Params(False, False, False, False, None,  False, True, False)
INFERRING           = Params(False, False, False, False, None,  True,  True, True)

class ImagePreprocessing():
    """This class processes images for the OCR"""
    def __init__(self, params):
        self.params = params

    def __call__(self, image_path, save_path=None):
        params = self.params

        images = [cv2.imread(image_path)]
        name, image_type = os.path.split(image_path)[-1].split('.')
        if params.extract_red_component:
            images = [_extract_red_component(images[0])]

        if params.horizontal_segmentation:
            all_images = []
            for image in images:
                new_images = _horizontal_segmentation(image,
                                                      HORIZONTAL_MIN_AREA,
                                                      merge_contours=False,
                                                      correct_skew=params.correct_skew)
                all_images.extend(new_images)
            images = all_images

        if params.vertical_segmentation:
            all_images = []
            for image in images:
                new_images = _vertical_segmentation(image, VERTICAL_MIN_AREA, merge_contours=True, correct_skew=params.correct_skew)
                all_images.extend(new_images)
            images = all_images

        if params.remove_white_columns:
            images = [_remove_white_columns(image) for image in images]

        if params.remove_white_rows:
            images = [_remove_white_rows(image) for image in images]

        if params.write_images:
            if save_path is None:
                raise ValueError("Need to provide save_path to write images")
            save_image_type = params.save_image_type if params.save_image_type is not None else image_type
            if len(images) > 1:
                for i, image in enumerate(images):
                    cv2.imwrite(os.path.join(save_path, "{}_%02d.{}".format(name, save_image_type) % i), image)
            else:
                cv2.imwrite(os.path.join(save_path, "{}.{}".format(name, save_image_type)), images[0])

        return images

def _naive_extract_red_component(image):
    lower = (30, 30, 0)
    upper = (255, 255, 255)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, lower, upper)
    result = cv2.bitwise_not(cv2.bitwise_and(image, image, mask=mask))

    return result

def _extract_red_component(image):
    """
    @author: Maqsood
    Only extracts the red pixels. Removes everything else.
    
    Args:
        image -- numpy representation of image shaped (h,w,channels) in "BGR" format
    
    Returns:
        numpy array shaped (h,w,channels); image is in "BGR" format
    """
    if VERBOSE: 
        cv2.imshow("Original Image", cv2.resize(image, (1000, 200)))
        cv2.waitKey(0)

    r_pixls = image[:,:,2]
    g_pixls = image[:,:,1]
    b_pixls = image[:,:,0]
    
    ## mask for red pixels
    red_mask = np.logical_and(r_pixls > b_pixls+2,
                                r_pixls > g_pixls+2)

    image[np.logical_not(red_mask)] = [255,255,255]

    ## conditions to see presence of grey/black pixels...
    ## which have high red component but actually are stains
    ## all conditions are heuristic
    cond1 = np.logical_and(r_pixls > g_pixls,r_pixls > b_pixls)
    cond2 = np.logical_and(g_pixls > b_pixls,
                            (r_pixls-g_pixls) < (g_pixls-b_pixls))
    cond3 = np.logical_and(cond1,cond2)
    cond4 = np.logical_not((r_pixls-g_pixls) > (g_pixls-b_pixls))
    cond5 = np.logical_and((g_pixls-b_pixls) <= 4,cond4)

    # mask for stains
    stain_mask = np.logical_or(cond3,cond5)

    image[stain_mask] = [255,255,255]

    if VERBOSE: 
        cv2.imshow("Extracted Red Component", cv2.resize(image, (1000, 200)))
        cv2.waitKey(0)

    return image

def _preprocess_image(image, correct_skew=True):
    # Crop image
    image = _trim_image(image)

    if correct_skew:
        # Correct skew
        image = _skew_correction(image)

        # Crop image again (Skew correction may introduce white spaces)
        image = _trim_image(image)

    return image

def _remove_white_columns(image):
    image = np.array(image)         # Just in case
    binarized_image = _binarize(image, blur=True)
    mask = (binarized_image == 0).all(0)
    image = image[:,~mask]

    return image

def _remove_white_rows(image):
    image = np.array(image)         # Just in case
    binarized_image = _binarize(image, blur=True)
    mask = (binarized_image.T == 0).all(0)
    image = image[~mask,:]

    return image

def _horizontal_segmentation(image, area, merge_contours=False, correct_skew=True):
    # Helps deal with noisy pixels at the ends of the image (usually introduced by
    # scanners)
    image = image[:,1:-1,:]

    if VERBOSE:
        cv2.imshow("original image", image)
        cv2.waitKey(0)

    # Binarize
    thresh = _binarize(image)

    # Dilation
    kernel = np.ones((20,300), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    if VERBOSE:
        # Show ROI
        cv2.imshow('Dilated image', img_dilation)#cv2.resize(img_dilatiion, (1000, 200)))
        cv2.waitKey(0)

    # Find contours
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if merge_contours:
        # Merge contours with smaller area
        ctrs = _repeatedly_merge_contours(ctrs, area)

    # Sort contours
    #sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.contourArea(ctr))
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
   
    if VERBOSE:
        print("Number of contours: ", len(sorted_ctrs))

    segmented_images = []
    for i, ctr in enumerate(sorted_ctrs):
        ctr_area = cv2.contourArea(ctr)

        if ctr_area > area or len(sorted_ctrs) == 1:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)

            # Get ROI
            roi = image[y:y+h, x:x+w]

            roi = _preprocess_image(roi, correct_skew)

            if VERBOSE:
                # Show ROI
                cv2.imshow('segment no: ' + str(i), cv2.resize(roi, (1000, 200)))
                cv2.waitKey(0)

            segmented_images.append(roi)

    return segmented_images

def _vertical_segmentation(image, area, merge_contours=True, correct_skew=False):
    rotated_image = np.rot90(image)
    images = _horizontal_segmentation(rotated_image, area, merge_contours=merge_contours, correct_skew=correct_skew)

    rotated_images = [np.rot90(np.rot90(np.rot90(i))) for i in images]

    return rotated_images

def _vertical_segmentation_using_projections(image):
    vertical_sum = np.sum(image, axis=0)
    max_value = np.max(vertical_sum)

    segmented_images = []
    x = image.shape[1]
    cnt = 1
    flag = False

    for col in range(image.shape[1]-1,-1,-1):
        if col==0 and x>5:
            segmented_images.append(image[:,0:x])
            if VERBOSE:
                # Show segment
                cv2.imshow('vertical segment no:' + str(cnt), roi)
                cv2.waitKey(0)
            cnt += 1

        elif vertical_sum[col] >= max_value-10:            
            if x-col>5:
                segmented_images.append(image[:,col:x])
                if VERBOSE:
                    # Show segment
                    cv2.imshow('vertical segment no:' + str(cnt), roi)
                    cv2.waitKey(0)
                x = col
                cnt += 1

            while vertical_sum[col] >= max_value-10:
                if col<2:
                    break
                col -= 1
                x = col

    return segmented_images

def _binarize(image, blur=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1].astype(np.uint8))

    if VERBOSE:
        cv2.imshow("binarized image", thresh)
        cv2.waitKey(0)

    return thresh

def _trim_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.bitwise_not(cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1].astype(np.uint8))
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)

    return image[y:y+h, x:x+w]

def _skew_correction(image):
    thresh = _binarize(image, blur=True)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

    return rotated

def _repeatedly_merge_contours(ctrs, area):
    # See _merge_contours for why this function is needed.
    while len(ctrs) != 1:
        new_ctrs = _merge_contours(ctrs, area)
        if len(new_ctrs) == len(ctrs):
            # all ctrs are at least the minimum area
            break
        ctrs = new_ctrs

    return ctrs

def _merge_contours(ctrs, area):
    # This function merges small contours together. However, the merged contour may still
    # be smaller than the minimum area that a contour should have (specified by the argument
    # 'area'). To remedy for this, one can call this function repeatedly to make sure that
    # all contours have at least the minimum area. However, in practice, only calling this
    # function once seems to work just fine.
    new_ctrs = []
    i = 0
    while i < len(ctrs):
        if i == len(ctrs)-1:
            new_ctrs.append(ctrs[i])
        else:
            if cv2.contourArea(ctrs[i+1]) > area:
                new_ctrs.append(ctrs[i])
            else:
                if i+2 < len(ctrs):
                    new_ctrs.append(_merge(ctrs[i], ctrs[i+2]))
                    i += 2
                else:
                    new_ctrs.append(_merge(ctrs[i], ctrs[i+1]))
                    i += 1
        i += 1

    return new_ctrs

def _merge(a, b):
    return np.array([[a[0,0]], [a[1,0]], [b[2,0]], [b[3,0]]])

def _resize_images(images, image_size, flip_image=True, buckets=1):
    features = []
    seq_len = []

    if buckets > 1 and image_size[0] is not None:
        raise ValueError("Cannot have more than one bucket if image size is completely specified")

    for image in images:
        if VERBOSE:
            cv2.imshow("original image", image)
            cv2.waitKey(0)
        f, s = _resize_image(_binarize(image, blur=True), image_size, flip_image=flip_image)
        if VERBOSE:
            cv2.imshow("resized image", f)
            cv2.waitKey(0)
        features.append(f)
        seq_len.append(s)

    if buckets == 1:
        features = [features]
        seq_len = [np.array(seq_len)]
        bucket_indices = None
    else:
        features, seq_len, bucket_indices = _bucket(features, seq_len, buckets)

    if image_size[0] is None:
        resized_features = []
        for features_bucket, seq_len_bucket in zip(features, seq_len):
            new_features = []
            max_seq_len = max([i for i in seq_len_bucket])
            for f in features_bucket:
                n_f = _pad_image_horizontally(f, max_seq_len, flip_image=flip_image)
                new_features.append(n_f)

                if VERBOSE:
                    cv2.imshow("padded image", n_f)
                    cv2.waitKey(0)

            resized_features.append(np.array(new_features))

        features = resized_features

    return np.array(features), np.array(seq_len), bucket_indices

def _bucket(images, seq_len, num_buckets):
    # Bucketing may require some experimentation for each dataset. If there are too many buckets and too few points
    # then some buckets may be empty because the total range is divided *equally* between all buckets. Empty buckets
    # may cause errors in later functions.
    # This function does not remove empty buckets deliberately, because otherwise the user may think that the data is
    # being divided into the number of buckets specified, which may then lead to reporting errors.

    min_seq_len = min(seq_len)
    max_seq_len = max(seq_len)
    bins = np.arange(min_seq_len, max_seq_len, (max_seq_len-min_seq_len)/num_buckets)
    bucket_indices = np.digitize(seq_len, bins) # should be a list of values ranging from 1 to num_buckets inclusively

    # This does bucketing in O(N) where N is the number of images, i.e. 
    # we only have to traverse the bucket_indices list of size N once.
    bucket_images = [[] for _ in range(num_buckets)]
    bucket_seq_len = [[] for _ in range(num_buckets)]

    for i, j in enumerate(bucket_indices):
        bucket_images[j-1].append(images[i])
        bucket_seq_len[j-1].append(np.array(seq_len[i]))

    for i in range(num_buckets):
        bucket_seq_len[i] = np.array(bucket_seq_len[i])

    return bucket_images, bucket_seq_len, bucket_indices

def _resize_image(image, image_size, flip_image=True):
    (wt, ht) = image_size
    (h, w) = image.shape

    if wt is None:
        f = ht / h
        new_size = (int(w*f), ht)
    else:
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))

    image = cv2.resize(image, new_size)

    if wt is not None:
        image = _copy_to_target(image, (ht,wt), flip_image)
        image = _normalize(image)

    return image, new_size[0]

def _transpose(image):
    return cv2.transpose(image)

def _normalize(image):
    (m, s) = cv2.meanStdDev(image)
    m = m[0][0]
    s = s[0][0]
    image = image - m
    image = image / s if s>0 else image

    return image

def _pad_image_horizontally(image, max_width, flip_image=True):
    image = _copy_to_target(image, [image.shape[0], max_width], flip_image=flip_image)
    image = _normalize(image)

    return image

def _copy_to_target(image, target_size, flip_image=True):
    image_size = image.shape
    target = np.ones(target_size) * 0.

    if flip_image:
        target[-image_size[0]:, -image_size[1]:] = image
    else:
        target[:image_size[0], :image_size[1]] = image

    # transpose (i.e. width first and then height)
    image = _transpose(target)

    if flip_image:
        image = cv2.flip(image, 0)

    return image

def _get_filenames(path):
    return [os.path.join(path, f) for f in os.listdir(path) if _valid_image(f)]

def _valid_image(f):
    return True if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") else False

##############################################################################################

def handle_dataset_processing(path, save_path, size=None, params=DATASET_PREPARATION, verbose=False):
    global VERBOSE 
    VERBOSE = verbose

    tic = time.time()
    preprocessor = ImagePreprocessing(params)

    files = _get_filenames(path)
    for i, f in enumerate(files):
        preprocessor(f, save_path)

        if i % 500 == 0:
            print("Done ", i, "images\tTime taken: ", time.time() - tic, "secs")

        if size is not None and i == size - 1:
            break

    toc = time.time()
    print("Done ", len(files), "images\tTime taken: ", toc - tic, "secs")

def handle_training(filenames, image_size, flip_image=True, buckets=1, params=TRAINING):
    preprocessor = ImagePreprocessing(params)
    
    images = []
    for filename in filenames:
        images.append(preprocessor(filename, save_path=None)[0])

    images, seq_len, bucket_indices = _resize_images(images, image_size, flip_image=flip_image, buckets=buckets)

    return images, seq_len, bucket_indices

def handle_inferring(paths, image_size, flip_image=True, buckets=1, params=INFERRING):
    """
    'paths' can be one of the following:
        - string: in which case it can be a:
            - path to an image, or
            - directory containing images
        - list: in which case each element, a string, may be a:
            - path to an image, or
            - directory containing images
    Images cannot be fed directly into this function because we want to
    read raw images and apply all operations on them that were applied
    on the training data.
    """
    filenames = []

    if type(paths) == str:
        if os.path.isfile(paths):
            filenames = [paths]

        elif os.path.isdir(paths):
            filenames = _get_filenames(paths)

        else:
            raise ValueError("'paths' is a string but is not a valid path to a file or a directory")

    elif type(paths) == list:
        for i, path in enumerate(paths):
            assert type(path) == str, "%d element in 'paths' is not a string. Has type %s" % (i, type(path))
 
            if os.path.isfile(path):
                filenames.extend([path])

            elif os.path.isdir(path):
                filenames.extend(_get_filenames(path))

            else:
                raise ValueError("%d element in 'paths' is not a path to a file or a directory" % i)

    else:
        raise ValueError("Unvalid 'paths' type %s" % type(paths))

    images = []
    preprocessor = ImagePreprocessing(params)

    for f in filenames:
        images.extend(preprocessor(f, save_path=None))

    images, seq_len, _ = _resize_images(images, image_size, flip_image=flip_image, buckets=buckets)
    
    if buckets == 1:
        images = images[0]
        seq_len = seq_len[0]

    return images, seq_len
