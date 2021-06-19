import cv2
import numpy as np


def resize_img(img, scale_factor):
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]),
                     new_size[1] / float(img.shape[1])]
    return new_img, actual_factor


def peturb_bbox(bboxes, pf=0, jf=0):
    '''
    Jitters and pads the input bbox.

    Args:
        bbox: Zero-indexed tight bbox.
        pf: padding fraction.
        jf: jittering fraction.
    Returns:
        pet_bbox: Jittered and padded box. Might have -ve or out-of-image coordinates
    '''
    jitter = np.random.random(4)
    pet_bboxes = []
    for bbox in bboxes:
        pet_bbox = [coord for coord in bbox]
        bwidth = bbox[2] - bbox[0] + 1
        bheight = bbox[3] - bbox[1] + 1

        pet_bbox[0] -= (pf * bwidth) + (1 - 2 * jitter[0]) * jf * bwidth
        pet_bbox[1] -= (pf * bheight) + (1 - 2 * jitter[1]) * jf * bheight
        pet_bbox[2] += (pf * bwidth) + (1 - 2 * jitter[2]) * jf * bwidth
        pet_bbox[3] += (pf * bheight) + (1 - 2 * jitter[3]) * jf * bheight

        pet_bboxes.append(pet_bbox)
    pet_bboxes = np.array(pet_bboxes)
    return pet_bboxes


def square_bbox(bboxes):
    '''
    Converts a bbox to have a square shape by increasing size along non-max dimension.
    '''
    bboxes_new = []
    for bbox in bboxes:
        sq_bbox = [int(round(coord)) for coord in bbox]
        bwidth = sq_bbox[2] - sq_bbox[0] + 1
        bheight = sq_bbox[3] - sq_bbox[1] + 1
        maxdim = float(max(bwidth, bheight))

        dw_b_2 = int(round((maxdim - bwidth) / 2.0))
        dh_b_2 = int(round((maxdim - bheight) / 2.0))

        sq_bbox[0] -= dw_b_2
        sq_bbox[1] -= dh_b_2
        sq_bbox[2] = sq_bbox[0] + maxdim - 1
        sq_bbox[3] = sq_bbox[1] + maxdim - 1
        bboxes_new.append(sq_bbox)
    bboxes_new = np.array(bboxes_new)
    return bboxes_new


def crop(img, bbox, bgval=0, mode='img'):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.

    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image        
    '''
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]

    img_out = np.ones((bheight, bwidth, nc)) * bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2] + 1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3] + 1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg
    if mode == 'img':
        img_cropped = img[y_min_src:y_max_src, x_min_src:x_max_src]
        x_p = bheight - img_cropped.shape[0]
        y_p = bwidth - img_cropped.shape[1]
        pad_size = [(x_p // 2, x_p // 2 + x_p % 2), (y_p // 2, y_p // 2 + y_p % 2), (0, 0)]
        # img_out = np.pad(img_cropped, pad_size, mode='edge')
        img_out = np.pad(img_cropped, pad_size, mode='constant', constant_values=1.)
        return img_out
    else:
        img_cropped = img[y_min_src:y_max_src, x_min_src:x_max_src]
        x_p = bheight - img_cropped.shape[0]
        y_p = bwidth - img_cropped.shape[1]
        pad_size = [(x_p // 2, x_p // 2 + x_p % 2), (y_p // 2, y_p // 2 + y_p % 2), (0, 0)]
        img_out = np.pad(img_cropped, pad_size, mode='constant')
        return img_out

def crop_v2(img, bbox, bgval=0):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.

    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image
    '''
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]

    img_out = np.ones((bheight, bwidth, nc)) * bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2] + 1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3] + 1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg

    img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = img[y_min_src:y_max_src, x_min_src:x_max_src, :]
    return img_out


def crop_landmarks(landmarks, img, bbox, bgval=0, mode='img'):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.

    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image
    '''

    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2] + 1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3] + 1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg
    img_cropped = img[y_min_src:y_max_src, x_min_src:x_max_src]

    vis = landmarks[:, 2] > 0
    landmarks[vis, 0] -= x_min_src
    landmarks[vis, 1] -= y_min_src

    x_p = bheight - img_cropped.shape[0]
    y_p = bwidth - img_cropped.shape[1]
    pad_size = [(x_p // 2, x_p // 2 + x_p % 2), (y_p // 2, y_p // 2 + y_p % 2), (0, 0)]
    landmarks[vis, 1] += x_p // 2
    landmarks[vis, 0] += y_p // 2

    return landmarks


def compute_dt(mask):
    """
    Computes distance transform of mask.
    """
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(1 - mask) / max(mask.shape)
    return dist


def compute_dt_barrier(mask, k=50):
    """
    Computes barrier distance transform of mask.
    """
    from scipy.ndimage import distance_transform_edt
    dist_out = distance_transform_edt(1 - mask)
    dist_in = distance_transform_edt(mask)

    dist_diff = (dist_out - dist_in) / max(mask.shape)

    dist = 1. / (1 + np.exp(k * -dist_diff))
    return dist
