import math
import os
import pickle as pkl

import gin
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from skimage import io
from torch.utils.data import Dataset
from . import image_utils
from . import transformations
import pickle
import glob


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


class COCO_final(Dataset):
    """YTVIS_final dataset."""

    def __init__(self, root, category, transforms=None, normalize=True,
                 max_length=None, split='train', img_size=256, mirror=False,
                 scale=True, crop=True, num_kps=19):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mirror = mirror
        self.category = category
        self.root_dir = root
        self.num_kps = num_kps
        print(self.root_dir + category + '/*.pkl')
        self.normalize = normalize
        self.file_paths = glob.glob(self.root_dir + category + '/*.pkl')
        self.file_paths.sort()
        self.file_paths = np.array(self.file_paths)
        self.transforms = transforms
        self.max_length = max_length
        self.scale = scale
        self.crop = crop
        self.split = split
        self.img_size = img_size
        self.calc_part_segmentation = True
        # neck keypoint is inconsistent wrt to orientation

        if self.split != 'all':
            num_videos = len(self.file_paths)
            video_range = np.random.RandomState(seed=42).permutation(num_videos)
            test_video = video_range[-14:]
            train_video = video_range[:-14]
            if self.split == 'train':
                self.file_paths = self.file_paths[train_video]
            else:
                self.file_paths = self.file_paths[test_video]
                print(test_video)

    def __len__(self):
        return self.file_paths.shape[0]

    def __getitem__(self, idx_loader):
        try:
            sample = pickle.load(open(self.root_dir + self.category + '/' + str(idx_loader) + '.pkl', 'rb'))
        except Exception as e:
            raise IndexError
        images = sample['video'] / 255
        segmentations = sample['segmentations']
        bboxes = sample['bboxes']
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        bboxes = image_utils.square_bbox(bboxes)

        if self.crop:
            # crop image around bbox, translate kps
            images, segmentations = self.crop_image(images, segmentations, bboxes)
        if self.scale:
            # scale image, and mask. And scale kps.
            images, segmentations = self.scale_image(images, segmentations)
        # # Mirror image on random.
        if self.mirror:
            images, segmentations = self.mirror_image(images, segmentations)

        if self.max_length is not None:
            idx_ = 0
            images = images[idx_:idx_ + self.max_length]
            segmentations = segmentations[idx_:idx_ + self.max_length]
            bboxes = bboxes[idx_:idx_ + self.max_length]

        sfm_poses = np.zeros((images.shape[0], 7))
        sfm_poses[:, 3] = 1
        landmarks = np.zeros((images.shape[0], self.num_kps, 3))
        sample = {'video': images.astype(np.float32),
                  'segmentations': segmentations.astype(np.bool),
                  'bboxes': bboxes.astype(np.float32), 'idx': None, 'sfm_poses': sfm_poses, 'landmarks': landmarks}
        return sample

    def mirror_image(self, images, segmentations_pred):
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            images_flip = images[:, :, ::-1, :].copy()
            segmentations_pred_flip = segmentations_pred[:, :, ::-1].copy()

            return images_flip, segmentations_pred_flip
        else:
            return images, segmentations_pred

    def crop_image(self, images, segmentations, bboxes_pred):
        # crop image and mask and translate kps
        images_new, segmentations_new = [], []
        for img, mask, bbox in zip(images, segmentations, bboxes_pred):
            img = image_utils.crop(img, bbox, bgval=1)
            mask = image_utils.crop(mask[..., None], bbox, bgval=0, mode='mask')[..., 0]
            images_new.append(img)
            segmentations_new.append(mask)

        return np.array(images_new), np.array(segmentations_new)

    def scale_image(self, images, segmentations_pred):
        # Scale image so largest bbox size is img_size
        images_new, segmentations_pred_new = [], []

        for img, mask in zip(images, segmentations_pred):
            bwidth = np.shape(img)[0]
            bheight = np.shape(img)[1]
            scale = self.img_size / float(max(bwidth, bheight))
            img_scale, _ = resize_img(img, scale)
            mask_scale, _ = resize_img(mask, scale)
            mask_scale = mask_scale.astype(np.bool)
            images_new.append(img_scale)
            segmentations_pred_new.append(mask_scale)

        return np.array(images_new), np.array(segmentations_pred_new)

    def normalize_kp(self, landmarks, sfm_poses, optical_flows, img_h, img_w):
        sfm_poses_new = sfm_poses.copy()
        optical_flows_new = optical_flows.copy()
        kp = landmarks[:, :, :2]
        vis_kp = landmarks[:, :, 2][..., None]
        new_kp = np.stack([2 * (kp[:, :, 0] / img_w) - 1,
                           2 * (kp[:, :, 1] / img_h) - 1]).transpose(1, 2, 0)
        optical_flows_new[..., 0] = 2.0 * (optical_flows_new[..., 0] / img_w) - 1
        optical_flows_new[..., 1] = 2.0 * (optical_flows_new[..., 1] / img_h) - 1

        new_landmarks = np.concatenate((vis_kp * new_kp, vis_kp), axis=-1)

        return new_landmarks, sfm_poses_new, optical_flows_new


import cv2


def resize_img(img, scale_factor):
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]),
                     new_size[1] / float(img.shape[1])]
    return new_img, actual_factor
