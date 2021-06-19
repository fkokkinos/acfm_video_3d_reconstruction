import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from . import image_utils
from . import transformations
import pickle
import glob
import numpy as np


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


class Pascal_VOC(Dataset):
    """TigDog dataset."""

    def __init__(self, root, category, transforms=None, normalize=True,
                 split='train', img_size=256, scale=True, crop=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert category in ['dog', 'cat', 'sheep', 'cow']
        self.category = category
        self.root_dir = root

        self.normalize = normalize
        self.file_paths = glob.glob(self.root_dir + category + '/all/*.pkl')
        self.file_paths.sort()
        self.file_paths = np.array(self.file_paths)
        self.transforms = transforms
        self.scale = scale
        self.crop = crop
        self.split = split
        self.img_size = img_size

        split_num = int(self.file_paths.shape[0] * 0.7)
        if split == 'train':
            self.file_paths = self.file_paths[:split_num]
        elif split == 'test':
            self.file_paths = self.file_paths[split_num:]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx_loader):
        sample = pickle.load(open(self.file_paths[idx_loader], 'rb'))
        # print(sample.keys())
        images = sample['image'][None]
        segmentations = sample['mask'][None]
        bboxes = sample['bbox'][None]
        landmarks = sample['landmarks'][None]
        P3 = sample['P3']
        # idx = sample['idx']
        # filenames = sample['filenames']
        sfm_poses = sample['sfm_pose'][None]
        sfm_poses[0, 0] = 1
        bboxes = image_utils.square_bbox(bboxes)

        if self.crop:
            # crop image around bbox, translate kps
            images, segmentations, landmarks = self.crop_image(images, segmentations, bboxes, landmarks)

        if self.scale:
            # scale image, and mask. And scale kps.
            images, segmentations, landmarks = self.scale_image(images, segmentations, landmarks)

        # Normalize kp to be [-1, 1]
        if self.normalize:
            img_h, img_w = images.shape[1:3]
            landmarks = self.normalize_kp(landmarks, img_h, img_w)

        sample = {'img': images[0].transpose(2, 0, 1).astype(np.float32),
                  'kp': landmarks[0].astype(np.float32),
                  'mask': segmentations[0].astype(np.float32), 'sfm_pose': sfm_poses[0], 'P3': P3}
        return sample

    def crop_image(self, images, segmentations, bboxes_pred, landmarks):
        # crop image and mask and translate kps
        images_new, segmentations_new, landmarks_new = [], [], []
        for img, mask, bbox, landmark in zip(images, segmentations, bboxes_pred, landmarks):
            img = image_utils.crop(img, bbox, bgval=1)
            mask = image_utils.crop(mask[..., None], bbox, bgval=0)[..., 0]
            vis = landmark[:, 2] > 0
            landmark[vis, 0] -= bbox[0].astype(int)
            landmark[vis, 1] -= bbox[1].astype(int)

            landmark[..., 2] = vis

            images_new.append(img)
            segmentations_new.append(mask)
            landmarks_new.append(landmark)

        return np.array(images_new), np.array(segmentations_new), np.array(landmarks_new)

    def scale_image(self, images, segmentations_pred, landmarks):
        # Scale image so largest bbox size is img_size
        images_new, segmentations_pred_new, landmarks_new = [], [], []

        for img, mask, landmark in zip(images, segmentations_pred, landmarks):
            bwidth = np.shape(img)[0]
            bheight = np.shape(img)[1]
            scale = self.img_size / float(max(bwidth, bheight))
            img_scale, _ = resize_img(img, scale)
            vis = landmark[:, 2] > 0
            mask_scale, _ = resize_img(mask, scale)

            mask_scale = mask_scale.astype(np.bool)
            landmark[vis, :2] = np.round(landmark[vis, :2].astype(np.float32) * scale)

            images_new.append(img_scale)
            segmentations_pred_new.append(mask_scale)
            landmarks_new.append(landmark)
        return np.array(images_new), np.array(segmentations_pred_new), np.array(landmarks_new)

    def normalize_kp(self, landmarks, img_h, img_w):
        kp = landmarks[:, :, :2]
        vis_kp = landmarks[:, :, 2][..., None]
        new_kp = np.stack([2 * (kp[:, :, 0] / img_w) - 1,
                           2 * (kp[:, :, 1] / img_h) - 1]).transpose(1, 2, 0)
        new_landmarks = np.concatenate((vis_kp * new_kp, vis_kp), axis=-1)

        return new_landmarks


import cv2


def resize_img(img, scale_factor):
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]),
                     new_size[1] / float(img.shape[1])]
    return new_img, actual_factor


# a simple custom collate function, just to show the idea
def TigDog_collate(batch):
    # find max number of frames
    min_f = int(min([item['video'].shape[0] for item in batch]))
    indices = []

    for item in batch:
        if (item['video'].shape[0] - min_f) != 0:
            idx_ = np.random.randint(0, item['video'].shape[0] - min_f)
        else:
            idx_ = 0
        indices.append(idx_)
    padded_batch = {}
    for k in batch[0].keys():
        if k != 'idx' and k != 'filenames':
            data = []
            for item, idx_ in zip(batch, indices):
                data.append(torch.Tensor(item[k])[idx_: idx_ + min_f])
            data_padded = torch.stack(data)
        else:
            if k == 'idx':
                data_padded = [item[k] for item in batch]
            elif k == 'filenames':
                data = []
                for item, idx_ in zip(batch, indices):
                    data.append(item[k][idx_: idx_ + min_f])
                data_padded = np.stack(data)
        padded_batch[k] = data_padded
    return padded_batch


# a simple custom collate function, just to show the idea
def TigDog_collate_pad(batch):
    # find max number of frames
    max_f = int(max([item['video'].shape[0] for item in batch]))
    padded_batch = {}
    for k in batch[0].keys():
        data = [torch.Tensor(item[k]) for item in batch]
        if k == 'video':
            data = [item.permute(3, 0, 1, 2)[:, None] for item in data]
            data_padded = []
            for item in data:
                while max_f - item.shape[2] > 0:
                    item = F.pad(item, (0, 0, 0, 0, 0, max_f - item.shape[2]), "circular")
                data_padded.append(item)

            data_padded = torch.stack([item[:, 0].permute(1, 2, 3, 0) for item in data_padded])

        elif k == 'segmentations_pred':
            data = [item[None, None] for item in data]
            data_padded = []
            for item in data:
                while max_f - item.shape[2] > 0:
                    item = F.pad(item, (0, 0, 0, 0, 0, max_f - item.shape[2]), "circular")
                data_padded.append(item)

            data_padded = torch.stack([item[0, 0] for item in data_padded]).bool()

        elif k == 'sfm_poses':
            data = [item[None, None] for item in data]
            data_padded = []
            for item in data:
                while max_f - item.shape[2] > 0:
                    item = F.pad(item, (0, 0, 0, max_f - item.shape[2]), "circular")
                data_padded.append(item)

            data_padded = torch.stack([item[0, 0] for item in data_padded])

        elif k == 'landmarks':
            data = [item[None, None] for item in data]
            data_padded = []
            for item in data:
                while max_f - item.shape[2] > 0:
                    item = F.pad(item, (0, 0, 0, 0, 0, max_f - item.shape[2]), "circular")
                data_padded.append(item)

            data_padded = torch.stack([item[0, 0] for item in data_padded])

        elif k == 'bboxes_pred':
            data = [item[None, None] for item in data]
            data_padded = []
            for item in data:
                while max_f - item.shape[2] > 0:
                    item = F.pad(item, (0, 0, 0, max_f - item.shape[2]), "circular")
                data_padded.append(item)

            data_padded = torch.stack([item[0, 0] for item in data_padded])
        elif k == 'optical_flows':
            data = [item.permute(3, 0, 1, 2)[:, None] for item in data]
            data_padded = []
            for item in data:
                while max_f - item.shape[2] > 0:
                    item = F.pad(item, (0, 0, 0, 0, 0, max_f - item.shape[2]), "circular")
                data_padded.append(item)

            data_padded = torch.stack([item[:, 0].permute(1, 2, 3, 0) for item in data_padded])
        elif k == 'idx':
            data_padded = [item[k] for item in batch]

        padded_batch[k] = data_padded
    return padded_batch
