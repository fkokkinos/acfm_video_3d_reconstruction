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
import flowiz as fz
from . import image_utils
from . import transformations
import pandas as pd
import pickle
import glob
import os.path as osp
import numpy as np
import random
import scipy.io as sio
from absl import flags, app
from .optical_flow import config_folder as cf
from .optical_flow.model import MaskFlownet, MaskFlownet_S, Upsample
import yaml
from skimage.measure import regionprops
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'misc', 'cachedir')
opts = flags.FLAGS


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


@gin.configurable
class YT_MultiFrame(Dataset):
    """TigDog dataset."""

    def __init__(self, root, category, sample_to_vid, samples_per_vid, num_frames=2, transforms=None, normalize=True,
                 remove_neck_kp=True, split='train', img_size=256, mirror=False, scale=True, crop=True, offset=5):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert category in ['cow', 'giraffe', 'elephant']
        self.mirror = mirror
        self.category = category
        self.root_dir = root
        self.num_frames = num_frames
        self.normalize = normalize
        self.file_paths = glob.glob(self.root_dir + category + '/*.pkl')
        self.file_paths.sort()
        self.file_paths = np.array(self.file_paths)
        self.transforms = transforms
        self.scale = scale
        self.crop = crop
        self.split = split
        self.img_size = img_size
        self.sample_to_vid = sample_to_vid
        self.samples_per_vid = samples_per_vid
        # neck keypoint is inconsistent wrt to orientation
        self.remove_neck_kp = remove_neck_kp
        self.offset = 3
        self.kp_perm = np.array([2, 1, 3, 5, 4, 7, 6, 8, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 19]) - 1
        if transforms is not None:
            self.transform = transformations.RandomAffine(scale=[0.9, 1.2], translate=(0.0, 0.0), resample=3)

    def __len__(self):
        return len(self.file_paths)

    def centralize(self, img1, img2):
        rgb_mean = torch.cat((img1, img2), 2)
        rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
        rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    def __getitem__(self, idx_loader):
        file_path = self.root_dir + self.category + '/' + str(idx_loader) + '.pkl'
        idx_loader = file_path.split('/')[-1]
        idx_loader = int(idx_loader.replace('.pkl', ''))
        vid = self.sample_to_vid[idx_loader]

        samples = self.samples_per_vid[vid].copy()
        idx_loader_pose = samples.index(idx_loader)
        of_left = max(idx_loader_pose - self.offset - 1, 0)
        of_right = min(idx_loader_pose + self.offset - 1, len(samples))
        samples = samples[of_left:of_right]
        samples.remove(idx_loader)
        frames = [idx_loader] + random.sample(samples, self.num_frames - 1)
        frames.sort()
        images = []
        segmentations = []
        bboxes = []
        landmarks = []
        sfm_poses = []
        for f in frames:
            sample = pickle.load(open(self.root_dir + self.category + '/' + str(f) + '.pkl', 'rb'))
            images.append(sample['video'])
            segmentations.append(sample['segmentations'])
            bboxes.append(sample['bboxes'])
            landmarks.append(sample['landmarks'])
            sfm_poses.append(sample['sfm_poses'])

        images = np.array(images)
        segmentations = np.array(segmentations)
        bboxes = np.array(bboxes)
        landmarks = np.array(landmarks)
        sfm_poses = np.array(sfm_poses)
        # optical_flows = []
        # for im0_idx, im0 in enumerate(images[:-1]):
        #     im1 = images[im0_idx + 1]
        #     with torch.no_grad():
        #
        #         im0 = torch.from_numpy(im0[None]).permute(0, 3, 1, 2).float()
        #         im1 = torch.from_numpy(im1[None]).permute(0, 3, 1, 2).float()
        #         im0 = F.interpolate(im0, size=[384, 768], mode='bilinear')
        #         im1 = F.interpolate(im1, size=[384, 768], mode='bilinear')
        #         im0_c, im1_c, _ = self.centralize(im0, im1)
        #
        #         shape = im0_c.shape
        #         pad_h = (64 - shape[2] % 64) % 64
        #         pad_w = (64 - shape[3] % 64) % 64
        #         if pad_h != 0 or pad_w != 0:
        #             im0_c = F.interpolate(im0_c, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
        #             im1_c = F.interpolate(im1_c, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
        #
        #         im0_c = im0_c.cuda()
        #         im1_c = im1_c.cuda()
        #         im0_ = torch.cat([im0_c, im1_c])
        #         im1_ = torch.cat([im1_c, im0_c])
        #         pred, flows, warpeds = self.of_net(im0_, im1_)
        #         up_flow = Upsample(pred[-1], 4)
        #         up_occ_mask = Upsample(flows[0], 4)
        #
        #         if pad_h != 0 or pad_w != 0:
        #             up_flow = F.interpolate(up_flow, size=[shape[2], shape[3]], mode='bilinear') * \
        #                       torch.tensor([shape[d] / up_flow.shape[d] for d in (2, 3)], device=im0.device).view(1, 2,
        #                                                                                                           1, 1)
        #             up_occ_mask = F.interpolate(up_occ_mask, size=[shape[2], shape[3]], mode='bilinear')
        #         flow_fw = up_flow[0].data.cpu().numpy()
        #         flow_bw = up_flow[1].data.cpu().numpy()
        #         norm_of = np.linalg.norm(flow_fw + flow_bw, ord=2, axis=0)
        #         perc_num = np.percentile(norm_of, 95)
        #         mask_of = norm_of < perc_num
        #         up_flow = up_flow[:1] * torch.from_numpy(mask_of).cuda()
        #         up_flow = F.interpolate(up_flow, size=[images.shape[1], images.shape[2]], mode='bilinear')
        #         optical_flows.append(up_flow.cpu().numpy())
        # optical_flows.append(torch.zeros_like(up_flow).cpu().numpy())
        # optical_flows = np.concatenate(optical_flows).transpose(0, 2, 3, 1)
        # idx = sample['idx']
        # filenames = sample['filenames']
        optical_flows = np.zeros((images.shape[0], images.shape[1], images.shape[2], 2))
        if self.crop:
            # crop image around bbox, translate kps
            images, segmentations, landmarks, sfm_poses, optical_flows = self.crop_image(images, segmentations, bboxes,
                                                                                         landmarks, sfm_poses,
                                                                                         optical_flows)
        if self.scale:
            # scale image, and mask. And scale kps.
            images, segmentations, landmarks, sfm_poses, optical_flows = self.scale_image(images, segmentations,
                                                                                          landmarks, sfm_poses,
                                                                                          optical_flows)
        # # Mirror image on random.
        mirror_flag = torch.zeros(images.shape[0])
        if self.mirror:
            images, segmentations, landmarks, sfm_poses, optical_flows, mirror_flag = self.mirror_image(images,
                                                                                                        segmentations,
                                                                                                        landmarks,
                                                                                                        sfm_poses,
                                                                                                        optical_flows)
        transform_params = np.zeros((images.shape[0], 4))
        transform_params[:, 0] = 1  # default scale is 1
        if self.transforms:
            images, segmentations, landmarks, transform_params = self.transform(images, segmentations, landmarks)

        optical_flows = []
        for im0_idx, im0 in enumerate(images[:-1]):
            im1 = images[im0_idx + 1]
            with torch.no_grad():

                im0 = torch.from_numpy(im0[None]).permute(0, 3, 1, 2).float()
                im1 = torch.from_numpy(im1[None]).permute(0, 3, 1, 2).float()
                im0 = F.interpolate(im0, size=[384, 768], mode='bilinear')
                im1 = F.interpolate(im1, size=[384, 768], mode='bilinear')
                im0_c, im1_c, _ = self.centralize(im0, im1)

                shape = im0_c.shape
                pad_h = (64 - shape[2] % 64) % 64
                pad_w = (64 - shape[3] % 64) % 64
                if pad_h != 0 or pad_w != 0:
                    im0_c = F.interpolate(im0_c, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
                    im1_c = F.interpolate(im1_c, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')

                im0_c = im0_c.cuda()
                im1_c = im1_c.cuda()
                # im0_ = torch.cat([im0_c, im1_c])
                # im1_ = torch.cat([im1_c, im0_c])
                pred, flows, warpeds = self.of_net(im0_c, im1_c)
                up_flow = Upsample(pred[-1], 4)
                up_occ_mask = Upsample(flows[0], 4)

                if pad_h != 0 or pad_w != 0:
                    up_flow = F.interpolate(up_flow, size=[shape[2], shape[3]], mode='bilinear') * \
                              torch.tensor([shape[d] / up_flow.shape[d] for d in (2, 3)], device=im0.device).view(1, 2,
                                                                                                                  1, 1)
                    up_occ_mask = F.interpolate(up_occ_mask, size=[shape[2], shape[3]], mode='bilinear')
                # flow_fw = up_flow[0].data.cpu().numpy()
                # flow_bw = up_flow[1].data.cpu().numpy()
                # norm_of = np.linalg.norm(flow_fw + flow_bw, ord=2, axis=0)
                # perc_num = np.percentile(norm_of, 95)
                # mask_of = norm_of < perc_num
                # up_flow = up_flow[:1] * torch.from_numpy(mask_of).cuda()
                up_flow = up_flow[:1]
                up_flow = F.interpolate(up_flow, size=[images.shape[1], images.shape[2]], mode='bilinear')
                optical_flows.append(up_flow.cpu().numpy())
        optical_flows.append(torch.zeros_like(up_flow).cpu().numpy())
        optical_flows = np.concatenate(optical_flows).transpose(0, 2, 3, 1)
        # # Normalize kp to be [-1, 1]
        if self.normalize:
            img_h, img_w = images.shape[1:3]
            landmarks, sfm_poses, optical_flows = self.normalize_kp(landmarks, sfm_poses, optical_flows, img_h, img_w)

        if self.remove_neck_kp:
            landmarks = landmarks[:, :-1]
        sample = {'img': images.transpose(0, 3, 1, 2).astype(np.float32),
                  'kp': landmarks.astype(np.float32),
                  'mask': segmentations.astype(np.float32), 'sfm_pose': sfm_poses,
                  'optical_flows': optical_flows, 'frames_idx': frames, 'mirror_flag': mirror_flag,
                  'transforms': transform_params.astype(np.float32)}
        return sample

    def mirror_image(self, images, segmentations_pred, landmarks, sfm_poses, optical_flows):
        kp_perm = self.kp_perm
        flag = torch.zeros(images.shape[0])
        if torch.rand(1) > 0.5:
            flag = torch.ones(images.shape[0])

            # Need copy bc torch collate doesnt like neg strides
            images_flip = images[:, :, ::-1, :].copy()
            optical_flows_flip = optical_flows[:, :, ::-1, :].copy()
            segmentations_pred_flip = segmentations_pred[:, :, ::-1].copy()

            # Flip kps.
            new_x = images.shape[2] - landmarks[:, :, 0] - 1
            kp_flip = np.concatenate((new_x[:, :, None], landmarks[:, :, 1:]), axis=-1)
            kp_flip = kp_flip[:, kp_perm, :]
            # Flip sfm_pose Rot.
            sfm_poses_flip = sfm_poses.copy()
            for sfm_pose in sfm_poses_flip:
                R = transformations.quaternion_matrix(sfm_pose[3:])
                flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
                sfm_pose[3:] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
                # Flip tx
                # tx = images.shape[2] - sfm_pose[1] - 1
                tx = -1 * sfm_pose[1]
                sfm_pose[1] = tx
            return images_flip, segmentations_pred_flip, kp_flip, sfm_poses_flip, optical_flows_flip, flag
        else:
            return images, segmentations_pred, landmarks, sfm_poses, optical_flows, flag

    def crop_image(self, images, segmentations, bboxes_pred, landmarks, sfm_poses, optical_flows):
        # crop image and mask and translate kps
        images_new, segmentations_new, landmarks_new, sfm_poses_new, optical_flows_new = [], [], [], [], []
        for img, mask, bbox, landmark, sfm_pose, optical_flow in zip(images, segmentations,
                                                                     bboxes_pred, landmarks,
                                                                     sfm_poses, optical_flows):
            landmark = image_utils.crop_landmarks(landmark, img, bbox, img.shape)
            img = image_utils.crop(img, bbox, bgval=1, mode='img')
            optical_flow = image_utils.crop(optical_flow, bbox, bgval=0, mode=None)
            mask = image_utils.crop(mask[..., None], bbox, bgval=0, mode=None)[..., 0]
            # vis = landmark[:, 2] > 0
            # landmark[vis, 0] -= bbox[0].astype(int)
            # landmark[vis, 1] -= bbox[1].astype(int)

            # landmark[..., 2] = vis
            # sfm_pose[1] -= bbox[0]
            # sfm_pose[2] -= bbox[1]

            images_new.append(img)
            segmentations_new.append(mask)
            landmarks_new.append(landmark)
            sfm_poses_new.append(sfm_pose)
            optical_flows_new.append(optical_flow)

        return np.array(images_new), np.array(segmentations_new), np.array(landmarks_new), \
               np.array(sfm_poses_new), np.array(optical_flows_new)

    def scale_image(self, images, segmentations_pred, landmarks, sfm_poses, optical_flows):
        # Scale image so largest bbox size is img_size
        images_new, segmentations_pred_new, landmarks_new, sfm_poses_new, optical_flows_new = [], [], [], [], []

        for img, mask, landmark, sfm_pose, optical_flow in zip(images, segmentations_pred, landmarks,
                                                               sfm_poses, optical_flows):
            bwidth = np.shape(img)[0]
            bheight = np.shape(img)[1]
            scale = self.img_size / float(max(bwidth, bheight))
            img_scale, _ = resize_img(img, scale)
            vis = landmark[:, 2] > 0
            mask_scale, _ = resize_img(mask.astype(np.float32), scale)
            H, W = optical_flow.shape[:2]
            optical_flow[:, :, 0] = 2.0 * optical_flow[:, :, 0].copy() / max(W - 1, 1) - 1.0
            optical_flow[:, :, 1] = 2.0 * optical_flow[:, :, 1].copy() / max(H - 1, 1) - 1.0
            optical_flow_scale, _ = resize_img(optical_flow, scale)
            H, W = optical_flow_scale.shape[:2]
            optical_flow_scale[:, :, 0] = (optical_flow_scale[:, :, 0].copy() + 1.0) * max(W - 1, 1) / 2
            optical_flow_scale[:, :, 1] = (optical_flow_scale[:, :, 1].copy() + 1.0) * max(H - 1, 1) / 2

            mask_scale = mask_scale.astype(np.bool)
            landmark[vis, :2] = np.round(landmark[vis, :2].astype(np.float32) * scale)
            # sfm_pose[0] *= scale
            # sfm_pose[1] *= scale
            # sfm_pose[2] *= scale

            images_new.append(img_scale)
            segmentations_pred_new.append(mask_scale)
            landmarks_new.append(landmark)
            sfm_poses_new.append(sfm_pose)
            optical_flows_new.append(optical_flow_scale)
        return np.array(images_new), np.array(segmentations_pred_new), np.array(landmarks_new), \
               np.array(sfm_poses_new), np.array(optical_flows_new)

    def normalize_kp(self, landmarks, sfm_poses, optical_flows, img_h, img_w):
        sfm_poses_new = sfm_poses.copy()
        kp = landmarks[:, :, :2]
        vis_kp = landmarks[:, :, 2][..., None]
        new_kp = np.stack([2 * (kp[:, :, 0] / img_w) - 1,
                           2 * (kp[:, :, 1] / img_h) - 1]).transpose(1, 2, 0)
        # sfm_poses_new[:, 0] *= (1.0 / img_w + 1.0 / img_h)
        # sfm_poses_new[:, 1] = 2.0 * (sfm_poses[:, 1] / img_w) - 1
        # sfm_poses_new[:, 2] = 2.0 * (sfm_poses[:, 2] / img_h) - 1

        new_landmarks = np.concatenate((vis_kp * new_kp, vis_kp), axis=-1)

        return new_landmarks, sfm_poses_new, optical_flows


import cv2


def resize_img(img, scale_factor):
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]),
                     new_size[1] / float(img.shape[1])]
    return new_img, actual_factor


# a simple custom collate function, just to show the idea
def YT_collate(batch):
    # find max number of frames
    min_f = int(min([item['img'].shape[0] for item in batch]))
    indices = []

    for item in batch:
        if (item['img'].shape[0] - min_f) != 0:
            idx_ = np.random.randint(0, item['img'].shape[0] - min_f)
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
