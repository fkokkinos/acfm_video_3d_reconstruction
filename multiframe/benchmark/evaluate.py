from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import pickle as pkl
import warnings

import numpy as np
import scipy.io as sio
import torch
import torchvision
import trimesh
from absl import app
from absl import flags
from data import tigdog_final as tf_final
from data import tigdog_mf_of as tigdog_mf
from data import ytvis_final as yt_final
from nnutils import predictor as pred_utils
from nnutils import test_utils
from skimage import io
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import bird_vis

warnings.filterwarnings("ignore")

flags.DEFINE_boolean('visualize', False, 'Visualize outputs')
flags.DEFINE_boolean('v2_crop', True, 'if true visualizes things')
flags.DEFINE_boolean('tight_bboxes', True, 'if true visualizes things')
flags.DEFINE_string('split', 'test', 'split')
flags.DEFINE_float('padding_frac', 0., 'padding_frac')
flags.DEFINE_string('tmp_dir', 'tmp/', 'tmp dir to extract video dataset')
flags.DEFINE_string('category', 'horse', 'Category')
flags.DEFINE_string('root_dir', '/media/filippos/MyFiles/TigDog_new_wnrsfm/', 'root dir of TigDog dataset')
flags.DEFINE_integer('num_lbs', 128, 'number of handles')
flags.DEFINE_integer('num_frames', 2, 'number of frames to load')
flags.DEFINE_integer('num_train_frames', 50, 'number of train frames to load (in case of evaluating on train set)')
flags.DEFINE_string('mesh_dir', 'meshes/horse_new.obj', 'tmp dir to extract dataset')
flags.DEFINE_string('kp_dict', 'meshes/horse_kp_dictionary.pkl', 'tmp dir to extract dataset')
flags.DEFINE_string('results_dir', './', 'Mask predictions to load')
flags.DEFINE_string('root_dir_yt', '/media/filippos/MyFiles/TigDog_new_wnrsfm/', 'Root dir of YTVis dataset')
flags.DEFINE_boolean('expand_ytvis', False, 'Expand data with YTVis dataset')
opts = flags.FLAGS


class ShapeTester(test_utils.Tester):
    def define_model(self):
        opts = self.opts

        self.predictor = pred_utils.MeshPredictor(opts, testing_samples=self.testing_samples)

        # for visualization
        self.renderer = self.predictor.vis_rend
        self.vis_counter = 0

    def init_dataset(self):
        opts = self.opts
        # create video dataloaders
        if opts.category in ['horse', 'tiger']:
            self.dataset = tf_final.TigDogDataset_Final(opts.root_dir, opts.category, transforms=None, normalize=False,
                                                        max_length=None, remove_neck_kp=False, split=opts.split,
                                                        img_size=256, mirror=False, scale=False, crop=False)
            if opts.category == 'horse' and opts.expand_ytvis and opts.split == 'train':
                self.dataset2 = yt_final.YTVIS_final(opts.root_dir_yt, opts.category, transforms=None, normalize=False,
                                                     max_length=None, split='all', img_size=256,
                                                     mirror=False, scale=False, crop=False)
                self.dataset = torch.utils.data.ConcatDataset([self.dataset, self.dataset2])

            self.collate_fn = tf_final.TigDog_collate

        elif opts.category in ['cow', 'giraffe', 'elephant']:
            self.dataset = yt_final.YTVIS_final(opts.root_dir_yt, opts.category, transforms=None,
                                                normalize=False, max_length=None, split='all', img_size=256,
                                                mirror=False, scale=False, crop=False, num_kps=opts.num_kps)
        # save individual frames for fast loading purposes
        directory = opts.tmp_dir + '/' + opts.category + '/'
        if not osp.exists(directory):
            os.makedirs(directory)

        save_counter = 0
        sample_to_vid = {}
        samples_per_vid = {}
        for i_sample, sample in enumerate(self.dataset):
            num_frames = sample['video'].shape[0]
            for i in range(num_frames):
                new_sample = {}
                for k in sample.keys():
                    if k in ['video', 'sfm_poses', 'landmarks', 'segmentations', 'bboxes']:
                        new_sample[k] = sample[k][i]
                pkl.dump(new_sample, open(directory + str(save_counter) + '.pkl', 'wb'))
                sample_to_vid[save_counter] = i_sample
                if i_sample in samples_per_vid:
                    samples_per_vid[i_sample].append(save_counter)
                else:
                    samples_per_vid[i_sample] = [save_counter]
                save_counter += 1

                if i >= opts.num_train_frames and opts.split == 'train':
                    break

        self.testing_samples = save_counter
        print('Testing samples', self.testing_samples)
        # create multi-frame dataloaders
        if opts.category in ['horse', 'tiger']:
            self.dataset = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir, opts.category,
                                                              num_frames=opts.num_frames,
                                                              sample_to_vid=sample_to_vid,
                                                              samples_per_vid=samples_per_vid,
                                                              normalize=True, transforms=None,
                                                              remove_neck_kp=True, split='all', img_size=256,
                                                              mirror=False, scale=True, crop=True,
                                                              v2_crop=opts.v2_crop,
                                                              tight_bboxes=opts.tight_bboxes, sequential=True)
        elif opts.category in ['cow', 'giraffe', 'elephant', 'zebra', 'fox']:
            self.dataset = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir, opts.category,
                                                              num_frames=opts.num_frames,
                                                              sample_to_vid=sample_to_vid,
                                                              samples_per_vid=samples_per_vid,
                                                              normalize=True, transforms=True,
                                                              remove_neck_kp=False, split='train', img_size=256,
                                                              mirror=True, scale=True, crop=True,
                                                              padding_frac=opts.padding_frac)
            self.collate_fn = tigdog_mf.TigDog_collate
        self.dataloader = DataLoader(self.dataset, opts.batch_size, drop_last=False, shuffle=False)

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def evaluate(self, outputs, batch):
        """
        Compute IOU and keypoint error
        """
        opts = self.opts
        batch_size, t = batch['img'].shape[:2]
        # compute iou
        mask_tensor = batch['mask'].type(torch.FloatTensor)
        mask_tensor = mask_tensor[:, 0]
        mask_gt = mask_tensor.view(batch_size, -1).numpy()
        mask_pred = outputs['mask_pred'].cpu().view(batch_size, t, -1).type_as(batch['mask']).numpy()
        mask_pred = mask_pred[:, 0]
        intersection = mask_gt * mask_pred
        union = mask_gt + mask_pred - intersection
        iou = intersection.sum(1) / union.sum(1)

        # Compute pck
        kps_gt = batch['kp'].type(torch.FloatTensor)
        kps_gt = kps_gt[:, 0]
        kps_vis = kps_gt[:, :, 2]
        kps_gt = kps_gt[:, :, 0:2].cuda()
        kps_pred = outputs['kp_pred']
        kps_pred = kps_pred.reshape(batch_size, t, *kps_pred.shape[1:])
        kps_pred = kps_pred[:, 0]
        kps_pred_inds = (kps_pred + 1) * opts.img_size / 2
        kps_gt_inds = (kps_gt + 1) * opts.img_size / 2
        kps_err = torch.norm((kps_pred_inds - kps_gt_inds[..., 0:2]), dim=2)
        kps_err = kps_err.detach().cpu().numpy()

        return iou, kps_err, kps_vis

    def visualize(self, outputs, batch):
        directory = 'results_viz/' + self.opts.name + '_' + self.opts.split + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        batch_size, t = batch['img'].shape[:2]
        img_tensor = batch['img'].type(torch.FloatTensor)
        img_tensor = img_tensor.reshape(batch_size * t, *img_tensor.shape[2:]).clone()
        kp_tensor = batch['kp'].type(torch.FloatTensor)
        kp_tensor = kp_tensor.reshape(batch_size * t, *kp_tensor.shape[2:]).clone()

        vert = outputs['verts'][0]
        cam = outputs['cam_pred'][0]
        texture = outputs['texture'][0]
        kp = outputs['kp_pred'][0]
        img_pred = self.renderer(vert, cam, texture=texture) / 255
        pred_transformed_kp_img = bird_vis.kp2im(kp.data, img_tensor[0].data)
        input_img = bird_vis.kp2im(kp_tensor[0].data, img_tensor[0].data)

        if self.opts.optimize:
            vert = outputs['verts_orig'][0]
            kp = outputs['kp_pred_orig'][0]
            cam_pred_orig = outputs['cam_pred_orig'][0]
            img_pred_orig = self.renderer(vert, cam_pred_orig, texture=texture) / 255
            pred_transformed_kp_img_orig = bird_vis.kp2im(kp.data, img_tensor[0].data)
            img_ = np.hstack(
                [input_img, pred_transformed_kp_img_orig, pred_transformed_kp_img, img_pred_orig * 255, img_pred * 255])

        else:
            img_ = np.hstack([input_img, pred_transformed_kp_img, img_pred * 255])

        io.imsave(directory + str(self.vis_counter) + '.jpg', img_.astype(np.uint8))
        mesh_ = trimesh.Trimesh(vert.detach().cpu().numpy(), self.predictor.faces[0].detach().cpu().numpy(),
                                vertex_colors=texture.detach().cpu().numpy())
        mesh_.export(directory + str(self.vis_counter) + '.ply')
        self.vis_counter += 1

    def test(self):
        opts = self.opts
        bench_stats = {'ious': [], 'kp_errs': [], 'kp_vis': []}

        result_path = osp.join(opts.results_dir, 'results.mat')
        print('Writing to %s' % result_path)
        for i, batch in enumerate(tqdm(self.dataloader)):
            if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                break
            outputs = self.predictor.predict(batch)
            if opts.visualize:
                self.visualize(outputs, batch)
            iou, kp_err, kp_vis = self.evaluate(outputs, batch)
            bench_stats['ious'].append(iou)
            bench_stats['kp_errs'].append(kp_err)
            bench_stats['kp_vis'].append(kp_vis)

            if opts.save_visuals and (i % opts.visuals_freq == 0):
                self.save_current_visuals(batch, outputs)

        bench_stats['kp_errs'] = np.concatenate(bench_stats['kp_errs'], axis=0)
        bench_stats['kp_vis'] = np.concatenate(bench_stats['kp_vis'], axis=0)
        bench_stats['ious'] = np.concatenate(bench_stats['ious'])

        sio.savemat(result_path, bench_stats)

        # Report numbers.
        mean_iou = bench_stats['ious'].mean()

        n_vis_p = np.sum(bench_stats['kp_vis'], axis=0)
        n_correct_p_pt1 = np.sum(
            (bench_stats['kp_errs'] < (0.1 * opts.img_size)) * bench_stats['kp_vis'], axis=0)
        n_correct_p_pt15 = np.sum(
            (bench_stats['kp_errs'] < (0.15 * opts.img_size)) * bench_stats['kp_vis'], axis=0)
        # remove non visible kps to avoid NaNs
        remove = []
        for i_vis, vis_p in enumerate(n_vis_p):
            if vis_p == 0:
                remove.append(i_vis)
        n_vis_p = np.delete(n_vis_p, remove)
        n_correct_p_pt1 = np.delete(n_correct_p_pt1, remove)
        n_correct_p_pt15 = np.delete(n_correct_p_pt15, remove)
        pck1 = (n_correct_p_pt1 / n_vis_p).mean()
        pck15 = (n_correct_p_pt15 / n_vis_p).mean()
        print('%s mean iou %.3g, pck.1 %.3g, pck.15 %.3g' %
              (osp.basename(result_path), mean_iou, pck1, pck15))


def main(_):
    opts.results_dir = osp.join(opts.results_dir_base, '%s' % (opts.split),
                                opts.name, 'epoch_%d' % opts.num_train_epoch)
    if not osp.exists(opts.results_dir):
        print('writing to %s' % opts.results_dir)
        os.makedirs(opts.results_dir)

    torch.manual_seed(0)
    tester = ShapeTester(opts)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run(main)
