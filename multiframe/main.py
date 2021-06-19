from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
from skimage.transform import resize
import os.path as osp
import numpy as np
import torch
import torchvision
from collections import OrderedDict

from data import tigdog_final as tf_final
from data import ytvis_final as yt_final
from data import coco_final
from utils import visutil
from utils import bird_vis
from utils import image as image_utils
from nnutils import train_utils
from nnutils import loss_utils
from nnutils import mesh_net
import os
from nnutils.nmr import NeuralRenderer, OF_NeuralRenderer
from torch.utils.data import DataLoader
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops import SubdivideMeshes
import pickle as pkl
from data import tigdog_mf_of as tigdog_mf
from data import youtube_mf_of as yt_mf
import torch.nn.functional as F
from pytorch3d.transforms import *
import pytorch3d
from data.optical_flow import config_folder as cf
from data.optical_flow.model import MaskFlownet, MaskFlownet_S, Upsample
import yaml
from nnutils.geom_utils import mesh_laplacian
from data import objects as objects_data


flags.DEFINE_string('tmp_dir', 'tmp/', 'tmp dir to extract dataset')
flags.DEFINE_string('category', 'horse', 'category')
flags.DEFINE_string('root_dir', '/media/filippos/MyFiles/TigDog_new_wnrsfm/', 'TigDog root dir')
flags.DEFINE_string('root_dir_yt', '/media/filippos/MyFiles/TigDog_new_wnrsfm/', 'YTVis root dir')
flags.DEFINE_string('root_dir_coco', '/media/filippos/MyFiles/TigDog_new_wnrsfm/', 'COCO root dir')
flags.DEFINE_boolean('expand_ytvis', False, '')
flags.DEFINE_boolean('expand_pascal', False, '')
flags.DEFINE_string('mesh_dir', 'meshes/horse_new.obj', 'location of template mesh')
flags.DEFINE_string('kp_dict', 'meshes/horse_kp_dictionary.pkl', "location of template's keypoint annotation ")
flags.DEFINE_integer('num_lbs', 15, 'number of handles')
flags.DEFINE_integer('num_kps', 15, 'number of keypoints')
flags.DEFINE_integer('num_training_frames', 50, 'number of training frames per video')
flags.DEFINE_integer('img_size', 256, ' Image size')
flags.DEFINE_integer('max_length', 5, '')
flags.DEFINE_integer('num_frames', 2, 'number of frames')
flags.DEFINE_integer('num_guesses', 8, 'number of guesses')
flags.DEFINE_float('kp_loss_wt', 0., 'keypoint loss weight')
flags.DEFINE_float('of_loss_wt', 1., 'optical flow loss weight')
flags.DEFINE_float('mask_loss_wt', 1., 'mask loss weight')
flags.DEFINE_float('rigid_wt', 0.5, 'rigid loss weight')
flags.DEFINE_float('cam_loss_wt', 2., 'weights to camera loss')
flags.DEFINE_float('deform_loss_wt', 2., 'weights to deformation loss')
flags.DEFINE_float('deform_reg_wt', 1, 'reg to deformation')
flags.DEFINE_float('handle_deform_reg_wt', 0., 'reg to deformation')
flags.DEFINE_float('boundaries_reg_wt', 1., 'reg to boundaries')
flags.DEFINE_float('edt_reg_wt', 0.1, 'weight of chamfer loss')
flags.DEFINE_float('bdt_reg_wt', 2., 'weight of boundaries loss')
flags.DEFINE_float('entropy_loss_wt', 2., 'reg to entropy loss')
flags.DEFINE_float('triangle_reg_wt', 0.1, 'weights to triangle smoothness prior')
flags.DEFINE_float('tex_loss_wt', .5, 'weights to tex loss')
flags.DEFINE_float('tex_dt_loss_wt', .5, 'weights to tex dt loss')
flags.DEFINE_boolean('use_gtpose', True, 'if true uses gt pose for projection, but camera still gets trained.')
flags.DEFINE_boolean('symmetrize', False, '')
flags.DEFINE_string('camera_save_path', '', 'tmp dir to extract dataset')
flags.DEFINE_float('scale_lr_decay', 0.05, 'lr of scale branch')
flags.DEFINE_float('scale_bias', 1.0, 'scale bias')
flags.DEFINE_boolean('az_el_cam', False, 'use az_el camera')
flags.DEFINE_float('az_euler_range', 30, 'Warmup the part transform')
flags.DEFINE_float('el_euler_range', 60, 'Warmup the part transform')
flags.DEFINE_float('cyc_euler_range', 60, 'Warmup the pose predictor')
flags.DEFINE_float('rot_reg_loss_wt', 0.01, 'Rotation Reg loss wt.')
flags.DEFINE_boolean('optimize_deform', False, 'optimize deform')
flags.DEFINE_float('optimize_deform_lr', 100, 'lr of optimization deform')
flags.DEFINE_boolean('scale_mesh', False, 'whether to scale template mesh')

opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'misc', 'cachedir')


def mirror_sample(img, sfm_pose, mask_pred, mask):
    # Mirror sample with respect to axis
    img_flip = torch.flip(img, dims=(3,))
    mask_pred_flip = torch.flip(mask_pred, dims=(2,))
    mask_flip = torch.flip(mask, dims=(2,))
    quat = sfm_pose[:, -4:]
    quat = standardize_quaternion(quat)
    diag = torch.diag(torch.tensor([-1., 1., -1.], device=sfm_pose.device))[None]
    quat_new = quaternion_multiply(matrix_to_quaternion(diag), quat)
    scale = sfm_pose[:, :1]
    tx = - sfm_pose[:, 1:2]
    ty = sfm_pose[:, 2:3]
    sfm_pose = torch.cat([scale, tx, ty, quat_new], dim=-1)
    return img_flip, sfm_pose, mask_pred_flip, mask_flip


def mirror_cameras(sfm_pose, img_shape, mirror_flag):
    # Mirror camera with respect to axis
    quat = sfm_pose[:, -4:]
    quat = standardize_quaternion(quat)
    diag = torch.diag(torch.tensor([-1., 1., -1.], device=sfm_pose.device))[None]
    quat_new = quaternion_multiply(matrix_to_quaternion(diag), quat)

    scale = sfm_pose[:, :1]
    tx = - sfm_pose[:, 1:2]
    ty = sfm_pose[:, 2:3]
    sfm_pose_new = torch.cat([scale, tx, ty, quat_new], dim=-1)
    sfm_pose_new = (1 - mirror_flag.float()) * sfm_pose + sfm_pose_new * mirror_flag.float()
    return sfm_pose_new


def transform_cameras(sfm_pose, im_shape, transforms):
    # Affine transformation of cameras
    transforms_flag = transforms[..., -1].unsqueeze(-1)
    quat = sfm_pose[:, -4:]
    scale = sfm_pose[:, :1] * transforms[..., :1]
    tx = sfm_pose[:, 1:2] * transforms[..., :1] + transforms[..., 1:2]
    ty = sfm_pose[:, 2:3] * transforms[..., :1] + transforms[..., 2:3]
    sfm_pose_new = torch.cat([scale, tx, ty, quat], dim=-1)

    sfm_pose_new = (1 - transforms_flag.float()) * sfm_pose + sfm_pose_new * transforms_flag.float()
    return sfm_pose_new



class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        # ----------
        # Options
        # ----------
        self.symmetric = opts.symmetric
        if opts.category in ['horse', 'tiger']:
            anno_sfm_path = 'data/sfm_inits/' + opts.category + '/'
            sfm_mean_shape = torch.load(anno_sfm_path + 'sfm.pth')
        else:
            sfm_mean_shape = None

        kp_dict = None
        if opts.kp_loss_wt > 0:
            kp_dict = pkl.load(open(opts.kp_dict, 'rb'))
        mesh_horse = pytorch3d.io.load_obj(opts.mesh_dir)
        v, f = mesh_horse[0].numpy(), mesh_horse[1].verts_idx.numpy()
        if opts.scale_mesh:
            scale = 2. / torch.max(torch.nn.functional.pdist(torch.from_numpy(v))).numpy()
            v = v * scale
            v = v - v.mean(0)

        shapenet_mesh = [v, f]
        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat, num_kps=opts.num_kps, sfm_mean_shape=sfm_mean_shape,
            cam_embeddings=self.training_samples, shapenet_mesh=shapenet_mesh, no_kps=True, kp_dict=kp_dict,
            az_el_cam=opts.az_el_cam, deform_embeddings=self.training_samples)
        self.model = torch.nn.DataParallel(self.model).cuda()

        if opts.num_pretrain_epochs > 0 and not opts.load_warmup:
            print('Loading pre-trained network')
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)
        if opts.load_warmup:
            self.load_warmup_network(self.model, 'pred', opts.num_pretrain_epochs)

        # For renderering.
        faces = self.model.module.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        self.renderer = NeuralRenderer(opts.img_size)
        self.renderer = torch.nn.DataParallel(self.renderer).cuda()
        self.of_renderer = OF_NeuralRenderer(opts.img_size).cuda()

        self.renderer_predcam = NeuralRenderer(opts.img_size)  # for camera loss via projection
        self.renderer_predcam = torch.nn.DataParallel(self.renderer_predcam).cuda()

        # Need separate NMR for each fwd/bwd call.
        if opts.texture:
            self.tex_renderer = NeuralRenderer(opts.img_size)
            self.tex_renderer = torch.nn.DataParallel(self.tex_renderer).cuda()

        mesh_template = Meshes(verts=[self.model.module.get_mean_shape()], faces=[self.faces[0]])
        self.sdivide = SubdivideMeshes(mesh_template)

        # For visualization
        self.faces_up = self.sdivide(mesh_template).faces_packed()
        num_verts = mesh_template.verts_packed().shape[1]
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, num_verts, self.faces[:1].data.cpu().numpy())

        # Load Optical Flow Network Network
        checkpoint = '5adNov03-0005_1000000.pth'
        config_yaml = 'sintel.yaml'
        config_model_yaml = 'MaskFlownet.yaml'
        with open(os.path.join('data', 'optical_flow', 'config_folder', config_yaml)) as f:
            config = cf.Reader(yaml.load(f))
        with open(os.path.join('data', 'optical_flow', 'config_folder', config_model_yaml)) as f:
            config_model = cf.Reader(yaml.load(f))
        self.of_net = eval(config_model.value['network']['class'])(config)
        checkpoint = torch.load(os.path.join('data', 'optical_flow', 'weights', checkpoint))
        self.of_net.load_state_dict(checkpoint)
        self.of_net = torch.nn.DataParallel(self.of_net).cuda()

    def init_dataset(self):
        opts = self.opts
        if opts.category in ['horse', 'tiger']:
            self.dataset1 = tf_final.TigDogDataset_Final(opts.root_dir, opts.category, transforms=None, normalize=False,
                                                         max_length=None, remove_neck_kp=False, split='train',
                                                         img_size=256, mirror=False, scale=False, crop=False)
            if opts.category in ['horse', 'tiger'] and opts.expand_ytvis:
                self.dataset2 = yt_final.YTVIS_final(opts.root_dir_yt, opts.category, transforms=None, normalize=False,
                                                     max_length=None, split='all', img_size=256,
                                                     mirror=False, scale=False, crop=False, num_kps=opts.num_kps)
                self.dataset3 = coco_final.COCO_final(opts.root_dir_coco, opts.category, transforms=None, normalize=False,
                                                     max_length=None, split='all', img_size=256,
                                                     mirror=False, scale=False, crop=False, num_kps=opts.num_kps)
                self.dataset = torch.utils.data.ConcatDataset([self.dataset1, self.dataset2, self.dataset3])
            else:
                self.dataset = self.dataset1
            self.collate_fn = tf_final.TigDog_collate
        if opts.category in ['cow', 'giraffe', 'elephant', 'fox', 'zebra', 'leopard', 'bear']:
            self.dataset1 = yt_final.YTVIS_final(opts.root_dir_yt, opts.category, transforms=None, normalize=False,
                                                 max_length=None, split='all', img_size=256,
                                                 mirror=False, scale=False, crop=False, num_kps=opts.num_kps)
            if opts.category == 'cow' and opts.expand_pascal:
                self.dataset2 = objects_data.imnet_pascal_quad_data_loader_v2(opts, pascal_only=True)
                self.dataset3 = coco_final.COCO_final(opts.root_dir_coco, opts.category, transforms=None, normalize=False,
                                                     max_length=None, split='all', img_size=256,
                                                     mirror=False, scale=False, crop=False, num_kps=opts.num_kps)
                self.dataset = torch.utils.data.ConcatDataset([self.dataset1, self.dataset2, self.dataset3])
            else:
                self.dataset = self.dataset1

        directory = opts.tmp_dir + '/' + opts.category + '/'
        if not osp.exists(directory):
            os.makedirs(directory)

        save_counter = 0
        sample_to_vid = {}
        samples_per_vid = {}
        print('Number of videos for ', opts.category, '-', len(self.dataset))
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
                if i >= opts.num_training_frames:
                    break

        self.training_samples = save_counter
        print('Training samples (frames):', self.training_samples)
        if opts.category in ['horse', 'tiger']:
            self.dataset = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir, opts.category, num_frames=opts.num_frames,
                                                              sample_to_vid=sample_to_vid,
                                                              samples_per_vid=samples_per_vid,
                                                              normalize=True, transforms=True,
                                                              remove_neck_kp=True, split='train', img_size=256,
                                                              mirror=True, scale=True, crop=True,
                                                              tight_bboxes=False, v2_crop=False)
            self.dataset_noag = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir, opts.category,
                                                                   num_frames=opts.num_frames,
                                                                   sample_to_vid=sample_to_vid,
                                                                   samples_per_vid=samples_per_vid,
                                                                   normalize=True, transforms=None,
                                                                   remove_neck_kp=True, split='train', img_size=256,
                                                                   mirror=False, scale=True, crop=True, padding_frac=0.,
                                                                   tight_bboxes=False, v2_crop=False)
            self.collate_fn = tigdog_mf.TigDog_collate
        if opts.category in ['cow', 'giraffe', 'elephant', 'fox', 'zebra', 'leopard', 'bear']:
            self.dataset = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir, opts.category, num_frames=opts.num_frames,
                                                              sample_to_vid=sample_to_vid,
                                                              samples_per_vid=samples_per_vid,
                                                              normalize=True, transforms=True,
                                                              remove_neck_kp=False, split='train', img_size=256,
                                                              mirror=True, scale=True, crop=True,
                                                              padding_frac=opts.padding_frac, tight_bboxes=True,
                                                              v2_crop=True)
            self.dataset_noag = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir, opts.category, num_frames=opts.num_frames,
                                                              sample_to_vid=sample_to_vid,
                                                              samples_per_vid=samples_per_vid,
                                                              normalize=True, transforms=None,
                                                              remove_neck_kp=False, split='train', img_size=256,
                                                              mirror=True, scale=True, crop=True,
                                                              padding_frac=opts.padding_frac, tight_bboxes=True,
                                                              v2_crop=True)
            self.collate_fn = tigdog_mf.TigDog_collate

        self.dataloader = DataLoader(self.dataset, opts.batch_size, drop_last=True, shuffle=True,
                                     collate_fn=self.collate_fn, num_workers=2)
        self.dataloader_noag = DataLoader(self.dataset_noag, opts.batch_size,
                                          drop_last=False, shuffle=False, collate_fn=self.collate_fn)
        print('Dataloader:', len(self.dataloader))

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def define_criterion(self):
        self.projection_loss = loss_utils.kp_l2_loss
        self.optical_flow_loss = loss_utils.Optical_Flow_Loss()
        self.mask_loss_fn = loss_utils.l1_loss
        self.entropy_loss = loss_utils.entropy_loss
        self.camera_loss_fn = loss_utils.camera_loss
        self.boundaries_fn = loss_utils.Boundaries_Loss()
        self.boundaries_fn = torch.nn.DataParallel(self.boundaries_fn)
        self.dt_fn = loss_utils.texture_dt_loss_v
        self.edt_fn = loss_utils.edt_loss
        self.locally_rigid_fn = loss_utils.Locally_Rigid()
        self.handle_deform_fn = loss_utils.deform_l2reg
        self.cross_entropy_fn = torch.nn.CrossEntropyLoss()

        if self.opts.texture:
            self.texture_loss = loss_utils.PerceptualTextureLoss_v2()
            self.texture_dt_loss_fn = loss_utils.texture_dt_loss_v

    def set_input(self, batch):
        opts = self.opts
        batch_size, t = batch['img'].shape[:2]
        # Image with annotations.
        input_img_tensor = batch['img'].type(torch.FloatTensor).clone()
        input_img_tensor = input_img_tensor.reshape(batch_size * t, *input_img_tensor.shape[2:])
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        img_tensor = batch['img'].type(torch.FloatTensor)

        img_tensor = img_tensor.reshape(batch_size * t, *img_tensor.shape[2:])

        mask_tensor = batch['mask'].type(torch.FloatTensor)
        mask_tensor = mask_tensor.reshape(batch_size * t, *mask_tensor.shape[2:])

        kp_tensor = batch['kp'].type(torch.FloatTensor)
        kp_tensor = kp_tensor.reshape(batch_size * t, *kp_tensor.shape[2:])

        cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)
        cam_tensor = cam_tensor.reshape(batch_size * t, *cam_tensor.shape[2:])

        self.input_imgs = input_img_tensor.cuda(device=opts.gpu_id)
        self.imgs = img_tensor.cuda(device=opts.gpu_id)
        self.masks = mask_tensor.cuda(device=opts.gpu_id)
        self.kps = kp_tensor.cuda(device=opts.gpu_id)
        self.cams = cam_tensor.cuda(device=opts.gpu_id)

        # Compute barrier distance transform.
        mask_dt = batch['mask'].clone()
        mask_dt = mask_dt.reshape(batch_size * t, *mask_dt.shape[2:])

        mask_edt = np.stack([image_utils.compute_dt(m, norm=False) for m in mask_dt])
        mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in mask_dt])
        dt_tensor = torch.tensor(mask_dts).float().cuda(device=opts.gpu_id)
        edt_tensor = torch.tensor(mask_edt).float().cuda(device=opts.gpu_id)
        # B x 1 x N x N
        self.dts_barrier = dt_tensor.unsqueeze(1)
        self.edts_barrier = edt_tensor.unsqueeze(1)

        self.boundaries = image_utils.compute_boundaries(self.masks.cpu().numpy())
        self.boundaries = torch.tensor(self.boundaries).float().cuda(device=opts.gpu_id)

        self.frames_idx = batch['frames_idx'].long().cuda()
        self.mirror_flag = batch['mirror_flag'].reshape(batch_size * t).long().cuda()

        self.cam_loss = torch.zeros(1)
        self.transforms = batch['transforms'].reshape(batch_size * t, -1).float().cuda()

        of_imgages = batch['img'].type(torch.FloatTensor).cuda()
        with torch.no_grad():
            im0 = of_imgages[:, 0]
            im1 = of_imgages[:, 1]
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
            pred, flows, warpeds = self.of_net(im0_c, im1_c)

            up_flow = Upsample(pred[-1], 4)
            if pad_h != 0 or pad_w != 0:
                up_flow = F.interpolate(up_flow, size=[shape[2], shape[3]], mode='bilinear') * \
                          torch.tensor([shape[d] / up_flow.shape[d] for d in (2, 3)], device=im0.device).view(1, 2,
                                                                                                              1, 1)
            up_flow = F.interpolate(up_flow, size=[self.imgs.shape[-2], self.imgs.shape[-1]], mode='bilinear')
        self.optical_flows = up_flow.permute(0, 2, 3, 1)
        self.optical_flows = self.optical_flows[:, None].repeat(1, 2, 1, 1, 1)
        self.optical_flows[:, 1::2] = 0

    def centralize(self, img1, img2):
        rgb_mean = torch.cat((img1, img2), 2)
        rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
        rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    def init_camera_emb(self):
        self.cams = transform_cameras(self.cams, self.input_imgs.shape, self.transforms)
        self.cams_rescaled = self.cams.clone()
        self.cams_rescaled[:, 0] = (self.cams_rescaled[:, 0].abs() - 1) / opts.scale_lr_decay
        cameras = self.model.module.cameras[0].weight.data
        cameras[self.frames_idx] = self.cams_rescaled.data.reshape(-1, opts.num_frames, 7)

        cameras = [cam_emb(self.frames_idx) for cam_emb in self.model.module.cameras]
        cameras = torch.stack(cameras)
        cameras = cameras.reshape(opts.num_guesses, -1, 7)
        quats = cameras[..., 3:]
        scales = F.relu(opts.scale_lr_decay * cameras[..., :1] + 1) + 1e-12
        translations = cameras[..., 1:3]
        # quats_n = quats
        quats_n = torch.nn.functional.normalize(quats, dim=-1)
        self.cam_pred = torch.cat([scales, translations, quats_n], dim=2)
        self.cam_pred = self.cam_pred.reshape(opts.num_guesses * self.cam_pred.shape[1], -1)
        self.cam_pred = self.cam_pred.squeeze(0)

    def warmup(self):
        opts = self.opts
        cameras = [cam_emb(self.frames_idx) for cam_emb in self.model.module.cameras]
        cameras = torch.stack(cameras)
        if opts.az_el_cam:
            cameras = cameras.reshape(opts.num_guesses, -1, 6)
            angles = cameras[..., 3:]
            scales = cameras[..., :1]
            translations = cameras[..., 1:3]
            cameras = self.model.module.multicampredictor(scales, translations, angles)
            scales = cameras[..., :1]
            translations = cameras[..., 1:3]
            quats_n = cameras[..., 3:]
        else:
            cameras = cameras.reshape(opts.num_guesses, -1, 7)
            quats = cameras[..., 3:]
            scales = F.relu(opts.scale_lr_decay * cameras[..., :1] + 1) + 1e-12
            translations = cameras[..., 1:3]
            quats_n = torch.nn.functional.normalize(quats, dim=-1)
        self.cam_pred = torch.cat([scales, translations, quats_n], dim=2)
        self.cam_pred = self.cam_pred.reshape(opts.num_guesses * self.cam_pred.shape[1], -1)
        self.cam_pred = mirror_cameras(self.cam_pred, self.input_imgs.shape,
                                       self.mirror_flag.repeat(opts.num_guesses)[:, None])
        self.cam_pred = transform_cameras(self.cam_pred, self.input_imgs.shape,
                                          self.transforms.repeat(opts.num_guesses, 1))
        self.mean_shape = self.model.module.get_mean_shape()
        self.mean_v = self.mean_shape[None].repeat(self.masks.shape[0], 1, 1)
        self.pred_v = self.mean_v

        proj_cam = self.cam_pred

        faces = self.faces.repeat(opts.num_frames, 1, 1)
        self.mask_pred, pix_to_face = self.renderer(self.mean_v.repeat(opts.num_guesses, 1, 1),
                                                    faces.repeat(opts.num_guesses, 1, 1), proj_cam)
        self.mask_loss = self.mask_loss_fn(self.mask_pred, self.masks.repeat(opts.num_guesses, 1, 1), reduce=False)
        self.mask_loss = self.mask_loss.reshape(opts.num_guesses, self.mean_v.shape[0])
        pred_proj = self.renderer.module.project_points(self.mean_v.repeat(opts.num_guesses, 1, 1), proj_cam)
        self.edt_loss = self.edt_fn(self.mask_pred, self.edts_barrier.repeat(opts.num_guesses, 1, 1, 1), reduce=False)
        self.bdt_loss = self.boundaries_fn(pred_proj, self.boundaries.repeat(opts.num_guesses, 1, 1),
                                           faces.repeat(opts.num_guesses, 1, 1), pix_to_face, reduce=False)
        self.sil_cons = opts.edt_reg_wt * self.edt_loss + opts.bdt_reg_wt * self.bdt_loss
        self.sil_cons = self.sil_cons.reshape(opts.num_guesses, self.mean_v.shape[0])

        if opts.of_loss_wt > 0:
            masks_of = self.masks.reshape(opts.batch_size, opts.num_frames, self.masks.shape[1], self.masks.shape[2])
            pred_v_of = self.mean_v.reshape(opts.batch_size, opts.num_frames, self.mean_v.shape[1],
                                            self.pred_v.shape[2])
            proj_cam_of = proj_cam
            faces_of = self.faces[:, None].repeat(1, opts.num_frames, 1, 1)
            optical_flows_f = torch.flip(self.optical_flows, dims=[1]) * masks_of[:, :, :, :, None]
            pred_v_of = pred_v_of.repeat(opts.num_guesses, 1, 1, 1)
            faces_of = faces_of.repeat(opts.num_guesses, 1, 1, 1)
            optical_flows_f = optical_flows_f.repeat(opts.num_guesses, 1, 1, 1, 1)
            self.of_loss, self.of_pred_cp, self.visible_vertices, self.verts_of, self.samples_ofs_gt = loss_utils.optical_flow_loss(
                pred_v_of,
                faces_of,
                proj_cam_of,
                optical_flows_f,
                self.of_renderer, pix_to_face=None,
                reduce=False)
            self.of_loss = self.of_loss.repeat(1, self.optical_flows.shape[1])
            self.of_loss = self.of_loss.reshape(opts.num_guesses, -1)
        else:
            self.of_loss = torch.zeros(1, device=self.masks.device)

        if opts.kp_loss_wt > 0.:
            vert2kp = torch.nn.functional.softmax(self.model.module.vert2kp, dim=1)
            self.kp_verts_pred_v = torch.matmul(vert2kp, self.pred_v)
            self.kp_pred_transformed = self.renderer.module.project_points(
                self.kp_verts_pred_v.repeat(opts.num_guesses, 1, 1),
                proj_cam)
            self.kp_loss = self.projection_loss(self.kp_pred_transformed, self.kps.repeat(opts.num_guesses, 1, 1),
                                                reduction='none')
            self.kp_loss = self.kp_loss.reshape(opts.num_guesses, self.mean_v.shape[0])

        total_loss = opts.mask_loss_wt * self.mask_loss + opts.of_loss_wt * self.of_loss + \
                     opts.boundaries_reg_wt * self.sil_cons
        if opts.kp_loss_wt > 0.:
            total_loss += opts.kp_loss_wt * self.kp_loss
        self.probs = torch.softmax(-total_loss, dim=0).detach()
        probs_weights_ = self.model.module.prob_embeddings.weight.data
        probs_weights_[self.frames_idx] = self.probs.reshape(opts.num_guesses, *self.frames_idx.shape).permute(1, 2, 0)
        self.total_loss = total_loss.mean()


    def forward(self, detach_camera=False, drop_deform=False):
        opts = self.opts

        img_feat, self.res_feats, pred_codes = self.model(self.input_imgs)
        self.delta_v_res, _, _, _ = pred_codes
        self.mean_shape = self.model.module.get_mean_shape()
        # Compute keypoints.

        if opts.optimize_deform:
            deforms = self.model.module.deform_emb(self.frames_idx)
            deforms = deforms.reshape(-1, opts.num_lbs, 3)

            deforms_mirror = self.model.module.deform_mirror_emb(self.frames_idx)
            deforms_mirror = deforms_mirror.reshape(-1, opts.num_lbs, 3)
            deforms = (1 - self.mirror_flag[:, None, None].float()) * deforms + deforms_mirror * self.mirror_flag[:,
                                                                                                 None, None].float()
            deforms = deforms * opts.optimize_deform_lr

        if opts.drop_hypothesis:
            probs_weights = self.model.module.prob_embeddings.weight.data
            selected_cameras_probs = \
                probs_weights[self.frames_idx].topk(opts.num_guesses, largest=True, dim=-1, sorted=True)[1]
            selected_cameras_probs = selected_cameras_probs.permute(2, 0, 1)
            selected_cameras = selected_cameras_probs[..., None].repeat(1, 1, 1, 7)
            selected_deforms = selected_cameras_probs.reshape(opts.num_guesses, -1)[..., None, None]
            selected_deforms = selected_deforms.repeat(1, 1, opts.num_lbs, 3)


        cameras = [cam_emb(self.frames_idx) for cam_emb in self.model.module.cameras]
        cameras = torch.stack(cameras)

        if opts.az_el_cam:
            cameras = cameras.reshape(opts.num_guesses, -1, 6)
            angles = cameras[..., 3:]
            scales = cameras[..., :1]
            translations = cameras[..., 1:3]
            cameras = self.model.module.multicampredictor(scales, translations, angles)
            cameras = cameras.reshape(-1, opts.batch_size, opts.num_frames, 7)
            if opts.drop_hypothesis:
                cameras = torch.gather(cameras, 0, selected_cameras)
            cameras = cameras.reshape(opts.num_guesses, -1, 7)
            scales = cameras[..., :1]
            translations = cameras[..., 1:3]
            quats_n = cameras[..., 3:]

        else:
            if opts.drop_hypothesis:
                cameras = torch.gather(cameras, 0, selected_cameras)

            cameras = cameras.reshape(opts.num_guesses, -1, 7)
            quats = cameras[..., 3:]
            scales = F.relu(opts.scale_lr_decay * cameras[..., :1] + 1) + 1e-12
            translations = cameras[..., 1:3]
            quats_n = torch.nn.functional.normalize(quats, dim=-1)
        self.cam_pred = torch.cat([scales, translations, quats_n], dim=2)
        self.cam_pred = self.cam_pred.reshape(opts.num_guesses * self.cam_pred.shape[1], -1)
        self.cam_pred = mirror_cameras(self.cam_pred, self.input_imgs.shape,
                                       self.mirror_flag.repeat(opts.num_guesses)[:, None])
        self.cam_pred = transform_cameras(self.cam_pred, self.input_imgs.shape,
                                          self.transforms.repeat(opts.num_guesses, 1))
        if detach_camera:
            self.cam_pred = self.cam_pred.detach()

        self.lbs = self.model.module.get_lbs().permute(1, 0)
        self.lbs = self.lbs[None].repeat(self.delta_v_res.shape[0], 1, 1)
        self.mean_v = self.mean_shape[None].repeat(self.delta_v_res.shape[0], 1, 1)
        self.delta_v_ms = self.lbs.bmm(self.mean_v)

        if drop_deform:
            self.delta_v = self.delta_v_ms
        else:
            if opts.optimize_deform:
                self.delta_v = self.delta_v_ms + deforms
            else:
                self.delta_v = self.delta_v_ms + self.delta_v_res

        # Deform mean shape:
        src_mesh = Meshes(verts=self.mean_shape[None], faces=self.faces[:1])
        L = mesh_laplacian(src_mesh, 'cot')
        L = L.repeat(self.delta_v.shape[0], 1, 1)
        delta = torch.bmm(L, self.mean_v)
        A = self.lbs
        A_augm = L.permute(0, 2, 1).matmul(L) + A.permute(0, 2, 1).matmul(A)
        b = L.permute(0, 2, 1) @ delta + A.permute(0, 2, 1) @ self.delta_v
        u = torch.cholesky(A_augm)
        self.pred_v = torch.cholesky_solve(b, u)
        self.pred_v = self.pred_v.repeat(opts.num_guesses, 1, 1)
        # Decide which camera to use for projection.
        if opts.use_gtpose:
            proj_cam = self.cams
        else:
            proj_cam = self.cam_pred


        self.handle_deform = self.handle_deform_fn(self.delta_v_res)
        faces = self.faces.repeat(opts.num_frames, 1, 1)
        if opts.texture:
            self.textures_colors = self.model.module.texture_predictor.forward(self.pred_v, self.res_feats)
            self.textures = self.textures_colors.repeat(opts.num_guesses, 1, 1, 1, 1)

            self.mask_pred, pix_to_face = self.renderer(self.pred_v,
                                                        faces.repeat(opts.num_guesses, 1, 1),
                                                        proj_cam)

            self.texture_pred, _, _ = self.tex_renderer(self.pred_v.detach(),
                                                        faces.repeat(opts.num_guesses, 1, 1), proj_cam,
                                                        textures=self.textures)
            self.imgs_flip, proj_cam_flip, self.mask_pred_flip, self.masks_flip = mirror_sample(self.imgs,
                                                                                                proj_cam,
                                                                                                self.mask_pred,
                                                                                                self.masks)
            self.texture_pred_flip, _, _ = self.tex_renderer(self.pred_v.detach(),
                                                             faces.repeat(opts.num_guesses, 1, 1),
                                                             proj_cam_flip, textures=self.textures)

        else:
            self.textures = None
            self.mask_pred, pix_to_face = self.renderer(self.pred_v,
                                                        faces.repeat(opts.num_guesses, 1, 1),
                                                        proj_cam)

        self.mask_loss = self.mask_loss_fn(self.mask_pred, self.masks.repeat(opts.num_guesses, 1, 1), reduce=False)
        self.mask_loss = self.mask_loss.reshape(opts.num_guesses, opts.batch_size * opts.num_frames)

        if opts.texture:
            self.tex_loss = 0.5 * self.texture_loss(self.texture_pred, self.imgs.repeat(opts.num_guesses, 1, 1, 1),
                                                    self.mask_pred, self.masks.repeat(opts.num_guesses, 1, 1),
                                                    reduce=False) + \
                            0.5 * self.texture_loss(self.texture_pred_flip,
                                                    self.imgs_flip.repeat(opts.num_guesses, 1, 1, 1),
                                                    self.mask_pred_flip, self.masks_flip.repeat(opts.num_guesses, 1, 1),
                                                    reduce=False)
            tex_l1 = 0.5 * (
                    F.mse_loss(self.texture_pred * self.masks.repeat(opts.num_guesses, 1, 1).unsqueeze(1),
                               (self.imgs * self.masks.unsqueeze(1)).repeat(opts.num_guesses, 1, 1, 1),
                               reduction='none') + F.mse_loss(
                self.texture_pred_flip * self.masks_flip.unsqueeze(1).repeat(opts.num_guesses, 1, 1, 1),
                (self.imgs_flip * self.masks_flip.unsqueeze(1)).repeat(opts.num_guesses, 1, 1, 1), reduction='none'))
            tex_l1 = tex_l1.mean((1, 2, 3))
            self.tex_loss += tex_l1
            self.tex_loss = self.tex_loss.reshape(opts.num_guesses, opts.batch_size * opts.num_frames)


        if opts.of_loss_wt > 0:
            masks_of = self.masks.reshape(opts.batch_size, opts.num_frames, self.masks.shape[1], self.masks.shape[2])
            pred_v_of = self.pred_v.reshape(opts.num_guesses * opts.batch_size, opts.num_frames, self.pred_v.shape[1],
                                            self.pred_v.shape[2])
            proj_cam_of = proj_cam
            faces_of = self.faces[:, None].repeat(1, opts.num_frames, 1, 1)
            optical_flows_f = torch.flip(self.optical_flows, dims=[1]) * masks_of[:, :, :, :, None]
            # pred_v_of = pred_v_of.repeat(opts.num_guesses, 1, 1, 1)
            faces_of = faces_of.repeat(opts.num_guesses, 1, 1, 1)
            optical_flows_f = optical_flows_f.repeat(opts.num_guesses, 1, 1, 1, 1)
            self.of_loss, self.of_pred_cp, self.visible_vertices, self.verts_of, self.samples_ofs_gt = loss_utils.optical_flow_loss(
                pred_v_of,
                faces_of,
                proj_cam_of,
                optical_flows_f,
                self.of_renderer, pix_to_face=None,
                reduce=False)

            self.of_loss = self.of_loss.reshape(opts.num_guesses, -1)
            self.of_loss = self.of_loss.repeat(1, self.optical_flows.shape[1])
            self.of_loss = self.of_loss.reshape(opts.num_guesses, -1)
        else:
            self.of_loss = torch.zeros(1, device=self.masks.device)

        if opts.kp_loss_wt > 0.:
            vert2kp = torch.nn.functional.softmax(self.model.module.vert2kp, dim=1)
            self.kp_verts_pred_v = torch.matmul(vert2kp, self.pred_v)
            self.kp_pred_transformed = self.renderer.module.project_points(self.kp_verts_pred_v, proj_cam)
            self.kp_loss = self.projection_loss(self.kp_pred_transformed, self.kps.repeat(opts.num_guesses, 1, 1),
                                                reduction='none')
            self.kp_loss = self.kp_loss.reshape(opts.num_guesses, opts.batch_size * opts.num_frames)

        # Priors:
        mesh_3d = Meshes(verts=self.pred_v, faces=faces)
        mesh_template = Meshes(verts=self.mean_v, faces=faces)

        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(mesh_3d, method="cot")
        self.triangle_loss = loss_laplacian

        if opts.texture:

            t_c = self.textures_colors
            t_c = t_c.reshape(-1, opts.num_frames, *self.textures.shape[1:]).permute(0, 2, 3, 4, 1, 5)
            t_c = t_c.reshape(-1, t_c.shape[2], t_c.shape[3])
            self.cycle_loss = torch.norm(t_c[:, :-1] - t_c[:, 1:], p=2, dim=-1)
            self.cycle_loss = self.cycle_loss.mean()

        self.rigid_loss = self.locally_rigid_fn(mesh_3d, mesh_template).mean()
        pred_proj = self.renderer.module.project_points(self.pred_v, proj_cam)
        self.edt_loss = self.edt_fn(self.mask_pred, self.edts_barrier.repeat(opts.num_guesses, 1, 1, 1), reduce=False)
        self.edt_loss = self.edt_loss.reshape(opts.num_guesses, opts.batch_size * opts.num_frames)
        self.bdt_loss = self.boundaries_fn(pred_proj, self.boundaries.repeat(opts.num_guesses, 1, 1),
                                           faces.repeat(opts.num_guesses, 1, 1), pix_to_face, reduce=False)
        self.bdt_loss = self.bdt_loss.reshape(opts.num_guesses, opts.batch_size * opts.num_frames)

        self.sil_cons = opts.edt_reg_wt * self.edt_loss + opts.bdt_reg_wt * self.bdt_loss

        # Finally sum up the loss.
        # Instance loss:
        self.total_loss = opts.mask_loss_wt * self.mask_loss
        self.total_loss += opts.of_loss_wt * self.of_loss
        self.total_loss += opts.boundaries_reg_wt * self.sil_cons
        if opts.kp_loss_wt > 0.:
            self.total_loss += opts.kp_loss_wt * self.kp_loss

        if opts.texture:
            self.total_loss += opts.tex_loss_wt * self.tex_loss

        self.camera_loss = self.total_loss.mean()
        self.probs = torch.softmax(-self.total_loss, dim=0).detach()
        if opts.drop_hypothesis:
            probs_weights_ = self.model.module.prob_embeddings.weight.data
            probs_ = self.probs.reshape(opts.num_guesses, -1, opts.num_frames)
            probs_weights_[self.frames_idx] = 0
            probs_weights_[self.frames_idx] = torch.scatter(probs_weights_[self.frames_idx].permute(2, 0, 1),
                                                            0, selected_cameras_probs, probs_).permute(1, 2, 0)

        total_loss = self.total_loss * self.probs
        total_loss = total_loss.sum(0).mean()
        self.total_loss = total_loss
        # Priors:
        self.total_loss += opts.rigid_wt * self.rigid_loss
        self.total_loss += opts.triangle_reg_wt * self.triangle_loss
        self.total_loss += opts.deform_reg_wt * self.cycle_loss
        self.total_loss += opts.handle_deform_reg_wt * self.handle_deform

        scale, trans, quat = self.model.module.camera_predictor(self.res_feats)
        self.predicted_camera = torch.cat([scale, trans, quat], 1)

        probs_weights = self.probs.reshape(opts.num_guesses, -1, opts.num_frames).permute(1, 2, 0)
        argmax_idx = probs_weights.argmax(dim=-1).reshape(-1)
        cam_tmp = self.cam_pred.reshape(-1, argmax_idx.shape[0], self.cam_pred.shape[-1])
        cam_sel = cam_tmp[argmax_idx, torch.arange(cam_tmp.shape[1], device=self.pred_v.device)]
        self.cam_loss = self.camera_loss_fn(self.predicted_camera, cam_sel.detach(), 0)

        self.total_loss += opts.cam_loss_wt * self.cam_loss
        self.deform_loss = F.mse_loss(self.delta_v_res.squeeze(1), deforms.detach())
        if opts.optimize_deform:
            self.total_loss += opts.deform_loss_wt * self.deform_loss

    def learn_camera_predictor(self):
        scale, trans, quat = self.model.module.camera_predictor(self.res_feats.detach())
        self.predicted_camera = torch.cat([scale, trans, quat], 1)
        cam_idx = self.probs.argmax(dim=0)
        cam_tmp = self.cam_pred.reshape(opts.num_guesses, -1, self.cam_pred.shape[-1])
        cam_sel = cam_tmp[cam_idx, torch.arange(cam_tmp.shape[1], device=scale.device)]
        self.cam_loss = self.camera_loss_fn(self.predicted_camera, cam_sel.detach(), 0)

    def get_current_visuals(self):
        with torch.no_grad():
            vis_dict = {}

            cam_idx = self.probs.argmax(dim=0)
            cam_tmp = self.cam_pred.reshape(opts.num_guesses, -1, self.cam_pred.shape[-1])
            cam_sel = cam_tmp[cam_idx, torch.arange(cam_tmp.shape[1])]
            pred_v = self.pred_v.reshape(opts.num_guesses, -1, *self.pred_v.shape[1:])
            pred_v = pred_v[cam_idx, torch.arange(pred_v.shape[1])]
            mean_v = self.mean_v
            delta_v = self.delta_v

            lbs = self.lbs
            texture_pred = self.texture_pred
            if opts.kp_loss_wt > 0:
                kp_pred_transformed = self.kp_pred_transformed.reshape(opts.num_guesses, -1,
                                                                       *self.kp_pred_transformed.shape[1:])
                kp_pred_transformed = kp_pred_transformed[cam_idx, torch.arange(kp_pred_transformed.shape[1])]

            faces = self.faces.repeat(opts.num_frames, 1, 1)
            mask_vis, _ = self.renderer(pred_v, faces, cam_sel)
            mask_pred_cam, _ = self.renderer(pred_v, faces, self.predicted_camera)
            mask_concat = torch.cat([self.masks, mask_vis, mask_pred_cam], 2)

            if self.opts.of_loss_wt > 0:
                import flowiz
                optical_flows = self.optical_flows
                b, t, h, w, _ = optical_flows.shape
                flows = optical_flows.reshape(b * t, h, w, 2)  # .permute(0, 3, 1, 2)
                color_flows = []
                for of in flows.cpu().numpy():
                    color_flow = flowiz.convert_from_flow(of)
                    color_flows.append(color_flow)
                color_flows = np.array(color_flows)

                masks_of = self.masks.reshape(opts.batch_size, opts.num_frames, self.masks.shape[1],
                                              self.masks.shape[2])
                pred_v_of = pred_v.reshape(opts.batch_size, opts.num_frames, pred_v.shape[1],
                                           pred_v.shape[2])
                proj_cam_of = cam_sel

                faces_of = faces.reshape(opts.batch_size, opts.num_frames, faces.shape[1], faces.shape[2])
                optical_flows_f = torch.flip(optical_flows, dims=[1]) * masks_of[:, :, :, :, None]

                of_loss, of_pred_cp, visible_vertices, verts_of, samples_ofs_gt = loss_utils.optical_flow_loss(
                    pred_v_of,
                    faces_of,
                    proj_cam_of,
                    optical_flows_f,
                    self.of_renderer, pix_to_face=None,
                    reduce=False)

                verts_ = verts_of.reshape(b * t, verts_of.shape[2], -1).clone()
                verts_ = self.masks.shape[-1] * (verts_ + 1) / 2

                samples_ofs_gt = samples_ofs_gt.reshape(samples_ofs_gt.shape[0] * samples_ofs_gt.shape[1],
                                                        samples_ofs_gt.shape[2], samples_ofs_gt.shape[3])
                of_pred_cp = of_pred_cp.reshape(of_pred_cp.shape[0] * of_pred_cp.shape[1],
                                                of_pred_cp.shape[2], of_pred_cp.shape[3])
                visible_vertices = visible_vertices.reshape(visible_vertices.shape[0] * visible_vertices.shape[1],
                                                            visible_vertices.shape[2])

            num_show = min(2, self.opts.batch_size)
            delta_v_ms = lbs.bmm(mean_v)
            delta_v_ms = self.renderer.module.project_points(delta_v_ms, cam_sel)
            delta_v_mv = self.renderer.module.project_points(delta_v, cam_sel)
            for i in range(num_show):
                if self.opts.of_loss_wt > 0:
                    if i % 2 == 0:
                        flow = color_flows[i]
                        samples_ofs_gt_ = samples_ofs_gt[i].detach().cpu().numpy()
                        colored_points = flowiz.convert_from_flow(samples_ofs_gt_[None])[0] / 255.
                        vv = (visible_vertices[i] > 0).cpu().numpy()
                        x_ = verts_[i + 1, :, 0].detach().cpu().numpy()
                        y_ = verts_[i + 1, :, 1].detach().cpu().numpy()
                        of_cp = of_pred_cp[i].detach().cpu().numpy()

                        fig = plt.figure(figsize=(6, 6))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.imshow(color_flows[i] / 255)
                        ax1.scatter(x=list(x_[vv]), y=list(y_[vv]), c=colored_points[vv].clip(0, 1))
                        ax1.axis('off')
                        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
                        fig.canvas.draw()
                        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        flow_gt = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        flow_gt = resize(flow_gt, self.masks.shape[-2:]) * 255
                        plt.close(fig)
                        fig = plt.figure(figsize=(6, 6))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.imshow(color_flows[i] / 255)
                        colored_points = flowiz.convert_from_flow(of_cp[None])[0] / 255.
                        ax1.scatter(x=x_[vv], y=y_[vv], c=colored_points[vv])
                        ax1.axis('off')
                        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
                        fig.canvas.draw()
                        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        flow_pred = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        flow_pred = resize(flow_pred, self.masks.shape[-2:]) * 255
                        plt.close(fig)

                input_img = self.imgs[i].data.permute(1, 2, 0).cpu().numpy() * 255

                input_img_ms = bird_vis.kp2im(delta_v_ms[i].data, self.imgs[i].data)
                input_img_mv = bird_vis.kp2im(delta_v_mv[i].data, self.imgs[i].data)
                input_img_bdt = bird_vis.kp2im(self.boundaries[i, :, :2].data, self.imgs[i].data)
                pred_proj = self.renderer.module.project_points(pred_v, cam_sel)
                input_img_bdt_pred = bird_vis.kp2im(pred_proj[i, :, :2].data, self.imgs[i].data)

                masks = bird_vis.tensor2mask(mask_concat[i].data)
                if opts.kp_loss_wt > 0:
                    pred_transformed_kp_img = bird_vis.kp2im(kp_pred_transformed[i].data, self.imgs[i].data)
                    input_img = bird_vis.kp2im(self.kps[i].data, self.imgs[i].data)

                if self.opts.texture:
                    input_img_texture = self.imgs[i].data.permute(1, 2, 0).cpu().numpy() * 255
                    texture_here = self.textures_colors[i]
                else:
                    texture_here = torch.ones_like(pred_v) * 0.8
                rend_predcam = self.vis_rend(pred_v[i], cam_sel[i], texture=texture_here)
                # Render from front & back:
                rend_frontal = self.vis_rend.diff_vp(pred_v[i], cam_sel[i], texture=texture_here)
                rend_top = self.vis_rend.diff_vp(pred_v[i], cam_sel[i], axis=[0, 1, 0], texture=texture_here)
                diff_rends = np.hstack((rend_frontal, rend_top))

                if self.opts.texture:
                    tex_img = bird_vis.tensor2im(texture_pred[i].data)
                    if opts.kp_loss_wt > 0:
                        imgs = np.hstack(
                            (input_img, pred_transformed_kp_img, tex_img, input_img_texture, input_img_bdt))
                    else:
                        imgs = np.hstack((input_img, tex_img, input_img_texture, input_img_bdt, input_img_bdt_pred))

                else:
                    imgs = input_img

                rends = np.hstack((diff_rends, rend_predcam))
                if self.opts.of_loss_wt > 0:
                    if i % 2 == 0:
                        vis_dict['%d' % i] = np.hstack(
                            (imgs, input_img_ms, input_img_mv, flow_gt, flow_pred, rends, masks))
                    else:
                        vis_dict['%d' % i] = np.hstack((imgs, input_img_ms, input_img_mv, rends, masks))
                else:
                    vis_dict['%d' % i] = np.hstack((imgs, rends, masks))

                vis_dict['masked_img %d' % i] = bird_vis.tensor2im((self.imgs[i] * self.masks[i]).data)

        return vis_dict

    def get_current_points(self):
        return {
            'mean_shape': visutil.tensor2verts(self.mean_shape.data),
            'verts': visutil.tensor2verts(self.pred_v.data),
        }

    def get_current_scalars(self):
        sc_dict = OrderedDict([
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss.item()),
            ('rigid_loss ', self.rigid_loss.item()),
            ('edt_loss', (self.probs * self.edt_loss).sum(0).mean().item()),
            ('bdt_loss', (self.probs * self.bdt_loss).sum(0).mean().item()),
            ('sil_cons', (self.probs * self.sil_cons).sum(0).mean().item()),
            ('mask_loss', (self.probs * self.mask_loss).sum(0).mean().item()),
            ('tri_loss', self.triangle_loss.item()),
            ('of_loss', (self.probs * self.of_loss).sum(0).mean().item()),
            ('camera_loss', self.cam_loss.item()),
            ('deform_loss', self.deform_loss.item()),
            ('handle_deform', self.handle_deform.item()),
        ])
        if self.opts.texture:
            sc_dict['tex_loss'] = (self.probs * self.tex_loss).sum(0).mean().item()
            sc_dict['cycle_loss'] = self.cycle_loss.mean().item()
        if self.opts.kp_loss_wt > 0:
            sc_dict['kp_loss'] = self.kp_loss.mean().item()

        return sc_dict


def main(_):
    torch.manual_seed(0)
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    trainer.train()


if __name__ == '__main__':
    app.run(main)
