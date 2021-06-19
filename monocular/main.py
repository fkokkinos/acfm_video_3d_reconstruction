from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

import os.path as osp
import numpy as np
import torch
import torchvision
import scipy.io as sio
from collections import OrderedDict

from data import cub as cub_data
from utils import visutil
from utils import bird_vis
from utils import image as image_utils
from nnutils import train_utils
from nnutils import loss_utils
from nnutils import mesh_net
from nnutils.nmr import NeuralRenderer
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops import SubdivideMeshes
from nnutils.geom_utils import mesh_laplacian
from pytorch3d.transforms import *
import pickle as pkl
import pytorch3d
from torch.nn import functional as F

flags.DEFINE_string('dataset', 'cub', 'cub')
flags.DEFINE_integer('num_lbs', 15, 'keypoint loss weight')
flags.DEFINE_string('mesh_dir', 'meshes/bird_aligned.obj', 'tmp dir to extract dataset')
flags.DEFINE_string('kp_dict', 'meshes/bird_kp_dictionary.pkl', 'tmp dir to extract dataset')
# Weights:
flags.DEFINE_float('kp_loss_wt', 30., 'keypoint loss weight')
flags.DEFINE_float('mask_loss_wt', 1., 'mask loss weight')
flags.DEFINE_float('cam_loss_wt', 2., 'weights to camera loss')
flags.DEFINE_float('deform_reg_wt', 10., 'reg to deformation')
flags.DEFINE_float('boundaries_reg_wt', 1., 'reg to sil consistency')
flags.DEFINE_float('edt_reg_wt', 0.1, 'weight for sil coverage')
flags.DEFINE_float('bdt_reg_wt', 0.1, 'weight for boundaries loss')
flags.DEFINE_float('triangle_reg_wt', 30., 'weights to triangle smoothness prior')
flags.DEFINE_float('vert2kp_loss_wt', .16, 'reg to vertex assignment')
flags.DEFINE_float('tex_loss_wt', .5, 'weights to tex loss')
flags.DEFINE_float('tex_dt_loss_wt', .5, 'weights to tex dt loss')
flags.DEFINE_boolean('use_gtpose', True, 'if true uses gt pose for projection, but camera still gets trained.')
flags.DEFINE_float('entropy_lbs_loss_wt', 0.0016, 'reg to vertex assignment')
flags.DEFINE_float('rigid_wt', 0.5, 'weight for rigid loss')

opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'misc', 'cachedir')


def mirror_sample(img, sfm_pose, mask_pred, mask):
    import kornia
    # Need copy bc torch collate doesnt like neg strides
    img_flip = torch.flip(img, dims=(3,))
    mask_pred_flip = torch.flip(mask_pred, dims=(2,))
    mask_flip = torch.flip(mask, dims=(2,))

    # Flip kps.
    # Flip sfm_pose Rot.
    quat = sfm_pose[:, -4:]
    quat = standardize_quaternion(quat)
    diag = torch.diag(torch.tensor([-1., 1., -1.], device=sfm_pose.device))[None]
    quat_new = quaternion_multiply(matrix_to_quaternion(diag), quat)
    scale = sfm_pose[:, :1]
    tx = - sfm_pose[:, 1:2]
    ty = sfm_pose[:, 2:3]
    sfm_pose = torch.cat([scale, tx, ty, quat_new], dim=-1)
    return img_flip, sfm_pose, mask_pred_flip, mask_flip


class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        # ----------
        # Options
        # ----------
        self.symmetric = opts.symmetric
        anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_train.mat')
        anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
        sfm_mean_shape = (np.transpose(anno_sfm['S']), anno_sfm['conv_tri'] - 1)

        kp_dict = None
        if opts.kp_loss_wt > 0:
            kp_dict = pkl.load(open(opts.kp_dict, 'rb'))
        mesh_horse = pytorch3d.io.load_obj(opts.mesh_dir)
        v, f = mesh_horse[0].numpy(), mesh_horse[1].verts_idx.numpy()
        shapenet_mesh = [v, f]

        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat, num_kps=opts.num_kps, sfm_mean_shape=sfm_mean_shape,
            shapenet_mesh=shapenet_mesh, kp_dict=kp_dict)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)

        self.model = self.model.cuda(device=opts.gpu_id)

        # For renderering.
        faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        self.renderer = NeuralRenderer(opts.img_size)
        self.renderer_predcam = NeuralRenderer(opts.img_size)  # for camera loss via projection

        # Need separate NMR for each fwd/bwd call.
        if opts.texture:
            self.tex_renderer = NeuralRenderer(opts.img_size)
            self.tex_renderer.ambient_light_only()

        mesh_template = Meshes(verts=[self.model.get_mean_shape()], faces=[self.faces[0]])
        self.sdivide = SubdivideMeshes(mesh_template)

        # For visualization
        self.faces_up = self.sdivide(mesh_template).faces_packed()
        num_verts_up = self.sdivide(mesh_template).verts_packed().shape[1]
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, num_verts_up, self.faces[:1].data.cpu().numpy())
        self.L = mesh_laplacian(mesh_template, 'uniform')
        return

    def init_dataset(self):
        opts = self.opts
        if opts.dataset == 'cub':
            self.data_module = cub_data
        else:
            print('Unknown dataset %d!' % opts.dataset)

        self.dataloader = self.data_module.data_loader(opts, shuffle=True)
        # self.dataloader = self.data_module.data_loader(opts, shuffle=False)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def define_criterion(self):
        self.template_reg = loss_utils.template_edge_loss
        self.projection_loss = loss_utils.kp_l2_loss
        self.tex_l1_loss_fn = torch.nn.L1Loss()
        self.mask_loss_fn = loss_utils.iou_loss
        self.boundaries_fn = loss_utils.bds_loss
        self.dt_fn = loss_utils.texture_dt_loss_v
        self.edt_fn = loss_utils.edt_loss
        self.entropy_loss = loss_utils.entropy_loss
        self.deform_reg_fn = loss_utils.deform_l2reg
        self.camera_loss = loss_utils.camera_loss
        self.locally_rigid_fn = loss_utils.Locally_Rigid()

        if self.opts.texture:
            self.texture_loss = loss_utils.PerceptualTextureLoss_v2()

            self.texture_dt_loss_fn = loss_utils.texture_dt_loss_v

    def set_input(self, batch):
        opts = self.opts

        # Image with annotations.
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        img_tensor = batch['img'].type(torch.FloatTensor)
        mask_tensor = batch['mask'].type(torch.FloatTensor)
        kp_tensor = batch['kp'].type(torch.FloatTensor)
        cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)

        self.input_imgs = input_img_tensor.cuda(device=opts.gpu_id)
        self.imgs = img_tensor.cuda(device=opts.gpu_id)
        self.masks = mask_tensor.cuda(device=opts.gpu_id)
        self.kps = kp_tensor.cuda(device=opts.gpu_id)
        self.cams = cam_tensor.cuda(device=opts.gpu_id)

        # Compute barrier distance transform.
        mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in batch['mask']])
        dt_tensor = torch.tensor(mask_dts).float().cuda(device=opts.gpu_id)
        # B x 1 x N x N
        self.dts_barrier = dt_tensor.unsqueeze(1)

        self.boundaries = image_utils.compute_boundaries(self.masks.cpu().numpy())
        self.boundaries = torch.tensor(self.boundaries).float().cuda(device=opts.gpu_id)
        mask_edt = np.stack([image_utils.compute_dt(m, norm=False) for m in batch['mask']])
        edt_tensor = torch.tensor(mask_edt).float().cuda(device=opts.gpu_id)
        # B x 1 x N x N
        self.edts_barrier = edt_tensor.unsqueeze(1)


    def forward(self):
        opts = self.opts
        # if opts.texture:
        #     pred_codes, self.textures = self.model(self.input_imgs)
        # else:
        img_feat, pred_codes, self.res_feats = self.model(self.input_imgs)
        scale, trans, quat = self.model.camera_predictor(self.res_feats)
        self.delta_v, _, _, _ = pred_codes
        batch_size = self.delta_v.shape[0]
        self.cam_pred = torch.cat([scale, trans, quat], 1)
        self.mean_shape = self.model.get_mean_shape()
        # Compute keypoints.

        self.vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        self.lbs = self.model.get_lbs().permute(1, 0)
        self.lbs = self.lbs[None].repeat(self.delta_v.shape[0], 1, 1)
        self.mean_v = self.mean_shape[None].repeat(self.delta_v.shape[0], 1, 1)
        self.delta_v_ms = self.lbs.bmm(self.mean_v)
        self.delta_v = self.delta_v_ms + self.delta_v[:, 0]

        # Deform mean shape:

        L = self.L.repeat(self.delta_v.shape[0], 1, 1)
        delta = torch.bmm(L, self.mean_v)
        A = self.lbs
        A_augm = L.permute(0, 2, 1).matmul(L) + A.permute(0, 2, 1).matmul(A)
        b = L.permute(0, 2, 1) @ delta + A.permute(0, 2, 1) @ self.delta_v
        u = torch.cholesky(A_augm)
        self.pred_v = torch.cholesky_solve(b, u)
        vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        self.kp_verts = torch.matmul(vert2kp, self.mean_v)
        self.kp_verts_pred_v = torch.matmul(vert2kp, self.pred_v)
        self.kp_verts_transformed = self.kp_verts_pred_v

        # Decide which camera to use for projection.
        if opts.use_gtpose:
            proj_cam = self.cams
        else:
            proj_cam = self.cam_pred

        # Project keypoints
        self.kp_pred_transformed = self.renderer.project_points(self.kp_verts_transformed, proj_cam)
        self.kp_pred = self.renderer.project_points(self.kp_verts, proj_cam)

        faces = self.faces
        if opts.texture:
            self.textures = self.model.texture_predictor.forward(self.pred_v, self.res_feats)
            self.mask_pred, pix_to_face = self.renderer(self.pred_v, faces, proj_cam)
            self.texture_pred, _, _ = self.tex_renderer(self.pred_v.detach(), self.faces, proj_cam,
                                                        textures=self.textures)
            self.imgs_flip, proj_cam_flip, self.mask_pred_flip, self.masks_flip = mirror_sample(self.imgs,
                                                                                                proj_cam,
                                                                                                self.mask_pred,
                                                                                                self.masks)
            self.texture_pred_flip, _, _ = self.tex_renderer(self.pred_v.detach(),
                                                             self.faces, proj_cam_flip, textures=self.textures)
        else:
            self.textures = None
            self.mask_pred = self.renderer(self.pred_v, self.faces, proj_cam)
            # Compute losses for this instance.
        self.kp_loss = self.projection_loss(self.kp_pred_transformed, self.kps)

        self.mask_loss = self.mask_loss_fn(self.mask_pred, self.masks)
        self.cam_loss = self.camera_loss(self.cam_pred, self.cams, 0)

        if opts.texture:
            self.tex_loss = 0.5 * self.texture_loss(self.texture_pred, self.imgs, self.mask_pred, self.masks) + \
                            0.5 * self.texture_loss(self.texture_pred_flip, self.imgs_flip, self.mask_pred_flip,
                                                    self.masks_flip)
            tex_l1 = 0.5 * (
                    F.mse_loss(self.texture_pred * self.masks.unsqueeze(1),
                              (self.imgs * self.masks.unsqueeze(1))) + F.mse_loss(
                self.texture_pred_flip * self.masks_flip.unsqueeze(1),
                (self.imgs_flip * self.masks_flip.unsqueeze(1))))
            self.tex_loss += tex_l1


        pred_proj = self.renderer.project_points(self.pred_v, proj_cam)
        self.edt_loss = self.edt_fn(self.mask_pred, self.edts_barrier)
        self.bdt_loss = self.boundaries_fn(pred_proj, self.boundaries, self.faces, pix_to_face)
        self.sil_cons = opts.edt_reg_wt * self.edt_loss + opts.bdt_reg_wt * self.bdt_loss

        # Priors:
        mesh_3d = Meshes(verts=self.pred_v, faces=self.faces)
        mesh_template = Meshes(verts=self.mean_v, faces=faces)
        self.rigid_loss = self.locally_rigid_fn(mesh_3d, mesh_template).mean()
        loss_laplacian = mesh_laplacian_smoothing(mesh_3d, method="uniform")
        self.vert2kp_loss = self.entropy_loss(vert2kp)
        self.deform_reg = self.deform_reg_fn(self.delta_v)
        self.triangle_loss = loss_laplacian
        # Finally sum up the loss.
        # Instance loss:
        self.total_loss = opts.mask_loss_wt * self.mask_loss
        self.total_loss += opts.boundaries_reg_wt * self.sil_cons

        self.total_loss += opts.kp_loss_wt * self.kp_loss
        self.total_loss += opts.cam_loss_wt * self.cam_loss
        if opts.texture:
            self.total_loss += opts.tex_loss_wt * self.tex_loss

        # Priors:
        self.total_loss += opts.vert2kp_loss_wt * self.vert2kp_loss
        self.total_loss += opts.rigid_wt * self.rigid_loss
        self.total_loss += opts.triangle_reg_wt * self.triangle_loss


    def get_current_visuals(self):
        vis_dict = {}
        mask_concat = torch.cat([self.masks, self.mask_pred], 2)


        num_show = min(2, self.opts.batch_size)

        for i in range(num_show):
            input_img = bird_vis.kp2im(self.kps[i].data, self.imgs[i].data)
            pred_kp_img = bird_vis.kp2im(self.kp_pred[i].data, self.imgs[i].data)
            pred_transformed_kp_img = bird_vis.kp2im(self.kp_pred_transformed[i].data, self.imgs[i].data)
            masks = bird_vis.tensor2mask(mask_concat[i].data)
            if self.opts.texture:
                texture_here = self.textures[i]
            else:
                texture_here = None

            rend_predcam = self.vis_rend(self.pred_v[i], self.cam_pred[i], texture=texture_here)
            # Render from front & back:
            rend_frontal = self.vis_rend.diff_vp(self.pred_v[i], self.cam_pred[i], texture=texture_here,
                                                 kp_verts=self.kp_verts_transformed[i])
            rend_top = self.vis_rend.diff_vp(self.pred_v[i], self.cam_pred[i], axis=[0, 1, 0], texture=texture_here,
                                             kp_verts=self.kp_verts_transformed[i])
            diff_rends = np.hstack((rend_frontal, rend_top))

            if self.opts.texture:
                tex_img = bird_vis.tensor2im(self.texture_pred[i].data)
                imgs = np.hstack((input_img, pred_kp_img, pred_transformed_kp_img, tex_img))
            else:
                imgs = np.hstack((input_img, pred_transformed_kp_img))

            rend_gtcam = self.vis_rend(self.pred_v[i], self.cams[i], texture=texture_here)
            rends = np.hstack((diff_rends, rend_predcam, rend_gtcam))
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
            ('kp_loss', self.kp_loss.item()),
            ('sil_cons', self.sil_cons.item()),
            ('edt_loss', self.edt_loss.item()),
            ('bdt_loss', self.bdt_loss.item()),
            ('mask_loss', self.mask_loss.item()),
            ('rigid_loss', self.rigid_loss.item()),
            ('vert2kp_loss', self.vert2kp_loss.item()),
            ('deform_reg', self.deform_reg.item()),
            ('tri_loss', self.triangle_loss.item()),
            ('cam_loss', self.cam_loss.item()),
        ])
        if self.opts.texture:
            sc_dict['tex_loss'] = self.tex_loss.item()
        return sc_dict


def main(_):
    torch.manual_seed(0)
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    trainer.train()


if __name__ == '__main__':
    app.run(main)
