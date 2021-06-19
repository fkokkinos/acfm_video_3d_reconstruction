from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pkl

import numpy as np
import pytorch3d
import torch
import torch.nn.functional as F
import torchvision
import yaml
from absl import flags
from data.optical_flow import config_folder as cf
from data.optical_flow.model import Upsample
from nnutils import loss_utils
from nnutils import mesh_net
from nnutils.geom_utils import mesh_laplacian
from nnutils.nmr import NeuralRenderer, OF_NeuralRenderer
from pytorch3d.structures import Meshes
from utils import bird_vis
from utils import image as image_utils
from data.optical_flow.model import MaskFlownet, MaskFlownet_S, Upsample

# These options are off by default, but used for some ablations reported.
flags.DEFINE_integer('num_lpl', 32, 'Number of handles')
flags.DEFINE_integer('num_kps', 16, 'Number of keypoints')
flags.DEFINE_integer('img_size', 256, 'Image Size for rendering and scaling of data')
flags.DEFINE_boolean('use_argmax_camera', False, 'Uses argmax mean camera (only for training)')
flags.DEFINE_integer('num_guesses', 4, 'Number of hypothesis for camera and deformation')
flags.DEFINE_boolean('optimize', False, 'Apply post-processing optimization for deformation')
flags.DEFINE_boolean('optimize_camera', False, 'Apply post-processing optimization for camera')
flags.DEFINE_integer('num_optim_iter', 20, 'Number of post-processing optimization iterations')
flags.DEFINE_float('of_loss_wt', 0.1, 'optical flow loss weight')
flags.DEFINE_float('mask_loss_wt', 1., 'mask loss weight')
flags.DEFINE_float('boundaries_reg_wt', 1., 'reg to deformation')
flags.DEFINE_float('edt_reg_wt', 0.1, 'reg to deformation')
flags.DEFINE_float('bdt_reg_wt', 0.1, 'reg to deformation')
flags.DEFINE_float('scale_lr_decay', 0.05, 'Warmup the pose predictor')
flags.DEFINE_float('scale_bias', 1.0, 'Warmup the pose predictor')
flags.DEFINE_boolean('scale_template', False, 'Scale template')


def mirror_sample(img, sfm_pose, mask_pred, mask):
    img_flip = torch.flip(img, dims=(3,))
    mask_pred_flip = torch.flip(mask_pred, dims=(2,))
    mask_flip = torch.flip(mask, dims=(2,))
    sfm_pose = sfm_pose * torch.FloatTensor([1, -1, 1, 1, 1, -1, -1]).to(device=sfm_pose.device)

    return img_flip, sfm_pose, mask_pred_flip, mask_flip


class MeshPredictor(object):
    def __init__(self, opts, testing_samples):
        self.opts = opts

        self.symmetric = opts.symmetric
        if opts.category in ['horse', 'tiger', 'cow']:
            anno_sfm_path = 'data/sfm_inits/' + opts.category + '/'
        sfm_mean_shape = torch.load(anno_sfm_path + 'sfm.pth')
        print('Setting up model..')
        kp_dict = pkl.load(open(opts.kp_dict, 'rb'))
        mesh_horse = pytorch3d.io.load_obj(opts.mesh_dir)
        v, f = mesh_horse[0].numpy(), mesh_horse[1].verts_idx.numpy()
        if opts.scale_template:
            scale = 2. / torch.max(torch.nn.functional.pdist(torch.from_numpy(v))).numpy()
            v = v * scale
            v = v - v.mean(0)

        shapenet_mesh = [v, f]
        img_size = (opts.img_size, opts.img_size)
        if opts.split == 'train':
            self.model = mesh_net.MeshNet(
                img_size, opts, nz_feat=opts.nz_feat, num_kps=opts.num_kps, sfm_mean_shape=sfm_mean_shape,
                cam_embeddings=testing_samples, shapenet_mesh=shapenet_mesh, no_kps=True, kp_dict=kp_dict)
        else:
            self.model = mesh_net.MeshNet(
                img_size, opts, nz_feat=opts.nz_feat, num_kps=opts.num_kps, sfm_mean_shape=sfm_mean_shape,
                cam_embeddings=None, shapenet_mesh=shapenet_mesh, no_kps=True, kp_dict=kp_dict)

        self.model = torch.nn.DataParallel(self.model).cuda()

        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.model.eval()

        # Setting up renderers
        self.renderer = NeuralRenderer(opts.img_size)
        self.of_renderer = OF_NeuralRenderer(opts.img_size).cuda()
        if opts.texture:
            self.tex_renderer = NeuralRenderer(opts.img_size)
            self.tex_renderer.ambient_light_only()

        faces = self.model.module.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        mesh_template = Meshes(verts=[self.model.module.get_mean_shape()], faces=[self.faces[0]])
        num_verts_up = mesh_template.verts_packed().shape[1]
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, num_verts_up, self.faces[:1].data.cpu().numpy())
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Instantiate Losses for post-processing optimization
        if self.opts.optimize:
            self.mask_loss_fn = loss_utils.l1_loss
            self.boundaries_fn = loss_utils.Boundaries_Loss()
            self.dt_fn = loss_utils.texture_dt_loss_v
            self.edt_fn = loss_utils.edt_loss
            self.texture_loss = loss_utils.PerceptualTextureLoss_v2()

        # Load OF Network
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

    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        network_dir = os.path.join(self.opts.checkpoint_dir, self.opts.name)
        save_path = os.path.join(network_dir, save_filename)
        print('loading {}..'.format(save_path))
        try:
            network.load_state_dict(torch.load(save_path))
        except Exception as e:
            # In case of error, try load with non-strict
            print(e)
            print('Loadining non-strict')
            network.load_state_dict(torch.load(save_path), strict=False)
        return

    def centralize(self, img1, img2):
        rgb_mean = torch.cat((img1, img2), 2)
        rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
        rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    def set_input(self, batch):
        opts = self.opts
        if batch['img'].ndim == 4:
            batch['img'] = batch['img'].unsqueeze(1)
        if batch['mask'].ndim == 3:
            batch['mask'] = batch['mask'].unsqueeze(1)
        if batch['kp'].ndim == 3:
            batch['kp'] = batch['kp'].unsqueeze(1)
        if batch['sfm_pose'].ndim == 2:
            batch['sfm_pose'] = batch['sfm_pose'].unsqueeze(1)

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

        self.cam_loss = torch.zeros(1)
        # Predict Optical flow
        if t > 1:
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

            batch['optical_flows'] = self.optical_flows

    def predict(self, batch):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.forward()
        return self.collect_outputs()

    def forward(self):
        img_feat, self.res_feats, pred_codes = self.model(self.input_imgs)
        self.delta_v_res, _, _, _ = pred_codes
        scale, trans, quat = self.model.module.camera_predictor(self.res_feats)
        if self.opts.use_argmax_camera:
            # If camera embeddings are available select argmax (only for the case of train set)
            probs_weights = self.model.module.prob_embeddings.weight.data
            selected_cameras_probs = probs_weights[self.frames_idx].topk(1, largest=True, dim=-1)[1]
            selected_cameras_probs = selected_cameras_probs.permute(2, 0, 1)
            selected_cameras = selected_cameras_probs[..., None].repeat(1, 1, 1, 7)
            cameras = [cam_emb(self.frames_idx) for cam_emb in self.model.module.cameras]
            cameras = torch.stack(cameras)
            cameras = torch.gather(cameras, 0, selected_cameras)[0, :, 0]
            quats = cameras[..., 3:]
            scales = torch.nn.functional.relu(self.opts.scale_lr_decay * cameras[..., :1] + 1) + 1e-12
            translations = cameras[..., 1:3]
            quats_n = torch.nn.functional.normalize(quats, dim=-1)
            self.cam_pred = torch.cat([scales, translations, quats_n], dim=1)
        else:
            # else predicted cameras are the network's regressed quantities
            self.cam_pred = torch.cat([scale, trans, quat], 1)

        self.mean_shape = self.model.module.get_mean_shape()
        self.vert2kp = torch.nn.functional.softmax(self.model.module.vert2kp, dim=1)

        self.lbs = self.model.module.get_lbs().permute(1, 0)
        self.lbs = self.lbs[None].repeat(self.delta_v_res.shape[0], 1, 1)
        self.mean_v = self.mean_shape[None].repeat(self.delta_v_res.shape[0], 1, 1)
        self.delta_v_ms = self.lbs.bmm(self.mean_v)

        self.delta_v = self.delta_v_ms + self.delta_v_res.squeeze(1)

        # Deform mean shape
        src_mesh = Meshes(verts=self.mean_shape[None], faces=self.faces[:1])
        L = mesh_laplacian(src_mesh, 'cot')
        L = L.repeat(self.delta_v.shape[0], 1, 1)
        delta = torch.bmm(L, self.mean_v)
        A = self.lbs
        A_augm = L.permute(0, 2, 1).matmul(L) + A.permute(0, 2, 1).matmul(A)
        b = L.permute(0, 2, 1).bmm(delta) + A.permute(0, 2, 1).bmm(self.delta_v)
        u = torch.cholesky(A_augm)
        self.pred_v = torch.cholesky_solve(b, u)


        # store predicted deformation, kps and masks
        self.pred_v_orig = self.pred_v.clone().detach()
        self.kp_verts_orig = torch.matmul(self.vert2kp, self.pred_v_orig)
        self.kp_pred_orig = self.renderer.project_points(self.kp_verts_orig, self.cam_pred)
        self.mask_pred_orig, _ = self.renderer.forward(self.pred_v_orig,
                                                       self.faces[:1].repeat(self.pred_v_orig.shape[0], 1, 1),
                                                       self.cam_pred)

        if self.opts.optimize:
            # Apply post-processing optimization
            if self.opts.optimize_camera:
                scale, trans, quat = scale.clone().detach().requires_grad_(True), trans.clone().detach().requires_grad_(
                    True), quat.clone().detach().requires_grad_(True)
            delta_v_res = self.delta_v_res[:, 0].clone().detach().requires_grad_(True)
            params = [delta_v_res]
            if self.opts.optimize_camera:
                params += [scale, trans, quat]
            post_optimizer = torch.optim.Adam(params, lr=5e-3)
            src_mesh = Meshes(verts=self.mean_shape[None], faces=self.faces[:1])
            L = mesh_laplacian(src_mesh, 'cot')
            L = L.repeat(self.delta_v.shape[0], 1, 1).detach()
            A = self.lbs.detach()
            A_augm = L.permute(0, 2, 1).matmul(L) + A.permute(0, 2, 1).matmul(A).detach()
            delta = torch.bmm(L, self.mean_v).detach()
            self.cam_pred = self.cam_pred.detach()

            for i in range(self.opts.num_optim_iter):

                if self.opts.optimize_camera:
                    quat_n = torch.nn.functional.normalize(quat, dim=-1)
                    self.cam_pred = torch.cat([scale, trans, quat_n], dim=1)
                self.delta_v = self.delta_v_ms.detach() + delta_v_res

                # Deform mean shape:
                b = L.permute(0, 2, 1) @ delta + A.permute(0, 2, 1) @ self.delta_v
                u = torch.cholesky(A_augm)
                self.pred_v = torch.cholesky_solve(b, u)
                faces = self.faces[:1].repeat(self.pred_v.shape[0], 1, 1).clone()
                self.mask_pred, pix_to_face = self.renderer(self.pred_v, faces, self.cam_pred)
                self.mask_loss = self.mask_loss_fn(self.mask_pred, self.masks)
                pred_proj = self.renderer.project_points(self.pred_v, self.cam_pred)
                self.edt_loss = self.edt_fn(self.mask_pred, self.edts_barrier)
                self.bdt_loss = self.boundaries_fn(pred_proj, self.boundaries, faces, pix_to_face)
                self.sil_cons = self.opts.bdt_reg_wt * self.edt_loss + self.opts.edt_reg_wt * self.bdt_loss

                if self.opts.of_loss_wt > 0:
                    batch_size = self.optical_flows.shape[0]
                    masks_of = self.masks.reshape(batch_size, self.opts.num_frames, self.masks.shape[1],
                                                  self.masks.shape[2])
                    pred_v_of = self.pred_v.reshape(batch_size, self.opts.num_frames, self.pred_v.shape[1],
                                                    self.pred_v.shape[2])
                    proj_cam_of = self.cam_pred
                    faces = self.faces[:1].repeat(batch_size, 1, 1).clone()
                    faces_of = faces[:, None].repeat(1, self.opts.num_frames, 1, 1)
                    optical_flows_f = torch.flip(self.optical_flows, dims=[1]) * masks_of[:, :, :, :, None]
                    self.of_loss, self.of_pred_cp, self.visible_vertices, self.verts_of, self.samples_ofs_gt = loss_utils.optical_flow_loss(
                        pred_v_of,
                        faces_of,
                        proj_cam_of,
                        optical_flows_f,
                        self.of_renderer, pix_to_face=pix_to_face)

                # Finally sum up the loss.
                # Instance loss:
                self.total_loss = self.opts.mask_loss_wt * self.mask_loss
                self.total_loss += self.opts.boundaries_reg_wt * self.sil_cons
                if self.opts.of_loss_wt > 0:
                    self.total_loss += self.opts.of_loss_wt * self.of_loss
                post_optimizer.zero_grad()
                self.total_loss.backward()
                post_optimizer.step()

        # Compute keypoints for deformed mesh.
        self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        # Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts, self.cam_pred)
        self.mask_pred, _ = self.renderer.forward(self.pred_v, self.faces[:1].repeat(self.pred_v.shape[0], 1, 1),
                                                  self.cam_pred)
        # Render texture.
        if self.opts.texture:
            faces = self.faces[:1].repeat(self.pred_v.shape[0], 1, 1).clone()
            self.textures = self.model.module.texture_predictor.forward(self.pred_v, self.res_feats)
            self.texture_pred = self.tex_renderer(self.pred_v, faces, self.cam_pred,
                                                  textures=self.textures)[0]
            self.texture_pred_orig = self.tex_renderer(self.pred_v_orig, faces, self.cam_pred,
                                                       textures=self.textures)[0]
        else:
            self.textures = None

    def collect_outputs(self):
        outputs = {
            'kp_pred': self.kp_pred.data,
            'mean_v': self.mean_v.data,
            'lbs': self.lbs.data,
            'faces': self.faces.data,
            'delta_v_res': self.delta_v_res.data,
            'kp_pred_orig': self.kp_pred_orig.data,
            'verts': self.pred_v.data,
            'verts_orig': self.pred_v_orig.data,
            'kp_verts': self.kp_verts.data,
            'kp_verts_orig': self.kp_verts_orig.data,
            'cam_pred': self.cam_pred.data,
            'mask_pred': self.mask_pred.data,
            'mask_pred_orig': self.mask_pred_orig.data,
        }
        if self.opts.texture:
            outputs['texture'] = self.textures
            outputs['texture_pred'] = self.texture_pred.data
            outputs['texture_pred_orig'] = self.texture_pred_orig.data

        return outputs
