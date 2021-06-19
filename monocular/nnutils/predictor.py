"""
Takes an image, returns stuff.
"""

import os
import os.path as osp
import pickle as pkl

import numpy as np
import pytorch3d
import scipy.io as sio
import torch
import torchvision
from absl import flags
from nnutils import mesh_net
from nnutils.geom_utils import mesh_laplacian
from nnutils.nmr import NeuralRenderer
from pytorch3d.structures import Meshes
from torch.autograd import Variable
from utils import bird_vis

# These options are off by default, but used for some ablations reported.
flags.DEFINE_boolean('ignore_pred_delta_v', False, 'Use only mean shape for prediction')
flags.DEFINE_integer('num_lbs', 32, '')
flags.DEFINE_boolean('use_sfm_ms', False, 'Uses sfm mean shape for prediction')
flags.DEFINE_string('mesh_dir', 'meshes/bird_aligned.obj', 'tmp dir to extract dataset')
flags.DEFINE_string('kp_dict', 'meshes/bird_kp_dictionary.pkl', 'tmp dir to extract dataset')
flags.DEFINE_boolean('optimize', False, 'Uses sfm mean shape for prediction')
flags.DEFINE_boolean('optimize_camera', False, 'Uses sfm mean shape for prediction')
flags.DEFINE_integer('num_optim_iter', 20, 'Uses sfm mean shape for prediction')


class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts

        self.symmetric = opts.symmetric
        anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_train.mat')
        anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
        sfm_mean_shape = (np.transpose(anno_sfm['S']), anno_sfm['conv_tri'] - 1)

        kp_dict = pkl.load(open(opts.kp_dict, 'rb'))
        mesh_horse = pytorch3d.io.load_obj(opts.mesh_dir)
        v, f = mesh_horse[0].numpy(), mesh_horse[1].verts_idx.numpy()
        shapenet_mesh = [v, f]

        img_size = (opts.img_size, opts.img_size)
        print('Setting up model..')
        self.model = mesh_net.MeshNet(img_size, opts, nz_feat=opts.nz_feat, sfm_mean_shape=sfm_mean_shape,
                                      shapenet_mesh=shapenet_mesh, kp_dict=kp_dict)

        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.model.eval()
        self.model = self.model.cuda(device=self.opts.gpu_id)

        self.renderer = NeuralRenderer(opts.img_size)

        if opts.texture:
            self.tex_renderer = NeuralRenderer(opts.img_size)
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()

        faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        mesh_template = Meshes(verts=[self.model.get_mean_shape()], faces=[self.faces[0]])
        num_verts_up = mesh_template.verts_packed().shape[1]
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, num_verts_up, self.faces[:1].data.cpu().numpy())

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        network_dir = os.path.join(self.opts.checkpoint_dir, self.opts.name)
        save_path = os.path.join(network_dir, save_filename)
        print('loading {}..'.format(save_path))
        try:
            network.load_state_dict(torch.load(save_path))
        except Exception as e:
            print(e)
            print('Loadining non-strict')
            network.load_state_dict(torch.load(save_path), strict=False)
        return

    def set_input(self, batch):
        opts = self.opts
        # original image where texture is sampled from.
        img_tensor = batch['img'].clone().type(torch.FloatTensor)
        # input_img is the input to resnet
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs = Variable(
            input_img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(device=opts.gpu_id), requires_grad=False)

        mask_tensor = batch['mask'].type(torch.FloatTensor)
        self.masks = mask_tensor.cuda(device=opts.gpu_id)

    def predict(self, batch):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.forward()
        return self.collect_outputs()

    def forward(self):
        img_feat, pred_codes, self.res_feats = self.model(self.input_imgs)
        scale, trans, quat = self.model.camera_predictor(self.res_feats)
        self.delta_v_res, _, _, _ = pred_codes

        self.cam_pred = torch.cat([scale, trans, quat], 1)
        self.mean_shape = self.model.get_mean_shape()
        # Compute keypoints.
        self.vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        self.lbs = self.model.get_lbs().permute(1, 0)
        self.lbs = self.lbs[None].repeat(self.delta_v_res.shape[0], 1, 1)

        self.mean_v = self.mean_shape[None].repeat(self.delta_v_res.shape[0], 1, 1)
        self.delta_v_ms = self.lbs.bmm(self.mean_v)
        self.delta_v = self.delta_v_ms + self.delta_v_res[:, 0]

        # Deform mean shape:
        src_mesh = Meshes(verts=self.mean_shape[None], faces=self.faces[:1])
        L = mesh_laplacian(src_mesh, 'uniform')
        L = L.repeat(self.delta_v.shape[0], 1, 1)
        delta = torch.bmm(L, self.mean_v)
        A = self.lbs
        A_augm = L.permute(0, 2, 1).matmul(L) + A.permute(0, 2, 1).matmul(A)
        b = L.permute(0, 2, 1) @ delta + A.permute(0, 2, 1) @ self.delta_v
        u = torch.cholesky(A_augm)
        self.pred_v = torch.cholesky_solve(b, u)

        # Project keypoints

        vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        self.kp_verts_pred_v = torch.matmul(vert2kp, self.pred_v)
        self.kp_verts_transformed = self.kp_verts_pred_v

        self.vert2kp = torch.nn.functional.softmax(
            self.model.vert2kp, dim=1)
        self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        # Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts, self.cam_pred)
        self.mask_pred, _ = self.renderer.forward(self.pred_v, self.faces[:1].repeat(self.pred_v.shape[0], 1, 1),
                                                  self.cam_pred)
        # Render texture.
        if self.opts.texture and not self.opts.use_sfm_ms:
            faces = self.faces[:1].repeat(self.pred_v.shape[0], 1, 1).clone()
            self.textures = self.model.texture_predictor.forward(self.pred_v, self.res_feats)
            self.texture_pred = self.tex_renderer(self.pred_v, faces, self.cam_pred,
                                                  textures=self.textures)[0]

        else:
            self.textures = None

    def collect_outputs(self):
        outputs = {
            'lbs': self.lbs.data,
            'mean_shape': self.mean_v.data,
            'faces': self.faces.data,
            'delta_v_res': self.delta_v_res.data,
            'kp_pred': self.kp_pred.data,
            'verts': self.pred_v.data,
            'kp_verts': self.kp_verts.data,
            'cam_pred': self.cam_pred.data,
            'mask_pred': self.mask_pred.data,
        }

        return outputs
