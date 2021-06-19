"""
Mesh net model.
"""
import gdist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from absl import flags
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from scipy.spatial import ConvexHull
from utils import geometry as geom_utils
from utils.mesh import create_sphere, compute_uvsampler, make_symmetric

from . import net_blocks as nb
from . import networks

# -------------- flags -------------#
# ----------------------------------#
flags.DEFINE_boolean('symmetric', True, 'Use symmetric mesh or not')
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_boolean('symmetric_texture', True, 'if true texture is symmetric!')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')

flags.DEFINE_boolean('only_mean_sym', False, 'If true, only the meanshape is symmetric')
flags.DEFINE_boolean('learnable_kp', True, 'If true, only the meanshape is symmetric')


# compute euclidean distance matrix
def euclidean_distance_matrix(x):
    r = np.sum(x * x, 1)
    r = r.reshape(-1, 1)
    distance_mat = r - 2 * np.dot(x, x.T) + r.T
    # return np.sqrt(distance_mat)
    return distance_mat


# update distance matrix and select the farthest point from set S after a new point is selected
def update_farthest_distance(far_mat, dist_mat, s):
    for i in range(far_mat.shape[0]):
        far_mat[i] = dist_mat[i, s] if far_mat[i] > dist_mat[i, s] else far_mat[i]
    return far_mat, np.argmax(far_mat)


# initialize matrix to keep track of distance from set s
def init_farthest_distance(far_mat, dist_mat, s):
    for i in range(far_mat.shape[0]):
        far_mat[i] = dist_mat[i, s]
    return far_mat


# get sample from farthest point on every iteration
def farthest_point_sampling(verts, faces, num_samples=1000):
    set_P = verts
    set_P = np.array(set_P)
    num_P = set_P.shape[0]
    # distance_mat = euclidean_distance_matrix(set_P)
    distance_mat = gdist.local_gdist_matrix(verts.astype(np.float64),
                                            faces.astype(np.int32))
    set_S = []
    selected = []

    s = 0  # np.random.randint(num_P)
    selected.append(s)
    far_mat = init_farthest_distance(np.zeros((num_P)), distance_mat, s)
    for i in range(num_samples):
        set_S.append(set_P[s])
        far_mat, s = update_farthest_distance(far_mat, distance_mat, s)
        selected.append(s)
    return np.array(set_S), np.array(selected)


# ------------- Modules ------------#
# ----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv.forward(img)
        out_enc_conv1 = self.enc_conv1(resnet_feat)

        out_enc_conv2 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv2)

        return feat, out_enc_conv1


class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, uv_sampler, opts, img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False,
                 symmetric=False, num_sym_faces=624):
        super(TexturePredictorUV, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.symmetric = symmetric
        self.num_sym_faces = num_sym_faces
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T * self.T, 2)
        if predict_flow:
            nc_final = 2
        else:
            nc_final = 3
        modules = [networks.ResLayer_2d(256, 256, 1), nn.Upsample(scale_factor=2, mode='bilinear'),
                   networks.ResLayer_2d(256, 256, 1),
                   networks.ResLayer_2d(256, 256, 1), nn.Upsample(scale_factor=2, mode='bilinear'),
                   networks.ResLayer_2d(256, 128, 1), nn.Upsample(scale_factor=2, mode='bilinear'),
                   networks.ResLayer_2d(128, 64, 1), nn.Upsample(scale_factor=2, mode='bilinear'),
                   networks.ResLayer_2d(64, 32, 1), nn.Upsample(scale_factor=2, mode='bilinear'),
                   networks.ResLayer_2d(32, 16, 1), networks.conv3x3(16, 3)]
        self.res_color_net = nn.Sequential(*modules)

    def forward(self, pred_v, feat):
        feat = F.interpolate(feat, scale_factor=[1, 2], mode='bilinear')

        # pdb.set_trace()
        uvimage_pred = self.res_color_net(feat)
        tex_pred = torch.nn.functional.grid_sample(uvimage_pred, self.uv_sampler, align_corners=True)
        tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)
        tex_pred = (F.tanh(tex_pred) + 1) / 2
        if self.symmetric:
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces:]
            return torch.cat([tex_pred, tex_left], 1)
        else:
            # Contiguous Needed after the permute..
            return tex_pred


class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(200, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        return quat


class ScalePredictor(nn.Module):
    def __init__(self, nz, opts):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)
        self.opts = opts

    def forward(self, feat):
        scale = self.pred_layer.forward(feat) + 1  # biasing the scale to 1
        scale = torch.nn.functional.relu(scale) + 1e-12
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat)
        return trans


class TransformationPredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, kp_size):
        super(TransformationPredictor, self).__init__()
        # self.pred_layer = nb.fc(True, nz_feat, num_verts)
        self.kp_size = kp_size
        self.final_layer_angles = nn.Linear(nz_feat, kp_size * 3, bias=False)
        self.final_layer_trans = nn.Linear(nz_feat, kp_size * 3)

        # Initialize pred_layer weights to be small so initial def aren't so big
        self.final_layer_angles.weight.data.normal_(0, 0.00001)
        self.final_layer_trans.weight.data.normal_(0, 0.00001)

    def forward(self, feat):
        feat = feat[:, None]
        if feat.ndim == 3:
            b, t, _ = feat.shape
            translations_pred = self.final_layer_trans(feat)
            translations_pred = translations_pred.view(translations_pred.shape[0], translations_pred.shape[1], -1, 3)
            return translations_pred
        else:
            raise NotImplementedError


class CodePredictor(nn.Module):
    def __init__(self, num_lbs, nz_feat=100, num_verts=1000, opts=None):
        super(CodePredictor, self).__init__()
        self.quat_predictor = QuatPredictor(nz_feat)
        self.transform_predictor = TransformationPredictor(nz_feat, num_lbs)
        self.scale_predictor = ScalePredictor(nz_feat, opts)
        self.trans_predictor = TransPredictor(nz_feat)

    def forward(self, feat):
        transform_pred = self.transform_predictor(feat)
        return transform_pred, None, None, None


class CameraPredictor(nn.Module):
    def __init__(self, opts, nz_feat=100):
        super(CameraPredictor, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(200, 200), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(200, 200), nn.LeakyReLU())

        self.conv_c = nn.Sequential(nn.Conv2d(256, 200, kernel_size=4), nn.LeakyReLU())
        self.quat_predictor = QuatPredictor(200)
        self.scale_predictor = ScalePredictor(200, opts)
        self.trans_predictor = TransPredictor(200)

    def forward(self, feat):
        feat = self.conv_c(feat)[..., 0, 0]
        feat = feat + self.fc1(feat)
        feat = feat + self.fc2(feat)
        scale_pred = self.scale_predictor.forward(feat)
        quat_pred = self.quat_predictor.forward(feat)
        trans_pred = self.trans_predictor.forward(feat)
        return scale_pred, trans_pred, quat_pred


def safe_ln(x, minval=0.0000000001):
    return torch.log(torch.clamp(x, min=minval))


# ------------ Mesh Net ------------#
# ----------------------------------#
class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, num_kps=15, sfm_mean_shape=None, shapenet_mesh=None,
                 kp_dict=None):
        # Input shape is H x W of the image.
        super(MeshNet, self).__init__()
        self.opts = opts
        self.pred_texture = opts.texture
        self.symmetric = opts.symmetric
        self.symmetric_texture = opts.symmetric_texture

        # Mean shape.
        if shapenet_mesh is not None:
            sfm_mean_shape = shapenet_mesh
        else:
            verts, faces = create_sphere(opts.subdivide)
            num_verts = verts.shape[0]
            if sfm_mean_shape is not None:
                hull = ConvexHull(sfm_mean_shape[0])
                sfm_mean_shape = (sfm_mean_shape[0], hull.simplices)

        if self.symmetric:
            verts, faces, num_indept, num_sym, num_indept_faces, num_sym_faces = make_symmetric(verts, faces)
            if sfm_mean_shape is not None:
                verts = geom_utils.fit_verts_to_mesh(verts, faces, sfm_mean_shape[0], sfm_mean_shape[1])
                print('sym', verts.shape, faces.shape)
            verts = verts.astype(np.float32)

            num_sym_output = num_indept + num_sym
            if opts.only_mean_sym:
                print('Only the mean shape is symmetric!')
                self.num_output = num_verts
            else:
                self.num_output = num_sym_output
            num_verts = verts.shape[0]
            self.num_sym = num_sym
            self.num_indept = num_indept
            self.num_indept_faces = num_indept_faces
            self.num_sym_faces = num_sym_faces
            # mean shape is only half.
            self.mean_v = nn.Parameter(torch.tensor(verts[:num_sym_output]))

            # Needed for symmetrizing..
            self.flip = torch.ones(1, 3).cuda()
            self.flip[0, 0] = -1
        else:
            if sfm_mean_shape is not None:
                verts, faces = sfm_mean_shape[0], sfm_mean_shape[1]
                print(verts.shape, faces.shape)
            verts = verts.astype(np.float32)
            num_verts = verts.shape[0]
            self.mean_v = nn.Parameter(torch.tensor(verts))
            self.num_output = num_verts

        verts_np = verts
        faces_np = faces
        self.faces = torch.tensor(faces).long().cuda()

        def safe_ln(x, minval=0.0000000001):
            return torch.log(torch.clamp(x, min=minval))

        if kp_dict:
            vert2kp_init = torch.zeros(len(kp_dict), verts.shape[0]).float()
            for i_kp, k in enumerate(kp_dict):
                idx = kp_dict[k]
                print(k, idx)
                vert2kp_init[i_kp, idx] = 1

            kps = vert2kp_init @ verts
            print('kps', kps)
            pp = 12
            dists_full = torch.zeros(verts.shape[0], num_kps).float()
            for i_lbs, v in enumerate(verts):
                dists = torch.nn.functional.pairwise_distance(torch.tensor(v), kps)
                dists_full[i_lbs] = dists

            vert2kp_init = 1 / dists_full ** pp
            vert2kp_init = vert2kp_init.permute(1, 0)
            for i_kp, k in enumerate(kp_dict):
                idx = kp_dict[k]
                vert2kp_init[i_kp, idx] = 0
                vert2kp_init[i_kp, idx] = vert2kp_init[i_kp].max()

            vert2kp_init = safe_ln(vert2kp_init)
            if opts.learnable_kp:
                self.vert2kp = nn.Parameter(vert2kp_init)
            else:
                self.vert2kp = vert2kp_init.cuda()

        else:
            verts_full = self.get_mean_shape().shape[0]
            pp = 4
            dists_full = torch.zeros(verts.shape[0], num_kps).float()
            for i_lbs, v in enumerate(verts):
                sel = sfm_mean_shape[0]
                dists = torch.nn.functional.pairwise_distance(torch.tensor(v), torch.tensor(sel))
                dists_full[i_lbs] = dists

            vert2kp_init = 1 / dists_full ** pp
            vert2kp_init = vert2kp_init.permute(1, 0)

            vert2kp_init = torch.nn.functional.normalize(vert2kp_init, p=1)
            vert2kp_init = safe_ln(vert2kp_init)

            self.vert2kp = nn.Parameter(vert2kp_init)

        self.num_verts = num_verts
        verts_ = self.get_mean_shape()
        _, idx_pts = farthest_point_sampling(verts_.detach().cpu().numpy(), faces_np, opts.num_lbs - 1)
        idx_pts.sort()
        self.idx_pts = idx_pts
        pp = 16

        def safe_ln(x, minval=0.0000000001):
            return torch.log(torch.clamp(x, min=minval))

        import gdist
        dists_full = torch.zeros(verts.shape[0], opts.num_lbs).float()
        distance_gd = gdist.local_gdist_matrix(
            verts_.detach().cpu().numpy().astype(np.float64),
            faces.astype(np.int32)
        )
        for i_lbs, v in enumerate(verts):
            dists = distance_gd[i_lbs, idx_pts].todense()
            dists_full[i_lbs] = torch.from_numpy(dists)
        lbs = 1 / dists_full ** pp

        lbs[torch.isinf(lbs)] = 0
        max_lbs = lbs.max(dim=0)[0]
        for i_lbs, idx_pt in enumerate(idx_pts):
            lbs[idx_pt, i_lbs] = max_lbs[i_lbs]

        lbs = safe_ln(lbs)
        self.lbs = lbs
        self.lbs = nn.Parameter(self.lbs)
        self.encoder = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat)
        self.code_predictor = CodePredictor(num_lbs=opts.num_lbs, nz_feat=nz_feat, num_verts=self.num_output)
        self.camera_predictor = CameraPredictor(opts=self.opts, nz_feat=256 * 4 * 4)

        if self.pred_texture:
            mesh_template = Meshes(verts=[self.get_mean_shape().cuda()], faces=[self.faces])

            self.sdivide = SubdivideMeshes(mesh_template)
            verts_full = self.sdivide(mesh_template).verts_padded().shape[1]
            if self.symmetric_texture:
                num_faces = self.num_indept_faces + self.num_sym_faces
                num_verts = num_sym_output
                num_sym = self.num_sym
            else:
                num_faces = faces.shape[0]
                self.num_sym_faces = -1
                num_verts = self.num_verts
                num_sym = 0

            uv_sampler = compute_uvsampler(verts_np, faces_np[:num_faces], tex_size=opts.tex_size)
            # F' x T x T x 2
            uv_sampler = torch.FloatTensor(uv_sampler).cuda()
            # B x F' x T x T x 2
            uv_sampler = uv_sampler.unsqueeze(0).repeat(self.opts.batch_size, 1, 1, 1, 1)
            img_H = int(2 ** np.floor(np.log2(np.sqrt(num_faces) * opts.tex_size)))
            img_W = 2 * img_H
            print('num_verts', num_verts)
            self.texture_predictor = TexturePredictorUV(
                nz_feat, uv_sampler, opts, img_H=img_H, img_W=img_W, predict_flow=True,
                symmetric=opts.symmetric_texture, num_sym_faces=self.num_sym_faces)

    def forward(self, img):
        img_feat, res_feats = self.encoder.forward(img)
        codes_pred = self.code_predictor.forward(img_feat)
        return img_feat, codes_pred, res_feats

    def get_mean_shape(self):
        mean_shape = self.mean_v
        return mean_shape

    def get_lbs(self):
        lbs = torch.nn.functional.softmax(self.lbs, dim=0)
        return lbs
