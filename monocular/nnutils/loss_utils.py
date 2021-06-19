"""
Loss Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import geom_utils
import lpips


def iou(predict, target, eps=1e-6, reduce=True):
    '''
    Computes iou between predict and target
    '''
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    if reduce:
        return (intersect / union).sum() / intersect.nelement()
    else:
        return intersect / union


def iou_loss(predict, target, reduce=True):
    return 1 - iou(predict[:, None], target[:, None], reduce=reduce)


def quat_conj(q):
    return torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)


def quat2ang(q):
    ang = 2 * torch.acos(torch.clamp(q[:, :, 0], min=-1 + 1E-6, max=1 - 1E-6))
    ang = ang.unsqueeze(-1)
    return ang


def hamilton_product(qa, qb):
    """Multiply qa by qb.
    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


def l1_loss(predict, target, reduce=True):
    loss = F.l1_loss(predict, target, reduction='none')
    if reduce:
        return loss.mean()
    else:
        return loss.mean(list(range(1, loss.ndim)))


def template_edge_loss(meshes, template_mesh):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.
    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.
    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    temp_edges_packed = template_mesh.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    temp_verts_packed = template_mesh.verts_packed()  # (sum(V_n), 3)

    verts_edges = verts_packed[edges_packed]
    temp_verts_edges = temp_verts_packed[temp_edges_packed]
    v0, v1 = verts_edges.unbind(1)
    t_v0, t_v1 = temp_verts_edges.unbind(1)
    edge_distance = (v0 - v1).norm(dim=1, p=2) ** 2.0
    t_edge_distance = (t_v0 - t_v1).norm(dim=1, p=2) ** 2.0
    loss = (edge_distance - t_edge_distance).norm(p=2)

    return loss / N


def mask_dt_loss(proj_verts, dist_transf):
    """
    proj_verts: B x N x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Computes the distance transform at the points where vertices land.
    """
    # Reshape into B x 1 x N x 2
    sample_grid = proj_verts.unsqueeze(1)
    # B x 1 x 1 x N
    dist_transf = torch.nn.functional.grid_sample(dist_transf, sample_grid, padding_mode='border', align_corners=True)
    return dist_transf.mean()


def texture_dt_loss(texture_flow, dist_transf, vis_rend=None, cams=None, verts=None, tex_pred=None):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Similar to geom_utils.sample_textures
    But instead of sampling image, it samples dt values.
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 1 x F x T*T
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid, align_corners=True)
    return dist_transf.mean()


def locally_rigid_fn(meshes, mesh_template):
    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    mesh_dist = ((v0 - v1).norm(dim=1, p=2))
    edges_packed_t = mesh_template.edges_packed()  # (sum(E_n), 3)
    verts_packed_t = mesh_template.verts_packed()  # (sum(V_n), 3)
    verts_edges_t = verts_packed_t[edges_packed_t]
    v0_t, v1_t = verts_edges_t.unbind(1)
    mesh_template_dist = ((v0_t - v1_t).norm(dim=1, p=2))
    loss = (mesh_dist - mesh_template_dist) ** 2
    loss = loss.sum() / N
    return loss


class Locally_Rigid(nn.Module):
    def forward(self, meshes, mesh_template):
        return locally_rigid_fn(meshes, mesh_template)


def texture_dt_loss_v(texture_flow, dist_transf, vis_rend=None, cams=None, verts=None, tex_pred=None, reduce=True):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Similar to geom_utils.sample_textures
    But instead of sampling image, it samples dt values.
    """
    # Reshape into B x F x T*T x 2
    V = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, V, 1, 2)
    # B x 1 x F x T*T
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid, align_corners=True)

    if reduce:
        return dist_transf.mean()
    else:
        dist_transf = dist_transf.mean(-1).mean(-1).squeeze(1)
        return dist_transf


def texture_loss(img_pred, img_gt, mask_pred, mask_gt):
    """
    Compute L1 texture loss between masked img_pred and img_gt

    """
    mask_pred = mask_pred.unsqueeze(1)
    mask_gt = mask_gt.unsqueeze(1)
    return torch.nn.functional.l1_loss(img_pred * mask_pred, img_gt * mask_gt)


def bds_loss(verts, bds, faces, pix_to_face, reduce=True, n_samples=1000, k=1):
    """
    Compute Boundaries loss
    """
    bt, nv, _ = verts.shape
    _, H, W, _ = pix_to_face.shape
    # select up to n_samples
    indices = torch.randperm(bds.shape[1])[:n_samples]
    bds_v = bds[..., indices, :-1]
    bds_m = bds[..., indices, -1]
    fi_maps = pix_to_face[..., 0].reshape(bt, -1).detach()
    visible_vertices_ = torch.zeros(bt * nv, device=faces.device)
    # compute visible vertices
    faces_ = faces + torch.arange(faces.shape[0], device=faces.device)[:, None, None] * nv
    faces_ = faces_.reshape(-1, 3)
    fmu_ = fi_maps[fi_maps >= 0]
    fmu_ = fmu_.long()
    sel_faces = faces_[fmu_].long()
    sel_faces = sel_faces.reshape(-1).unique(dim=0).long()
    visible_vertices_.scatter_(0, sel_faces, 1)
    visible_vertices = visible_vertices_.reshape(bt, nv).detach()
    # compute euclidean distance between points on the boundary and verts
    dist = torch.cdist(bds_v, verts) ** 2
    # set non visible vertices as far away to avoid topk selection
    dist = (1 - visible_vertices[:, None]) * 1000 + visible_vertices[:, None] * dist
    # select closest vert to boundary point and compute distance
    min_dists = dist.topk(k, largest=False, sorted=False)[0]
    viz = bds_m.sum(-1)
    bds_m = bds_m[:, :, None]
    loss = (min_dists * bds_m).mean(-1).sum(-1)
    if reduce:
        return loss.mean()
    else:
        return loss


class Boundaries_Loss(nn.Module):
    def forward(self, verts, bds, faces, pix_to_face, reduce=True, n_samples=1000):
        return bds_loss(verts, bds, faces, pix_to_face, reduce=reduce, n_samples=n_samples)


def edt_loss(mask_rendered, edt, reduce=True):
    # Compute chamfer distance for points projected outside of mask
    bsize = mask_rendered.shape[0]
    mask_con_err = edt * mask_rendered[:, None]
    loss = mask_con_err.reshape(bsize, -1).mean(-1)
    if reduce:
        return loss.mean()
    else:
        return loss


def hinge_loss(loss, margin):
    # Only penalize if loss > margin
    zeros = torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=False)
    return torch.max(loss - margin, zeros)


def quat_loss_geodesic(q1, q2):
    '''
    Geodesic rotation loss.

    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : N x 1
    '''
    q1 = torch.unsqueeze(q1, 1)
    q2 = torch.unsqueeze(q2, 1)
    q2_conj = torch.cat([q2[:, :, [0]], -1 * q2[:, :, 1:4]], dim=-1)
    q_rel = geom_utils.hamilton_product(q1, q2_conj)
    q_loss = 1 - torch.abs(q_rel[:, :, 0])
    return q_loss


def camera_loss(cam_pred, cam_gt, margin):
    rot_pred = cam_pred[:, -4:]
    rot_gt = cam_gt[:, -4:]

    rot_loss = hinge_loss(quat_loss_geodesic(rot_pred, rot_gt), margin)
    # Scale and trans.
    st_loss = (cam_pred[:, :3] - cam_gt[:, :3]) ** 2
    st_loss = hinge_loss(st_loss.view(-1), margin)

    return rot_loss.mean() + st_loss.mean()


def triangle_loss(verts, edge2verts):
    """
    Encourages dihedral angle to be 180 degrees.

    Args:
        verts: B X N X 3
        edge2verts: B X E X 4
    Returns:
        loss : scalar
    """
    indices_repeat = torch.stack([edge2verts, edge2verts, edge2verts], dim=2)  # B X E X 3 X 4

    verts_A = torch.gather(verts, 1, indices_repeat[:, :, :, 0])
    verts_B = torch.gather(verts, 1, indices_repeat[:, :, :, 1])
    verts_C = torch.gather(verts, 1, indices_repeat[:, :, :, 2])
    verts_D = torch.gather(verts, 1, indices_repeat[:, :, :, 3])

    # n1 = cross(ad, ab)
    # n2 = cross(ab, ac)
    n1 = geom_utils.cross_product(verts_D - verts_A, verts_B - verts_A)
    n2 = geom_utils.cross_product(verts_B - verts_A, verts_C - verts_A)

    n1 = torch.nn.functional.normalize(n1, dim=2)
    n2 = torch.nn.functional.normalize(n2, dim=2)

    dot_p = (n1 * n2).sum(2)
    loss = ((1 - dot_p) ** 2).mean()
    return loss


def deform_l2reg(V):
    """
    l2 norm on V = B x N x 3
    """
    V = V.view(-1, V.size(2))
    return torch.mean(torch.norm(V, p=2, dim=1))


def entropy_loss(A):
    """
    Input is K x N
    Each column is a prob of vertices being the one for k-th keypoint.
    We want this to be sparse = low entropy.
    """
    entropy = -torch.sum(A * torch.log(A), 1)
    # Return avg entropy over 
    return torch.mean(entropy)


def kp_l2_loss(kp_pred, kp_gt, reduction='mean'):
    """
    L2 loss between visible keypoints.

    \Sum_i [0.5 * vis[i] * (kp_gt[i] - kp_pred[i])^2] / (|vis|)
    """
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss(reduction='none')

    vis = (kp_gt[:, :, 2] > 0).float()
    loss = criterion(kp_pred, kp_gt[:, :, :2]).sum(-1) * vis
    loss = loss.mean(-1) / (vis.mean(-1) + 1e-4)
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss


class PerceptualTextureLoss_v2(object):
    def __init__(self, net='alex', lpips_f=False):
        self.loss_fn_alex = lpips.LPIPS(net=net, lpips=lpips_f, spatial=True).cuda()
        self.loss_fn_alex = nn.DataParallel(self.loss_fn_alex)

    def __call__(self, img_pred, img_gt, mask_pred, mask_gt, reduce=True):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
        mask_pred, mask_gt: B x H x W
        """
        mask_gt = mask_gt.unsqueeze(1)
        # Only use mask_gt..
        pred = img_pred * mask_gt
        target = img_gt * mask_gt
        pred = 2 * pred - 1
        target = 2 * target - 1
        dist = self.loss_fn_alex(pred, target)
        dist = dist * mask_gt
        dist = dist.mean(-2, keepdim=True).mean(-1, keepdim=True)
        dist = dist.squeeze(-1).squeeze(-1).squeeze(-1)
        if reduce:
            return dist.mean()
        else:
            return dist


class TexCycle(nn.Module):
    def __init__(self, im_size=256, nf=1280, eps=1e-12):
        super(TexCycle, self).__init__()

    def forward(self, flow, prob, aggr_info):
        """
        INPUTS:
         - flow: learned texture flow (nb * nf * nr * nr * 2)
         - prob: affinity between image & mesh by renderer (nb * nf * 2)
         - aggr_info: provide information about visible faces.
        OUTPUTS:
         - texture cycle loss
        IDEA:
         - make averaged coords of projected face equals to predicted flow
        """
        nb, nf, nr, _, _ = flow.size()

        flow_grid = flow.view(nb, nf, -1, 2)
        avg_flow = torch.mean(flow_grid, dim=2)

        # mask: nb x nf x 2
        # only rows correspond to visible faces are set to 1
        mask = torch.zeros(avg_flow.size())
        for cnt in range(nb):
            fids = torch.unique(aggr_info[cnt]).long()
            mask[cnt, fids, :] = 1

        mask = mask.cuda()
        loss = torch.nn.MSELoss()(avg_flow * mask, prob * mask)
        # second term for visilization purpose
        return loss, avg_flow[0, 0:10, :]


def optical_flow_loss(meshes, faces, cams, flows, renderer, pix_to_face, reduce=True):
    H, W = flows.shape[2:4]
    b, t, nv, _ = meshes.shape
    bt = b * t
    predicted_points = renderer.proj_fn(meshes.reshape(b * t, nv, -1), cams.reshape(b * t, -1))

    # compute visible vertices using z-buffer
    with torch.no_grad():
        if pix_to_face is None:
            pix_to_face = renderer(predicted_points.reshape(bt, nv, 3), faces.reshape(bt, -1, 3))
            pix_to_face = pix_to_face.long()
        else:
            pix_to_face = pix_to_face[..., :1].long()
        fi_maps = pix_to_face.reshape(bt, -1)
        faces = faces.reshape(bt, faces.shape[2], 3).long()

        visible_vertices_ = torch.zeros(bt * nv, device=faces.device)
        faces_ = faces + torch.arange(faces.shape[0], device=faces.device)[:, None, None] * nv
        faces_ = faces_.reshape(-1, 3)
        fmu_ = fi_maps[fi_maps >= 0]
        fmu_ = fmu_.long()
        sel_faces = faces_[fmu_].long()
        sel_faces = sel_faces.reshape(-1).unique(dim=0).long()
        visible_vertices_.scatter_(0, sel_faces, 1)
        visible_vertices = visible_vertices_.reshape(b, t, nv)

    predicted_points = predicted_points.reshape(b, t, nv, -1)
    predicted_points = predicted_points[:, :, :, :2]
    predicted_points = predicted_points.reshape(b * t, nv, -1)[:, :, None, :2]

    flows = flows.reshape(bt, flows.shape[2], flows.shape[3], -1).permute(0, 3, 1, 2)
    samples_ofs_gt = F.grid_sample(flows, predicted_points, align_corners=False, mode='nearest')
    samples_ofs_gt = samples_ofs_gt[..., 0].permute(0, 2, 1)
    samples_ofs_gt = samples_ofs_gt.reshape(b, t, samples_ofs_gt.shape[1], samples_ofs_gt.shape[2])

    predicted_points = predicted_points.reshape(b, t, nv, -1)
    predicted_points_ = W * (predicted_points + 1) / 2

    current_frame = predicted_points_[:, :-1]
    next_frame = predicted_points_[:, 1:]
    of_pred = current_frame - next_frame  # how much next_frame should move to much current_frame nf + of = cf

    # drop flow of last frame
    visible_vertices = (samples_ofs_gt.abs().sum(-1) != 0).bool() * visible_vertices.bool()  # logical end
    visible_vertices = visible_vertices.float()
    visible_vertices = visible_vertices[:, 1:].detach()
    samples_ofs_gt = visible_vertices[..., None] * samples_ofs_gt[:, 1:]  # samples_ofs_gt[:, :-1]

    of_pred = visible_vertices[..., None] * of_pred
    loss = torch.norm(samples_ofs_gt[..., 0] - of_pred[..., 0], p=1, dim=-1) + torch.norm(
        samples_ofs_gt[..., 1] - of_pred[..., 1], p=1, dim=-1)
    loss = loss / H / (visible_vertices.sum(-1) + 1)  # +1 to avoid NaN

    if reduce:
        loss = loss.sum()
    return loss, of_pred, visible_vertices, predicted_points, samples_ofs_gt


class Optical_Flow_Loss(nn.Module):
    def forward(self, meshes, faces, cams, flows, renderer, pix_to_face, reduce=True):
        return optical_flow_loss(meshes, faces, cams, flows, renderer, pix_to_face, reduce=reduce)
