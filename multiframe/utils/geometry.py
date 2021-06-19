"""
Geometry stuff.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def triangle_direction_intersection(tri, trg):
    '''
    Finds where an origin-centered ray going in direction trg intersects a triangle.
    Args:
        tri: 3 X 3 vertex locations. tri[0, :] is 0th vertex.
    Returns:
        alpha, beta, gamma
    '''
    p0 = np.copy(tri[0, :])
    # Don't normalize
    d1 = np.copy(tri[1, :]) - p0;
    d2 = np.copy(tri[2, :]) - p0;
    d = trg / np.linalg.norm(trg)

    mat = np.stack([d1, d2, d], axis=1)

    try:
        inv_mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return False, 0

    # inv_mat = np.linalg.inv(mat)

    a_b_mg = -1 * np.matmul(inv_mat, p0)
    is_valid = (a_b_mg[0] >= 0) and (a_b_mg[1] >= 0) and ((a_b_mg[0] + a_b_mg[1]) <= 1) and (a_b_mg[2] < 0)
    if is_valid:
        return True, -a_b_mg[2] * d
    else:
        return False, 0


def project_verts_on_mesh(verts, mesh_verts, mesh_faces):
    verts_out = np.copy(verts)
    for nv in range(verts.shape[0]):
        max_norm = 0
        vert = np.copy(verts_out[nv, :])
        for f in range(mesh_faces.shape[0]):
            face = mesh_faces[f]
            tri = mesh_verts[face, :]
            # is_v=True if it does intersect and returns the point
            is_v, pt = triangle_direction_intersection(tri, vert)
            # Take the furthest away intersection point
            if is_v and np.linalg.norm(pt) > max_norm:
                max_norm = np.linalg.norm(pt)
                verts_out[nv, :] = pt

    return verts_out


import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


def fit_verts_to_mesh(verts, faces, trg_verts, trg_faces):
    verts, faces = torch.from_numpy(verts).float().cuda(), torch.from_numpy(faces).float().cuda()
    trg_verts, trg_faces = torch.from_numpy(trg_verts).float().cuda(), torch.from_numpy(trg_faces).float().cuda()

    center = trg_verts.mean(0)
    trg_verts = trg_verts - center
    scale = max(trg_verts.abs().max(0)[0])
    trg_verts = trg_verts / scale

    trg_mesh = Meshes(verts=[trg_verts.cuda()], faces=[trg_faces.cuda()])

    deform_verts = torch.full(verts.shape, 0.0, device=verts.device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=1., momentum=0.9)
    # Number of optimization steps
    Niter = 2000
    # Weight for the chamfer loss
    w_chamfer = 1.0
    # Weight for mesh edge loss
    w_edge = 1.0
    # Weight for mesh normal consistency
    w_normal = 0.01
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1
    # Plot period for the losses
    loop = range(Niter)

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()

        # Deform the mesh
        new_src_mesh = Meshes(verts=[verts + deform_verts], faces=[faces])
        # src_mesh.offset_verts(deform_verts)

        # We sample 5k points from the surface of each mesh
        sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        sample_src = sample_points_from_meshes(new_src_mesh, 5000)

        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_src_mesh)

        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_src_mesh)

        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="cot")
        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
        if i % 500 == 0:
            print('Fitting sphere', i, loss.item(), loss_chamfer, loss_edge, loss_normal, loss_laplacian)
        # Optimization step
        loss.backward()
        optimizer.step()

    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    # Scale normalize back to the original target size
    final_verts = final_verts * scale + center

    return final_verts.cpu().detach().numpy()
