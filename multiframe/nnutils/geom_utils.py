"""
Utils related to geometry like projection,,
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def sample_textures(texture_flow, images):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x F x T x T x 3
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid, align_corners=True)
    # B x 3 x F x T x T
    samples = samples.view(-1, 3, F, T, T)
    # B x F x T x T x 3
    return samples.permute(0, 2, 3, 4, 1)


def sample_textures_v(texture_flow, images):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x F x T x T x 3
    """
    # Reshape into B x F x T*T x 2
    V = texture_flow.size(-2)
    flow_grid = texture_flow.view(-1, V, 1, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid, align_corners=True)
    samples = samples.squeeze(-1).permute(0, 2, 1)
    return samples  # .permute(0, 2, 3, 4, 1)


def orthographic_proj(X, cam):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    return scale * X_rot[:, :, :2] + trans


def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    proj = scale * X_rot

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)


def cross_product(qa, qb):
    """Cross product of va by vb.

    Args:
        qa: B X N X 3 vectors
        qb: B X N X 3 vectors
    Returns:
        q_mult: B X N X 3 vectors
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]

    # See https://en.wikipedia.org/wiki/Cross_product
    q_mult_0 = qa_1 * qb_2 - qa_2 * qb_1
    q_mult_1 = qa_2 * qb_0 - qa_0 * qb_2
    q_mult_2 = qa_0 * qb_1 - qa_1 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2], dim=-1)


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


def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]] * 0 + 1
    q = torch.unsqueeze(q, 1) * ones_x

    q_conj = torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)
    X = torch.cat([X[:, :, [0]] * 0, X], dim=-1)

    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]


import torch


def mesh_laplacian(meshes, method: str = "uniform"):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent cuvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
    for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
    vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
    the surface normal, while the curvature variant LckV[i] scales the normals
    by the discrete mean curvature. For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.

    .. code-block:: python

               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.

    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.

    .. code-block:: python

               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have

        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

        Putting these together, we get:

        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH


    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed().to_dense()
        elif method in ["cot"]:
            L, inv_areas = laplacian_cot(meshes)
            norm_w = torch.sparse.sum(L, dim=1).to_dense()
            M = torch.diag(norm_w)
            L = L.to_dense() - M

    return L


def laplacian_cot(meshes):
    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.

    Args:
        meshes: Meshes object with a batch of meshes.
    Returns:
        2-element tuple containing
        - **L**: FloatTensor of shape (V,V) for the Laplacian matrix (V = sum(V_n))
           Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
           See the description above for more clarity.
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
           face areas containing each vertex
    """
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    # V = sum(V_n), F = sum(F_n)
    V, F = verts_packed.shape[0], faces_packed.shape[0]

    face_verts = verts_packed[faces_packed]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces_packed[:, [1, 2, 0]]
    jj = faces_packed[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # For each vertex, compute the sum of areas for triangles containing it.
    idx = faces_packed.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=meshes.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas
