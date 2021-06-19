from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import torch
import tqdm
from nnutils import geom_utils
from pytorch3d.renderer import Textures
# rendering components
from pytorch3d.renderer import (
    look_at_view_transform, RasterizationSettings, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, DirectionalLights,
    SfMOrthographicCameras
)
from pytorch3d.renderer.mesh import TexturesAtlas
# 3D transformations functions
from pytorch3d.renderer.mesh.shader import SoftPhongShader
# import soft_renderer as sr
from pytorch3d.structures import Meshes
from torch import nn


# from neural_renderer import Renderer as NMR


#############
### Utils ###
#############
def convert_as(src, trg):
    return src.to(trg.device).type_as(trg)


class MeshRenderer(nn.Module):

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments


class NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """

    def __init__(self, img_size=256):
        super(NeuralRenderer, self).__init__()
        self.img_size = img_size
        device = 'cuda'
        # Initialize an OpenGL perspective camera.
        self.cameras = SfMOrthographicCameras()

        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=0)

        self.sil_raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * self.blend_params.sigma,
            faces_per_pixel=10, bin_size=None
        )

        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.sil_raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=self.blend_params)
        )

        self.blend_params_tex = BlendParams(background_color=0)

        self.tex_raster_settings = RasterizationSettings(
            image_size=img_size, blur_radius=0., faces_per_pixel=1,
            clip_barycentric_coords=True)

        # We can add a point light in front of the object.
        self.lights = DirectionalLights(ambient_color=((1., 1., 1.),),
                                        diffuse_color=((0., 0., 0.),),
                                        specular_color=((0., 0., 0.),),
                                        direction=((0., 1., 0.),))

        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.tex_raster_settings
            ),
            shader=SoftPhongShader(device=device, cameras=self.cameras, lights=self.lights,
                                   blend_params=self.blend_params)
        )

        # Make it a bit brighter for vis
        self.raster_settings_of = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings_of
        )

        self.proj_fn = geom_utils.orthographic_proj_withz

        self.offset_z = 0.

    def ambient_light_only(self):
        return

    def set_bgcolor(self, color):
        return

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def rasterize_of(self, verts, faces, R, T):
        mesh = Meshes(
            verts=verts,
            faces=faces,
        )
        cameras = SfMOrthographicCameras(device=verts.device)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings_of
        )
        return rasterizer(mesh, R=R, T=T)

    def forward(self, vertices, faces, cams, textures=None, atlas=True):
        eye = torch.tensor([[0, 0, -2.732]], device=vertices.device)
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z)
        vs = verts
        vs[:, :, 1] *= -1
        R, T = look_at_view_transform(eye=eye, device=vertices.device)
        R[:, 0, 0] *= -1
        if textures is None:
            self.mask_only = True
            mesh = Meshes(verts=vs, faces=faces)
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=0)
            cameras = SfMOrthographicCameras(device=vs.device)
            sil_raster_settings = RasterizationSettings(
                image_size=self.img_size,
                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                faces_per_pixel=20, bin_size=None
            )

            silhouette_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=sil_raster_settings
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params)
            )

            masks, fragments = silhouette_renderer(meshes_world=mesh, R=R, T=T)
            masks = masks[..., -1]
            pix_to_face = fragments.pix_to_face
            return masks, pix_to_face
        else:
            self.mask_only = False
            if atlas:
                textures_obj = TexturesAtlas(atlas=textures.to(vs.device))
            else:
                if textures.ndim == 2: textures = textures[None]
                textures_obj = Textures(verts_rgb=textures)

            mesh = Meshes(
                verts=vs,
                faces=faces, textures=textures_obj
            )
            cameras = SfMOrthographicCameras(device=vs.device)
            phong_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=self.tex_raster_settings
                ),
                shader=SoftPhongShader(device=vs.device, cameras=cameras, lights=self.lights.clone().to(vs.device),
                                       blend_params=self.blend_params_tex)
            )

            imgs, fragments = phong_renderer(meshes_world=mesh, R=R, T=T)
            pix_to_face = fragments.pix_to_face
            sil = imgs[..., -1]
            imgs = imgs[..., :-1]
            imgs = imgs.permute(0, 3, 1, 2)
            return imgs, sil, pix_to_face


class OF_NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """

    def __init__(self, img_size=256):
        super(OF_NeuralRenderer, self).__init__()
        self.img_size = img_size
        self.raster_settings_of = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1)
        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, verts, faces):
        eye = torch.Tensor([[0, 0, -2.732]]).to(verts.device)
        R, T = look_at_view_transform(eye=eye)
        R[:, 0, 0] *= -1
        mesh = Meshes(
            verts=verts,
            faces=faces,
        )
        cameras = SfMOrthographicCameras(device=verts.device)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings_of
        )
        fragments = rasterizer(mesh, R=R.to(verts.device), T=T.to(verts.device))
        return fragments.pix_to_face


############# TESTS #############
def exec_main():
    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    vertices, faces = neural_renderer.load_obj(obj_file)

    renderer = NMR()
    renderer.to_gpu(device=0)

    masks = renderer.forward_mask(vertices[None, :, :], faces[None, :, :])
    print(np.sum(masks))
    print(masks.shape)

    grad_masks = masks * 0 + 1
    vert_grad = renderer.backward_mask(grad_masks)
    print(np.sum(vert_grad))
    print(vert_grad.shape)

    # Torch API
    mask_renderer = NeuralRenderer()
    vertices_var = torch.autograd.Variable(torch.from_numpy(vertices[None, :, :]).cuda(device=0), requires_grad=True)
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))

    for ix in range(100):
        masks_torch = mask_renderer.forward(vertices_var, faces_var)
        vertices_var.grad = None
        masks_torch.backward(torch.from_numpy(grad_masks).cuda(device=0))

    print(torch.sum(masks_torch))
    print(masks_torch.shape)
    print(torch.sum(vertices_var.grad))


# @DeprecationWarning
def teapot_deform_test():
    #
    obj_file = 'birds3d/external/neural_renderer/examples/data/teapot.obj'
    img_file = 'birds3d/external/neural_renderer/examples/data/example2_ref.png'
    img_save_dir = 'birds3d/cachedir/nmr/'

    vertices, faces = neural_renderer.load_obj(obj_file)
    from skimage.io import imread
    image_ref = imread(img_file).astype('float32').mean(-1) / 255.
    image_ref = torch.autograd.Variable(torch.Tensor(image_ref[None, :, :]).cuda(device=0))

    mask_renderer = NeuralRenderer()
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
    cams = np.array([1., 0, 0, 1, 0, 0, 0], dtype=np.float32)
    cams_var = torch.autograd.Variable(torch.from_numpy(cams[None, :]).cuda(device=0))

    class TeapotModel(torch.nn.Module):
        def __init__(self):
            super(TeapotModel, self).__init__()
            vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
            self.vertices_var = torch.nn.Parameter(vertices_var)

        def forward(self):
            return mask_renderer.forward(self.vertices_var, faces_var, cams_var)

    opt_model = TeapotModel()

    optimizer = torch.optim.Adam(opt_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    # from time import time
    loop = tqdm.tqdm(range(300))
    print('Optimizing Vertices: ')
    for ix in loop:
        # t0 = time()
        optimizer.zero_grad()
        masks_pred = opt_model()
        loss = torch.nn.MSELoss()(masks_pred, image_ref)
        loss.backward()
        if ix % 20 == 0:
            im_rendered = masks_pred.data[0, :, :]
            scipy.misc.imsave(img_save_dir + 'iter_{}.png'.format(ix), im_rendered)
        optimizer.step()
        # t1 = time()
        # print('one step %g sec' % (t1-t0))


if __name__ == '__main__':
    # exec_main()
    teapot_deform_test()
