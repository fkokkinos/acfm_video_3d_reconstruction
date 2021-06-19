
File:
load_and_process.py: Loads model and weights and process a single batch. Output is a dict that contains the following:

Outputs:
Dict with elements (['lbs', 'mean_shape', 'faces', 'delta_v_res', 'kp_pred', 'verts', 'kp_verts', 'cam_pred', 'mask_pred'])

lbs: matrix related to laplacian deformation
mean_shape: the template (642 vertices)
faces: template faces
delta_v_res: Delta_h from paper
kp_pred: 2d key points of mesh (projected using cam_pred)
verts: deformed mesh
kp_verts: 3D keypoints of mesh
cam_pred: predicted camera in R^7 -> [scale, tx, ty, quats]
mask_pred: rendered mesh


Commands:
1) Model with 64 Handles
python3 load_and_process.py --split test  --name bird_net_64 --num_train_epoch 330   --num_lbs 64 --nz_feat 256  --symmetric_texture=False --symmetric=False --cub_dir ../cmr/misc/CUB_200_2011/  --batch_size 12
PCK: 0.915 (best score)

2)Model with 8 Handles
python3 load_and_process.py --split test  --name bird_net_8 --num_train_epoch 140   --num_lbs 8 --nz_feat 256  --symmetric_texture=False --symmetric=False --cub_dir ../cmr/misc/CUB_200_2011/  --batch_size 12
PCK: 0.846

3)Model with 32 Handles
python3 load_and_process.py --split test  --name bird_net_32 --num_train_epoch 130   --num_lbs 32 --nz_feat 256  --symmetric_texture=False --symmetric=False --cub_dir ../cmr/misc/CUB_200_2011/  --batch_size 12
PCK: 0.897


