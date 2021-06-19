## Setup annotations and data directories


### Download dataset pretrained models
Download from [here](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabfko_ucl_ac_uk/EfZLL9d51zNAiH9DeQcNjOYBdqQyauQc9sctFC49vCyOBw?e=xKUITN)

```
cd multiframe/misc
7z t cachedir.7z
```

Video data for each category where created using TigDog and YoutubeVIS datasets. We provide them in a pickled form for ease of use [here](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabfko_ucl_ac_uk/Eqm5tnypzRNFlhR_BpV4XIsBzKr6xQvaNxCehBkxOkhjkw?e=h0lcnN) . Please point the arguments --root_dir and --root_dir_yt to the appropriate folder.

### Training using the Video Datasets
For training the following commands can be used for all classes.

#####  Train Horses w/ KP:
```
CUDA_VISIBLE_DEVICES=0 python -B main.py --name=horse_net_kp --category horse --display_port 8097 --batch_size=12 --learning_rate 1e-4 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --symmetric=False --mesh_dir meshes/horse_aligned.obj --rigid_wt 10. --root_dir ~/data/TigDog_new_wnrsfm/ --tmp_dir tmp_horse_sfm/ --of_loss_wt 0.1 --mask_loss_wt 2. --boundaries_reg_wt 1. --bdt_reg_wt 0.05 --edt_reg_wt 0.1 --kp_loss_wt 10. --deform_reg_wt 1e-4  --display_freq 50 --print_freq 50  --tex_loss_wt 1. --init_camera_emb --optimize_deform
```

#####  Train Horses w/o KP:
We recommend multi-gpu training for experiment without keypoints.
```
CUDA_VISIBLE_DEVICES=0,1 python -B main.py --name=horse_net_nokp --category horse --display_port 8097 --batch_size=8 --learning_rate 1e-4 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --num_guesses 6 --use_gtpose=False --symmetric=False --mesh_dir meshes/horse.obj  --root_dir ~/data/TigDog_new_wnrsfm/ --tmp_dir tmp_horsenokp/ --kp_loss_wt 0. --of_loss_wt 0.1 --mask_loss_wt 2. --boundaries_reg_wt 1. --bdt_reg_wt 0.1 --edt_reg_wt 0.5  --deform_reg_wt 0.0001  --rigid_wt 10. --display_freq 100  --drop_hypothesis  --print_freq 100  --cam_loss_wt 2.  --scale_lr_decay 0.1  --root_dir_yt ~/data/youtube_vis/pkls/ --expand_ytvis=True  --save_epoch_freq 10 --num_reps 20 --tex_num_reps 20 --tex_loss_wt 1. --az_el_cam True --scale_bias=0.9 --az_euler_range 360 --num_kps 19 --triangle_reg_wt 0.01  --optimize_deform --warmup --texture_warmup --scale_mesh=True
```


### Evaluate Quadrupeds:
#####  Evaluate Horses w/ KP:
```
CUDA_VISIBLE_DEVICES=0 python -B -m benchmark.evaluate --name=horse_net_kp --category horse --batch_size=12 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --symmetric=False --mesh_dir meshes/horse_aligned.obj --root_dir ~/data/TigDog_new_wnrsfm_new/ --tmp_dir tmp_horse_sfm_eval/  --num_train_epoch 200 --v2_crop=False --tight_bboxes=False
```
#####  Evaluate Horses w/ KP + post-processing optimization
```
CUDA_VISIBLE_DEVICES=0 python -B -m benchmark.evaluate --name=horse_net_kp --category horse --batch_size=12 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --symmetric=False --mesh_dir meshes/horse_aligned.obj --root_dir ~/data/TigDog_new_wnrsfm_new/ --tmp_dir tmp_horse_sfm_eval/  --num_train_epoch 200 --v2_crop=False --tight_bboxes=False --num_frames 2 --optimize --mask_loss_wt 2. --of_loss_wt 0.01 --num_optim_iter 20   
```

#####  Evaluate Horses w/o KP:
```
CUDA_VISIBLE_DEVICES=0 python -B -m benchmark.evaluate --name=horse_net_nokp --category horse --batch_size=12 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --symmetric=False --mesh_dir meshes/horse.obj --root_dir ~/data/TigDog_new_wnrsfm_new/ --tmp_dir tmp_horse_sfm_eval/  --num_train_epoch 200 --num_guesses 6 --num_frames 1 --scale_template True  --scale_lr 0.1
```

#####  Evaluate Horses w/o KP + post-processing optimization
```
CUDA_VISIBLE_DEVICES=0 python -B -m benchmark.evaluate --name=horse_net_nokp --category horse --batch_size=12 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --symmetric=False --mesh_dir meshes/horse.obj --root_dir ~/data/TigDog_new_wnrsfm_new/ --tmp_dir tmp_horse_sfm_eval/  --num_train_epoch 200 --num_guesses 6 --num_frames 2 --scale_template True  --scale_lr 0.1 --optimize --mask_loss_wt 2.--num_optim_iter 20
```
#####  Evaluate Tigers w/ KP:
```
CUDA_VISIBLE_DEVICES=0 python -B -m benchmark.evaluate --name=tiger_net_kp --category tiger --batch_size=12 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --symmetric=False --mesh_dir meshes/tiger_aligned.obj --root_dir ~/data/TigDog_new_wnrsfm_new/ --tmp_dir tmp_tiger_sfm_eval/  --num_train_epoch 200 --v2_crop=False --tight_bboxes=False --kp_dict  meshes/tiger_kp_dictionary.pkl
```

#####  Evaluate Tigers w/ KP + post-processing optimization
```
CUDA_VISIBLE_DEVICES=2 python -B -m benchmark.evaluate --name=tiger_net_kp --category tiger --batch_size=12 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --symmetric=False --mesh_dir meshes/tiger_aligned.obj --root_dir ~/data/TigDog_new_wnrsfm_new/ --tmp_dir tmp_tiger_sfm_eval/  --num_train_epoch 200 --v2_crop=False --tight_bboxes=False --kp_dict  meshes/tiger_kp_dictionary.pkl  --optimize --optimize_camera --num_optim_iter 50 --mask_loss_wt 2.
```
#####  Evaluate Tigers w/o KP:

```
CUDA_VISIBLE_DEVICES=0 python -B -m benchmark.evaluate --name=tiger_net_nokp --category tiger --batch_size=12 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --symmetric=False --mesh_dir meshes/tiger.obj --root_dir ~/data/TigDog_new_wnrsfm_new/ --tmp_dir tmp_tiger_sfm_eval/  --num_train_epoch 200 --num_guesses 6 --kp_dict meshes/tiger_kp_dictionary.pkl --scale_bias=0.9  --scale_lr_decay 0.1 --v2_crop=False --tight_bboxes=False --num_frames 1 --scale_template True  --scale_lr 0.1
```
#####  Evaluate Tigers w/o KP + post-processing optimization
```
CUDA_VISIBLE_DEVICES=2 python -B -m benchmark.evaluate --name=tiger_net_nokp --category tiger --batch_size=12 --num_lbs 16 --nz_feat 256 --symmetric_texture=False --symmetric=False --mesh_dir meshes/tiger.obj --root_dir ~/data/TigDog_new_wnrsfm_new/ --tmp_dir tmp_tiger_sfm_eval/  --num_train_epoch 200 --num_guesses 6 --kp_dict meshes/tiger_kp_dictionary.pkl --scale_bias=0.9  --scale_lr_decay 0.1 --v2_crop=False --tight_bboxes=False --num_frames 2 --scale_template True  --scale_lr 0.1  --optimize --optimize_camera --mask_loss_wt 2.
```

We provide pre-trained models for 9 classes like giraffes, bears, foxes, etc.
