## Setup annotations and data directories


### Download dataset pretrained models
Download from [here](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabfko_ucl_ac_uk/ESr8Bb5NiOBJos8ivKR3mIMBaNs4xQbJBgdVwKZHxOMpdA?e=Y2EaPo)

```
cd monocular/misc
7z t cachedir.7z

```
CUB dataset can be downloaded from [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). Unzip the file in misc/, else set the --cub_dir argument appropriately.

### Training and Testing on CUB dataset
#####  Train Birds:
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --name=bird_net --num_lbs 32 --symmetric_texture=False --nz_feat 256 --cam_loss_wt 2. --mask_loss_wt 2. --symmetric=False --print_freq 100 --display_freq 100 --boundaries_reg_wt 1. --bdt_reg_wt 0.1 --edt_reg_wt 0.1  --tex_size 6 --save_epoch_freq 10 --kp_loss_wt 50. --tex_loss_wt 1.
```



##### Evaluate Birds:

Bird with 64 handles:
```
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --split test  --name bird_net_64 --num_train_epoch 200   --num_lbs 64 --nz_feat 256  --symmetric_texture=False --symmetric=False --cub_dir misc/CUB_200_2011/  --batch_size 12
```

Bird with 32 handles:
```
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --split test  --name bird_net_32 --num_train_epoch 200   --num_lbs 32 --nz_feat 256  --symmetric_texture=False --symmetric=False --cub_dir misc/CUB_200_2011/  --batch_size 12
```
