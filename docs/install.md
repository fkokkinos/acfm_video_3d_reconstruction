# Installation Instruction

## Clone code
```
git clone https://github.com/fkokkinos/acfm_video_3d_reconstruction
cd acfm_video_3d_reconstruction/
```

## Setup Conda Env
* Create Conda environment
```
conda create -n acfm
conda activate acfm
conda env update --file environment.yml
```

* Install other packages and dependencies

  Refer [here](https://pytorch.org/get-started/previous-versions/) for pytorch installation
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```

* Setup Pytorch3D

  To setup Pytorch3D follow the instructions located [here](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)

* Install correlation package for optical flow
```
cd multiframe/data/optical_flow/model/correlation_package/
python setup.py install
```
