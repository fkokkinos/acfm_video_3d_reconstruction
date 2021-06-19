# Learning monocular 3D reconstruction of articulated categories from motion (ACFM)
[Filippos Kokkinos](https://fkokkinos.github.io/)  and [Iasonas Kokkinos](http://www0.cs.ucl.ac.uk/staff/I.Kokkinos/)

[Paper](https://arxiv.org/abs/2103.16352)
[Project Page](https://fkokkinos.github.io/video_3d_reconstruction/)

<img src="https://fkokkinos.github.io/video_3d_reconstruction/resources/images/teaser2.jpg" width="50%">

## Requirements
* Python 3.6+
* PyTorch 1.7.0
* PyTorch3D 0.3.0, there are some breaking changes with more recent versions.
* cuda 11.0

For setup and installation refer to [docs/install.md](docs/install.md) instructions.


## Setup Evaluation and Training
We have separate folder for video and monocular based training.

For ease of access we provide python scripts that can generate slurm scripts that can be used to generate the results in the paper.

* Downloading pre-trained model and annotations. Follow setup instructions [here](docs/setup.md)

* Training from scratch.  Follow setup instructions [here](docs/setup.md)


## Citation
If you find the code useful for your research, please consider citing:-
```
@InProceedings{Kokkinos_2021_CVPR,
               author = {Kokkinos, Filippos and Kokkinos, Iasonas},
               title = {Learning monocular 3D reconstruction of articulated categories from motion},
               booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
               month = {June},
               year = {2021}
}
```

## Acknowledgements
This code repository uses code from [CMR](https://github.com/akanazawa/cmr/) and [CSM](https://github.com/nileshkulkarni/csm/) repos.

## Contact
For questions feel free to contact me at filippos.kokkinos[at]ucl.ac.uk .
