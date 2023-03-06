## DVA 

An open-source implementation of the SIGGRAPH 2022 paper [Drivable Volumetric Avatars using Texel-Aligned Features](https://arxiv.org/pdf/TODO)

### Data and Assets

This repository contains an implementation that is using
[ZJU-Lightstage](https://chingswy.github.io/Dataset-Demo/) and 
[SMPLX](https://smpl-x.is.tue.mpg.de/). 

Please refer to the corresponding pages on details on how to download
those, and please make sure to reference those works appropriately.

You might need to update provided configuration files to set the
paths.

### Running Training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 dva/scripts/train.py configs/sample_386.yml
```

You this will create a folder with checkpoints, config and a monitoring image:
```bash
data/zju_lightstage/386/training-logs/cond_model_mvp
```

Please note that this version uses an open-source implementation of `mvp`, so results
may differ slightly with the paper.

### Requirements and building

```
pytorch
pytorch3d
omegaconf
opencv
easymocap (https://github.com/zju3dv/EasyMocap)
mvp (https://github.com/facebookresearch/mvp)
```

Building raymarching extensions
```
cd dva
git clone https://github.com/facebookresearch/mvp
cd mvp/extensions/mvpraymarch
make -j4
```

Installing easymocap:

```
git clone https://github.com/zju3dv/EasyMocap
cd EasyMocap
pip install --user .
```

### License

See LICENSE.


### Citing 

If you use this repository, consider citing:

```
@inproceedings{remelli2022drivable,
  title={Drivable volumetric avatars using texel-aligned features},
  author={Remelli, Edoardo and Bagautdinov, Timur and Saito, Shunsuke and Wu, Chenglei and Simon, Tomas and Wei, Shih-En and Guo, Kaiwen and Cao, Zhe and Prada, Fabian and Saragih, Jason and others},
  booktitle={ACM SIGGRAPH 2022 Conference Proceedings},
  pages={1--9},
  year={2022}
}
```