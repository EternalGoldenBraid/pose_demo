## Setup
Please start by installing [Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) 
with Pyhton3.9 or above.

### Conda/Mamba

mamba env create --file environment.yml

In conda env install pip and with pip install:

[Pyrealsense](https://pypi.org/project/pyrealsense/)
- pip install pyrealsense2==2.50.0.3812

[Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
- pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

or build from source :)

# Pretrained modules
`mkdir checkpoints; cd checkpoints`
`wget https://tu-dortmund.sciebo.de/s/ISdLcDMduHeW1ay/download  -O FAT_trained_Ml2R_bin_fine_tuned.pth`
`wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aXkYOpvka5VAPYUYuHaCMp0nIvzxhW9X' \
-O OVE6D_pose_model.pth`

Or manually from below.

### Segmentation modules
Segmentation: https://github.com/AnasIbrahim/image_agnostic_segmentation

`https://github.com/AnasIbrahim/image_agnostic_segmentation`

`https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend`
	- In progress

## pre-trained weight for OVE6D
Our pre-trained OVE6D weights can be found [here](https://drive.google.com/drive/folders/16f2xOjQszVY4aC-oVboAD-Z40Aajoc1s?usp=sharing). Please download and save to the directory ``checkpoints/``.


# Acknowledgement
- OVE6D: [Project page](https://dingdingcai.github.io/ove6d-pose/) 


### TODO:
https://stackoverflow.com/questions/65385983/prevent-conda-from-automatically-downgrading-python-package
https://stackoverflow.com/questions/65483245/how-to-avoid-using-conda-forge-packages-unless-necessary

https://github.com/facebookresearch/pytorch3d/issues/1076
