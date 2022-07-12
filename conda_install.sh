### Conda
#Constrained by Pytorch3d cudatoolkit=10.2 requirement.
conda create -n pytorch3d_test python=3.8
conda activate pytorch3d_test
conda install pytorch==1.8.0::pytorch torchvision==0.9.0::pytorch torchaudio==0.8.0::pytorch cudatoolkit=11.1 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d=0.6.2=py38_cu111_pyt180 matplotlib ipdb -c pytorch3d -c conda-forge

conda install detectron2=0.6=torch18_cuda111_py38hb821680_0 -c conda-forge
conda install opencv=4.5.5=py38ha5a2927_2 ipykernel ipdb


#conda install --freeze-installed detectron2=0.6=torch19_cuda102_py39hfdd23e2_0 -c conda-forge
