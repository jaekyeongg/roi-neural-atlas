# RNA: Video Editing with ROI-based Neural Atlas

### [Project Page](https://jaekyeongg.github.io/RNA) | [Paper](https://arxiv.org/abs/2410.07600)

## Installation

Our code is compatible and validate with Python 3.10m PyTorch 1.13.1, and CUDA 11.7.

```
conda create -n rna  python=3.10
conda activate rna
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib tensorboard scipy  scikit-image tqdm
pip install opencv-python imageio-ffmpeg gdown easydict  fairscale
CC=gcc-9 CXX=g++-9 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
git submodule update --init
```

## Directory structures for datasets

```
data
├── <video_name>
│   ├── video_frames
│   │   └── %05d.jpg or %05d.png ...
```

## Data preparations

### Video frames

The video frames follows the format of [DAVIS](https://davischallenge.org/) dataset. The file type of images should be all either in png or jpg and named as `00000.jpg`, `00001.jpg`, ...

### Preprocess optical flow

We extract the optical flow using [RAFT](https://arxiv.org/abs/2003.12039). The models can be downloaded by the following command:

```
cd thirdparty/RAFT/
./download_models.sh
cd ../..
```

Set the dataset name, reference frame, and the ROI of the area you want to edit in the config file.
To create optical flow for the video and make initial mask, run:

```
python preprocess_main.py --config config/config.py
```

The script will automatically generate the corresponding optical flow and initial mask in the right directory.


## Training

To train a video, run:

```
python train.py --config config/config.py
```

The config file and checkpoint file will be stored to the assigned result folder.

## Postprocess

Download the model from [VittMatte](https://drive.google.com/file/d/12VKhSwE_miF9lWQQCgK7mv83rJIls3Xe/view) and place it in the `thirdparty/Vitmatte/models` directory.
To matte the mask, run : 

```
python postprocess.py --config config/config.py
```

The matted mask will be stored to the refined_mask folder.


## Editing

You can edit the `tex0.png` to edit the video and set the checkpoint path, the edited texture file in the config file.
After that, run: 

```
python edit.py --config config/config.py
```

The edited video will be generated in the same folder and named as `edit.mp4`.

## Citation

If you find our work useful in your research, please consider citing:

```
@article{lee2024rna,
    title={RNA: Video Editing with ROI-based Neural Atlas},
    author={Lee, Jaekyeong and Kim, Geonung and Cho, Sunghyun},
    journal={arXiv preprint arXiv:2410.07600},
    year={2024}
    }
```

## Acknowledgement

We thank [Layered Neural Atlases](https://github.com/ykasten/layered-neural-atlases) and [Hashing NVD](https://github.com/vllab/hashing-nvd) for using their code implementation as our code base. We modify the code to meet our requirements.


