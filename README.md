# PerfectRep - 3D Pose Analysis for Powerlifters 🏋️🦾

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

[![Build Status](https://github.com/AndrewBoessen/PerfectRep/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/AndrewBoessen/PerfectRep/actions/workflows/python-app.yml)

![PerfectRep](./assets/extended_banner.png)

PerfectRep is a 3D pose estimation model tailored specifically for powerlifting analysis. It allows for precise tracking and analysis of lifter's movements to ensure perfect form and technique.

## Dependencies

- Python >= 3.7
- Pytorch
- Cuda=11.8
- NumPy
- Matplotlib
- Pandas

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AndrewBoessen/PerfectRep.git
   ```

2. Navigate to the project directory:

   ```bash
   cd PerfectRep
   ```

3. Create Virtual Environment:

   ```bash
   conda create -n perfectrep python=3.7 anaconda
   conda activate perfectrep
   # Please install PyTorch according to your CUDA version.
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

## Training Instructions

1. Prepare your dataset in the appropriate format. Have required preprocessed data in the correct directory. See [Data Preprocessing](#data-preprocessing) for more info

2. Train the model using the provided training script:

   ```bash
   python train.py --dataset /path/to/dataset --epochs 100 --batch_size 32
   ```

3. Monitor the training progress and adjust hyperparameters as necessary.

4. Once satisfied with the training, save the trained model for later use.

## Usage Examples

1. Perform inference on a single image:

   ```bash
   python infer.py --image /path/to/image.jpg
   ```

2. Process a video to analyze multiple frames:

   ```bash
   python infer.py --video /path/to/video.mp4
   ```

## Data Preprocessing

For pretraing we use the Fit3D data set which is also used for finetuning 3D pose and classification. The data set must be preprocessed before being used for training.

### Download

> Note that the preprocessed data is only intended for reproducing our results more easily. If you want to use the dataset, please register to the [Fit3D website](https://fit3d.imar.ro/home) and download the dataset in its original format.

| Dataset   | Description                                                                | Size    | Download Link                                                                                               |
| --------- | -------------------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| Fit3D     | Fit3D is a dataset for 3D human-interpretable models for fitness training. | 1.96 GB | [Download Fit3D](https://drive.google.com/file/d/1B8BT67Q_ZLbT638cbT3msoIYWUwYWzxz/view?usp=drive_link)     |

1. Once downloaded unzip the files into `data/motion3d`

2. Slice the data into clips (len=243, stride=81)
```
python process_fit3d.py
```

> To processes the raw dataset downloaded from [Fit3D website](https://fit3d.imar.ro/home) place the train dataset and `fit3d_info.json`file in `data/fit3d/` and run

```
python compress_fit3d.py
```
> Note it is still necessary to slice the data into clips after the raw data set has been preprocessed

## Documentation

[![view - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](/docs/ "Go to project documentation")

- [Pretrain Model](./docs/pretrain.md)
- [3D Pose Estimation](./docs/3D-pose.md)
- [Form Analysis](./docs/form-analysis.md)
- [Data Set](./docs/dataset.md)
- [References](./docs/reference.md)

## References

- [MotionBERT](https://arxiv.org/pdf/2210.06551.pdf)
- [AIFit](https://mihaifieraru.github.io/publication/fieraru_2021_cvpr/Fieraru_2021_CVPR.pdf)
