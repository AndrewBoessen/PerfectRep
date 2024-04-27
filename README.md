# PerfectRep - 3D Pose Analysis for Powerlifters 🏋️🦾

![PerfectRep](./assets/extended_banner.png)

PerfectRep is a 3D pose estimation model tailored specifically for powerlifting analysis. It allows for precise tracking and analysis of lifter's movements to ensure perfect form and technique.

## Dependencies

- Python >= 3.7
- Pytorch
- Cuda=11.6
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
   conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

## Training Instructions

1. Prepare your dataset in the appropriate format. Ensure it includes labeled data for powerlifting movements.

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

## Documentation

- [Pretrain Model](./docs/pretrain.md)
- [3D Pose Estimation](./docs/3D-pose.md)
- [Form Analysis](./docs/form-analysis.md)
- [Data Set](./docs/dataset.md)
- [References](./docs/reference.md)

## References

- [MotionBERT](https://arxiv.org/pdf/2210.06551.pdf)
- [AIFit](https://mihaifieraru.github.io/publication/fieraru_2021_cvpr/Fieraru_2021_CVPR.pdf)
