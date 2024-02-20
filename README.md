# PerfectRep - 3D Pose Analysis for Powerlifters

_inspired by [MotionBERT](https://arxiv.org/pdf/2210.06551.pdf)_

## Table of Contents

1. [Pretrain Model](#1-pretrain-model)
   - [Architecture](#--architecture)
   - [Implementation](#--implementation)
   - [Training](#--training)
2. [3D Pose-Estimation](#2-3d-pose-estimation)
3. [Pose Classification](#3-pose-classification)
4. [Form Analysis](#4-form-analysis)
5. [Rep Counter](#5-rep-counter)
6. [Data Set: Fit3D](#6-data-set-fit3d)
   - [Data Collection](#data-collection)
   - [Dataset Info](#dataset-info)

[Reference Articles](#reference-articles)

## 1. Pretrain Model

![pretrain](./assets/pretrain.png)

### _A motion encoder to accomplish the 2D-to-3D lifting task_

### - Architecture:

![architecture](./assets/dstformer.png)

- **Input** _2D skeleton sequences_

  - For in-the-wild inference, use [Alpha Pose](https://github.com/MVIG-SJTU/AlphaPose#quick-start) to extract 2D keypoints from video
  - Represented as a tensor. Each set of keypoints $x$ in the series is a vector.

    $x \in \mathbb{R}^{(T \times J \times C_{\text{in}})}$

    Here, $T$ denotes the sequence length, and $J$ denotes the number of body joints. $C_{\text{in}}$ denotes the channelnumber of input.

- **Output** _3D skeleton sequences_

  - Represented as tensor. Each set of keypoints $\hat{X}$ in the series is a vector

    $\hat{X} \in \mathbb{R}^{(T \times J \times C_{\text{out}})}$

    Here, $T$ denotes the sequence length, and $J$ denotes the number of body joints. $C_{\text{out}}$ denotes the channelnumber of output.

- **Backbone** _DSTformer_

  - DSTformer consists of $N$ dual-stream-fusion modules.

    Each module contains two branches of spatial or temporal MHSA and MLP.

    The Spatial MHSA models the connection among different joints within a timestep, while the Temporal MHSA models the movement of one joint

  - **Spatial Block**

    - Spatial MHSA (S-MHSA) aims at modeling the relationship among the joints within the same time step. It is defined as

      $\text{S-MHSA}(Q_S, K_S, V_S) = [\text{head}_1; \ldots; \text{head}_h]W_S^P$,

      $\text{head}_i = \text{softmax}\left(\frac{Q_S^i (K_S^i)^\prime}{\sqrt{d_K}}\right)V_S^i$

      where $W_S^P$ is a projection parameter matrix, $h$ is the number of the heads, $i \in 1, . . . , h$, and $′$ denotes matrix transpose.

      We utilize self-attention to get the query $Q^S$, key $K^S$, and value $V^S$ from input per-frame spatial feature $F_S \in \mathbb{R}^{J \times Ce}$ for each head $_i$

      $Q_S^i = F_{S}W_{S}^{(Q,i)}, \quad K_S^i = F_{S}W_{S}^{(K,i)}, \quad V_S^i = F_{S}W_{S}^{(V,i)}$,

      where $W_S^{(Q,i)}$, $W_S^{(K,i)}$, $W_S^{(V,i)}$ are projection matrices, and $d_K$ is the feature dimension of $K_S$.

      We apply S-MHSA to 3 features of different time steps in parallel. Residual connection and layer normalization (LayerNorm) are used to the S-MHSA result, which is further fed into a multilayer perceptron (MLP), and followed by a residual connection and LayerNorm following

  - **Temporal Block**

    - Temporal MHSA (T-MHSA) aims at
      modeling the relationship across the time steps for a body
      joint. Its computation process is similar with S-MHSA except that the MHSA is applied to the per-joint temporal
      feature $F_T \in \mathbb{R}^{T \times C_e}$ and parallelized over the spatial dimension

      $\text{T-MHSA}(Q_T, K_T, V_T) = [\text{head}_1; \ldots; \text{head}_h]W_T^P$,

      $\text{head}_i = \text{softmax}\left(\frac{Q_T^i (K_T^i)^\prime}{\sqrt{d_K}}\right)V_T^i$

      where $i \in 1,...,h, Q_T, K_T, V_T$ are computed similar with S-MHSA

### - Implementation:

We implement the proposed motion encoder DSTformer
with depth $N = 5$, number of heads $h = 8$, feature size
$C_f = 512$, embedding size $C_e = 512$. For pretraining, we
use sequence length $T = 243$. The pretrained model could
handle different input lengths thanks to the Transformerbased backbone. During finetuning, we set the backbone
learning rate to be $0.1 ×$ of the new layer learning rate. We
introduce the experiment datasets in the following sections
respectively

### - Training:

We extract 3D keypoints from Fit3D by camera projection. We sample motion clips with length $T = 243$ for
3D mocap data. For 2D data, we utilize the provided annotations of PoseTrack. We further include 2D motion from
an unannotated video dataset InstaVariety extracted by
OpenPose. Since the valid sequence lengths for in-thewild videos are much shorter, we use $T = 30$ (PoseTrack)
and $T = 81$ (InstaVariety). We convert keypoints of 2D
datasets (COCO and OpenPose format) to Fit3D using permutation and interpolation following previous works.
We set the input channels $C_{in} = 3 (x, y$ coordinates and
confidence). Random horizontal flipping is applied as data augmentation. The whole network
is trained for 90 epochs with learning rate 0.0005 and batch
size 64 using an Adam optimizer.

## 2. 3D Pose-Estimation

### - Finetuning:

### - Training:

## 3. Pose Classification

### - Data Pipeline:

### - Finetuning:

### - Training:

## 4. Form Analysis

## 5. Rep Counter

## 6. Data Set: [Fit3D](https://fit3d.imar.ro/)

### - 611 multi-view sequences; minimum 5 annotated repetitions per sequence;

### - 2,964,236 highly accurate ground truth 3d skeletons,

### - GHUM & SMPLX human pose and shape parameters

### Data Collection:

- **Collected from multiple subjects**

  ![subjects exercising](./assets/fit3d_single_view.gif)

  - Traning Set: 8 Subjects (all trainees)
  - Testing Set: 3 Subjects (1 trainer, 2 trainees)

- **47 Exercises**

  ![multiple exercises](./assets/fit3d_actions.gif)

  - Use all 47 for traning / testing
  - Use squat, deadlift, push-up for classification

- **3D Skeletons**

  ![3d skeletons](./assets/fit3d_multi_view_skeleton_400.gif)

  - Ground-truth 3d skeletons with 25 joints (including the 17 Human3.6m joints)
  - 50 fps

### Dataset Info

### **Reference Articles**

1. [What is a Transformer?](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
2. [What is BERT? How it is trained ? A High Level Overview](https://medium.com/@Suraj_Yadav/what-is-bert-how-it-is-trained-a-high-level-overview-1207a910aaed)
3. [The Perceptron Algorithm: How it Works and Why it Works](https://medium.com/geekculture/the-perceptron-algorithm-how-it-works-and-why-it-works-3668a80f8797)
4. [Understanding Backpropagation Algorithm](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)
