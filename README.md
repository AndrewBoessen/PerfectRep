# PerfectRep - 3D Pose Analysis for Powerlifters

_inspired by [MotionBERT](https://arxiv.org/pdf/2210.06551.pdf)_

## Table of Contents

1. [Pretrain Model](#1-pretrain-model)
   - [Data Set: Fit3D](#--data-set-fit3d)
   - [Architecture](#--architecture)
2. [3D Pose-Estimation](#2-3d-pose-estimation)
3. [Pose Classification](#3-pose-classification)
4. [Form Analysis](#4-form-analysis)
5. [Rep Counter](#5-rep-counter)

## 1. Pretrain Model

![pretrain](./assets/pretrain.png)

### _A motion encoder to accomplish the 2D-to-3D lifting task_

### - Data Set: [Fit3D](https://fit3d.imar.ro/)

- **Collected from multiple subjects**

  ![subjects exercising](./assets/fit3d_single_view.gif)

  - Traning Set: 8 Subjects (all tranees)
  - Testing Set: 3 Subjects (1 trainer, 2 trainees)

- **47 Exercises**

  ![multiple exercises](./assets/fit3d_actions.gif)

  - Use all 47 for traning / testing
  - Use squat, deadlift, push-up for classification

- **3D Skeletons**

  ![3d skeletons](./assets/fit3d_multi_view_skeleton_400.gif)

  - Ground-truth 3d skeletons with 25 joints (including the 17 Human3.6m joints)
  - 50 fps

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

      $\text{S-MHSA}(Q_S, K_S, V_S) = [\text{head}_1; \ldots; \text{head}_h]W_P S$,

      $\text{head}_i = \text{softmax}\left(\frac{Q_i S(K_i S)^\prime}{\sqrt{d_K}}\right)V_i S$

      where $W_P S$ is a projection parameter matrix, $h$ is the number of the heads, $i \in 1, . . . , h$, and $′$ denotes matrix transpose.

  - **Temporal Block**

    - Temporal MHSA (T-MHSA) aims at
      modeling the relationship across the time steps for a body
      joint. Its computation process is similar with S-MHSA except that the MHSA is applied to the per-joint temporal
      feature $F_T \in \mathbb{R}^{T \times C_e}$ and parallelized over the spatial dimension

## 2. 3D Pose-Estimation

## 3. Pose Classification

## 4. Form Analysis

## 5. Rep Counter
