# PerfectRep - 3D Pose Analysis for Powerlifters

_inspired by [MotionBERT](https://arxiv.org/pdf/2210.06551.pdf)_

## Table of Contents

1. [Pretrain Model](#1-pretrain-model)
   - [Architecture](#--architecture)
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

## 2. 3D Pose-Estimation

## 3. Pose Classification

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

| Subject | Activities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Type     |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| s03     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Training |
| s04     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Training |
| s05     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Training |
| s07     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Training |
| s08     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Training |
| s09     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Training |
| s10     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing, ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9 | Training |
| s11     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Training |
| s02     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Test     |
| s12     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Test     |
| s13     | band_pull_apart, dumbbell_high_pulls, dumbbell_reverse_lunge, pushup, drag_curl, warmup_13, warmup_19, side_lateral_raise, deadlift, warmup_14, warmup_12, warmup_6, dumbbell_hammer_curls, w_raise, standing_ab_twists, warmup_11, warmup_1, barbell_dead_row, one_arm_row, squat, clean_and_press, warmup_2, diamond_pushup, dumbbell_scaptions, barbell_shrug, dumbbell_biceps_curls, warmup_3, dumbbell_overhead_shoulder_press, burpees, overhead_extension_thruster, warmup_16, warmup_18, man_maker, warmup_5, warmup_8, warmup_10, dumbbell_curl_trifecta, barbell_row, overhead_trap_raises, neutral_overhead_shoulder_press, walk_the_box, warmup_4, warmup_7, mule_kick, warmup_17, warmup_15, warmup_9  | Test     |

### **Reference Articles**

1. [What is a Transformer?](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
2. [What is BERT? How it is trained ? A High Level Overview](https://medium.com/@Suraj_Yadav/what-is-bert-how-it-is-trained-a-high-level-overview-1207a910aaed)
3. [The Perceptron Algorithm: How it Works and Why it Works](https://medium.com/geekculture/the-perceptron-algorithm-how-it-works-and-why-it-works-3668a80f8797)
4. [Understanding Backpropagation Algorithm](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)
