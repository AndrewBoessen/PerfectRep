# Pretrain Model

![pretrain](../assets/pretrain.png)

### _A motion encoder to accomplish the 2D-to-3D lifting task_

### - Architecture:

![architecture](../assets/dstformer.png)

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
