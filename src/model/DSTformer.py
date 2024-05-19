import torch
import torch.nn as nn
import math
import warnings
import random
import numpy as np
from collections import OrderedDict
from functools import partial
from itertools import repeat
from enum import Enum
from src.model.drop_path import DropPath

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class AttentionType(Enum):
    """
    Enumerates different types of attention mechanisms.

    Attributes:
        VANILLA (int): Vanilla attention mechanism.
        SPATIAL (int): Spatial attention mechanism.
        TEMPORAL (int): Temporal attention mechanism.
    """
    VANILLA = 1
    SPATIAL = 2
    TEMPORAL = 3

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) neural network model.
    Used after attention block to get motion encoding

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of neurons in the hidden layer.
            If not provided, defaults to the same number as input features.
        out_features (int, optional): Number of output features.
            If not provided, defaults to the same number as input features.
        act_layer (torch.nn.Module, optional): Activation function to be used in hidden layer.
            Defaults to GELU activation function.
        drop (float, optional): Dropout probability. Default is 0, indicating no dropout.

    Attributes:
        fc1 (torch.nn.Linear): Input layer.
        act (torch.nn.Module): Activation function.
        fc2 (torch.nn.Linear): Hidden layer.
        drop (torch.nn.Dropout): Dropout layer applied after each fully connected layer.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) # Input layer
        self.act = act_layer() # GELU activation
        self.fc2 = nn.Linear(hidden_features, out_features) # First hidden layer
        self.drop = nn.Dropout(drop) # Neuron dropout. Applied after each layer

    def forward(self, x):
        x = self.fc1(x) # Input layer
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) # Hidden layer
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        dim (int): Dimensionality of the input.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, include bias terms in the calculation of Q, K, and V.
        qk_scale (float, optional): Scale factor for Q and K.
        attn_drop (float): Dropout probability applied to attention weights.
        proj_drop (float): Dropout probability applied to the output.
        st_mode (AttentionType): Type of attention mechanism to be used.

    Attributes:
        num_heads (int): Number of attention heads.
        scale (float): Scale factor for Q and K.
        attn_drop (Dropout): Dropout layer applied to attention weights.
        proj (Linear): Linear transformation applied to the input.
        mode (AttentionType): Type of attention mechanism being used.
        qkv (Linear): Linear transformation to compute Q, K, and V.
        proj_drop (Dropout): Dropout layer applied to the output.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., st_mode=AttentionType.VANILLA):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # Dimesion of an individual head embedding space
        self.scale = qk_scale or head_dim ** -0.5 # Sale be root of embedding dimension

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) # Linear transformation applied to encoded motion
        self.mode = st_mode

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # Linear transformation to hold params of Q,K,V
        self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x, seqlen=1):
            """
            Compute forward pass for the multi-head attention block.

            Args:
                x (Tensor): Input tensor of shape (B, N, C), where B is the batch size, N is the number of elements, and C is the embedding dimension.
                seqlen (int, optional): Sequence length (for temporal attention).

            Returns:
                Tensor: Output tensor after applying multi-head attention, with shape (B, N, C).
            """
            B, N, C = x.shape # Batch, Frames, Joints Embedding

            if self.mode == AttentionType.VANILLA:
                # Multiply input by projection matrix to get Q K V matrices
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (3, B, H, N, C)
                q, k, v = qkv[0], qkv[1], qkv[2] # Extract Q K V each with shape (B, H, N, C) 
                x = self.forward_spatial(q, k, v)
            elif self.mode == AttentionType.TEMPORAL:
                # Multiply input by projection matrix to get Q K V matrices
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (3, B, H, N, C)
                q, k, v = qkv[0], qkv[1], qkv[2] # Extract Q K V each with shape (B, H, N, C)  
                x = self.forward_temporal(q, k, v, seqlen=seqlen)
            elif self.mode == AttentionType.SPATIAL:
                # Multiply input by projection matrix to get Q K V matrices
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (3, B, H, N, C)
                q, k, v = qkv[0], qkv[1], qkv[2] # Extract Q K V each with shape (B, H, N, C) 
                x = self.forward_spatial(q, k, v)
            else:
                raise NotImplementedError(self.mode)

            assert qkv.shape[-1] == C // self.num_heads
            assert qkv.shape[2] == self.num_heads

            x = self.proj(x) # Apply linear projection to get motion encoding
            x = self.proj_drop(x) # Dropout
            return x
        
        def forward_spatial(self, q, k, v):
            """
            Forward pass for spatial attention mechanism.

            Args:
                q (Tensor): Query tensor.
                k (Tensor): Key tensor.
                v (Tensor): Value tensor.

            Returns:
                Tensor: Output tensor after applying spatial attention.
            """
            B, _, N, C = q.shape
            attn = (q @ k.transpose(-2, -1)) * self.scale # (Q * K^T) / d_K ** 0.5
            attn = attn.softmax(dim=-1) # Softmax to get probabilty distribution
            attn = self.attn_drop(attn) # Apply dropout

            x = attn @ v # Multiply by values
            x = x.transpose(1,2).reshape(B, N, C*self.num_heads) # Concat heads into result
            return x

        def forward_temporal(self, q, k, v, seqlen=8):
            """
            Forward pass for temporal attention mechanism.

            Args:
                q (Tensor): Query tensor.
                k (Tensor): Key tensor.
                v (Tensor): Value tensor.
                seqlen (int): Sequence length of clip.

            Returns:
                Tensor: Output tensor after applying temporal attention.
            """
            B, _, N, C = q.shape
            # Reshape to parallelize over the spatial dimension
            qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4) #(B, H, N, T, C)
            kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4) #(B, H, N, T, C)
            vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4) #(B, H, N, T, C)

            attn = (qt @ kt.transpose(-2, -1)) * self.scale # (Q * K^T) / d_K ** 0.5
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ vt #(B, H, N, T, C)
            x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C*self.num_heads) # Reshape and concat heads into result
            return x

class Block(nn.Module):
    """
    Transformer block consisting of both spatial and temporal attention mechanisms.

    Args:
        dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of hidden dimension to input dimension for the MLP. Default is 4.0.
        mlp_out_ratio (float, optional): Ratio of output dimension to input dimension for the MLP. Default is 1.0.
        qkv_bias (bool, optional): If True, include bias to query, key, and value tensors. Default is True.
        qk_scale (float, optional): Scaling factor for query and key. Default is None.
        drop (float, optional): Dropout rate. Default is 0.0.
        attn_drop (float, optional): Dropout rate for attention weights. Default is 0.0.
        drop_path (float, optional): Drop path rate. Default is 0.0.
        act_layer (torch.nn.Module, optional): Activation function. Default is nn.GELU.
        norm_layer (torch.nn.Module, optional): Normalization layer. Default is nn.LayerNorm.
        st_mode (str, optional): Mode for the order of spatial and temporal attention. Default is 'stage_st'.
        att_fuse (bool, optional): If True, fuse spatial and temporal attention with learned weights. Default is False.

    Attributes:
        st_mode (str): Mode for the order of spatial and temporal attention.
        norm1_s (nn.LayerNorm): Normalization layer for spatial attention.
        norm1_t (nn.LayerNorm): Normalization layer for temporal attention.
        attn_s (Attention): Spatial attention mechanism.
        attn_t (Attention): Temporal attention mechanism.
        drop_path (nn.Identity or DropPath): Stochastic depth layer.
        norm2_s (nn.LayerNorm): Second normalization layer for spatial attention.
        norm2_t (nn.LayerNorm): Second normalization layer for temporal attention.
        mlp_s (MLP): MLP for motion encoding in spatial stream.
        mlp_t (MLP): MLP for motion encoding in temporal stream.
        att_fuse (bool): Whether to fuse spatial and temporal attention.
        ts_attn (nn.Linear, optional): Linear layer to fuse spatial and temporal attention.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='stage_st', att_fuse=False):
        super().__init__()

        self.st_mode = st_mode
        # First layer norm after inital attention block
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, st_mode=AttentionType.SPATIAL)
        self.attn_t = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, st_mode=AttentionType.TEMPORAL)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # Stochastic depth
        # Second layer norm after second attention block
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        # MLP to get motion encoding for each stream
        self.mlp_s = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer, drop=drop)
        self.mlp_t = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer, drop=drop)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.Linear(dim*2, dim*2) # Fuse spatial and temproal attention with learned weights

        def forward(self, x, seqlen=1):
            """
            Forward pass through the transformer block.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).
                seqlen (int, optional): Sequence length. Default is 1.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
            """
            if self.st_mode=='stage_st':
                x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
                x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
                x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
                x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            elif self.st_mode=='stage_ts':
                x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
                x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
                x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
                x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            else:
                raise NotImplementedError(self.st_mode)
            return x

class DSTformer(nn.Module):
    """
    Dual-stream transformer for spatiotemporal feature extraction and motion encoding.

    Args:
        dim_in (int, optional): Dimensionality of the input. Default is 3.
        dim_out (int, optional): Dimensionality of the output. Default is 3.
        dim_feat (int, optional): Dimensionality of the feature embeddings. Default is 256.
        dim_rep (int, optional): Dimensionality of the representation layer. Default is 512.
        depth (int, optional): Number of transformer blocks. Default is 5.
        num_heads (int, optional): Number of attention heads. Default is 8.
        mlp_ratio (float, optional): Ratio of hidden dimension to input dimension for the MLP. Default is 4.0.
        num_joints (int, optional): Number of joints. Default is 17.
        maxlen (int, optional): Maximum sequence length. Default is 243.
        qkv_bias (bool, optional): If True, include bias terms in the calculation of Q, K, and V. Default is True.
        qk_scale (float, optional): Scale factor for Q and K. Default is None.
        drop_rate (float, optional): Dropout rate. Default is 0.0.
        attn_drop_rate (float, optional): Dropout rate for attention weights. Default is 0.0.
        drop_path_rate (float, optional): Drop path rate. Default is 0.0.
        norm_layer (torch.nn.Module, optional): Normalization layer. Default is nn.LayerNorm.
        att_fuse (bool, optional): If True, fuse spatial and temporal attention with learned weights. Default is True.

    Attributes:
        dim_out (int): Dimensionality of the output.
        dim_feat (int): Dimensionality of the feature embeddings.
        joints_embed (nn.Linear): Linear layer for input embeddings.
        pos_drop (nn.Dropout): Dropout layer for positional encoding.
        blocks_st (nn.ModuleList): List of spatial transformer blocks.
        blocks_ts (nn.ModuleList): List of temporal transformer blocks.
        norm (nn.LayerNorm): Layer normalization.
        pre_logits (nn.Sequential or nn.Identity): Fully connected layer for motion encoding.
        head (nn.Linear or nn.Identity): Linear layer for final output transformation.
        temp_embed (nn.Parameter): Temporal embedding.
        pos_embed (nn.Parameter): Spatial embedding.
        att_fuse (bool): Whether to fuse spatial and temporal attention.
        ts_attn (nn.ModuleList, optional): List of linear layers for attention fusion.

    Methods:
        _init_weights(m):
            Initialize the weights of the given module.
        get_classifier():
            Return the head classifier.
        reset_classifier(dim_out, global_pool=''):
            Reset the head classifier with new output dimensions.
        forward(x, return_rep=False):
            Forward pass through the transformer.
        get_representation(x):
            Get the representation layer output.
    """
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=4, 
                 num_joints=17, maxlen=243, 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat) # Embeddings in feature space
        
        # Apply dropout to positional encoding
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # Spatial Steam Block
        self.blocks_st = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                st_mode="stage_st")
            for i in range(depth)])
        # Temporal Stream Block
        self.blocks_ts = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                st_mode="stage_ts")
            for i in range(depth)])
        self.norm = norm_layer(dim_feat) # Layer Normalization
        
        # Fully Connected layer to transform from feature embedding to motion encoding
        if dim_rep:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim_feat, dim_rep)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        # Transformation from motion encoding to output dimension
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()

        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat)) # Temporal embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat)) # Spatial embedding
        # Fill embedding matrices weights from normal distribution
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights) # Fill weights with one and bias zero

        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat*2, 2) for i in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)
        
        def _init_weights(self, m):
            """
            Initialize the weights of the given module.
        
            This function initializes the weights and biases of the given module `m` based on its type:
            - For `nn.Linear` layers, the weights are initialized using a truncated normal distribution with a standard deviation of 0.02, and biases are set to zero if they exist.
            - For `nn.LayerNorm` layers, both weights and biases are set to one and zero respectively.
        
            Args:
            m : torch.nn.Module
                The module to initialize. This should be an instance of either `nn.Linear` or `nn.LayerNorm`.
            """
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        def get_classifier(self):
            return self.head

        def reset_classifier(self, dim_out, global_pool=''):
            self.dim_out = dim_out
            self.head = nn.Linear(self.dim_feat, dim_out) if dim_out > 0 else nn.Identity()
        
        def forward(self, x, return_rep=False):   
            B, F, J, C = x.shape # Batch : Frame : Joint : Embedding Space
            x = x.reshape(-1, J, C) # Concat all batches
            BF = x.shape[0]
            x = self.joints_embed(x)
            x = x + self.pos_embed # Add positional embedding
            _, J, C = x.shape
            x = x.reshape(-1, F, J, C) + self.temp_embed[:,:F,:,:] # Temportal encoding
            x = x.reshape(BF, J, C)
            x = self.pos_drop(x)
            alphas = [] # Fuse alpha vals
            for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
                x_st = blk_st(x, F)
                x_ts = blk_ts(x, F)
                if self.att_fuse:
                    att = self.ts_attn[idx]
                    alpha = torch.cat([x_st, x_ts], dim=-1)
                    BF, J = alpha.shape[:2]
                    alpha = att(alpha)
                    alpha = alpha.softmax(dim=-1)
                    x = x_st * alpha[:,:,0:1] + x_ts * alpha[:,:,1:2]
                else:
                    x = (x_st + x_ts)*0.5 # Assign equal weight is fuse not applied
            x = self.norm(x)
            x = x.reshape(B, F, J, -1)
            x = self.pre_logits(x) # [B, F, J, dim_feat]
            if return_rep:
                return x
            x = self.head(x)
            return x

        def get_representation(self, x):
            return self.forward(x, return_rep=True)