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
from lib.model.drop_path import DropPath

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
