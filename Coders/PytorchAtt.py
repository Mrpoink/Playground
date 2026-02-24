import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.NeuralNetwork as NN
import BaseFunctions.BaseTokenizer as Tokenizer
import Coders.Attention as ATT
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

_math = MATH.Math()
_mle = MLE.SMLE()

class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
        
    def forward(self, Q, K, V, mask=None):
        
        batch_size = Q.size(0)
        
        ## Replaces _math.dot_product
        q_proj = self.Wq(Q)
        k_proj = self.Wk(K)
        v_proj = self.Wv(V)
        
        ## Replaces _mle.split_heads
        q_proj = q_proj.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k_proj = k_proj.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v_proj = v_proj.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        ## Replaces _math.dot_product and scaling
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            
            ## Replaces _math.matrix_addition with mask enabled
            scores = scores.masked_fill(mask == 0, -1e9)
        
        ## Replaces _mle.softmax
        weights = F.softmax(scores, dim=-1)
        
        ## Replaces _math.dot_product for scaling softmax
        context = torch.matmul(weights, v_proj)
        
        # Replaces _math.concate_heads and final Wo dot product
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.Wo(context)
        
        return output, weights