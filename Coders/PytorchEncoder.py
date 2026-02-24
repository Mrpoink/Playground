import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.NeuralNetwork as NN
import BaseFunctions.BaseTokenizer as Tokenizer
import Coders.PytorchAtt as ATT
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

_math = MATH.Math()
_mle = MLE.SMLE()


class Encoder(nn.Module):
    
    def __init__(self, input_size, n_heads, dim_ffn = 1000):
        super().__init__()
        self.mha = ATT.MultiHeadAttention(n_heads, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        
        ## Replaces FFN implementation
        self.ffn = nn.Sequential(
            nn.Linear(input_size, dim_ffn),
            nn.ReLU(),
            nn.Linear(dim_ffn, input_size)
        )
        
    def forward(self, x, mask=None):
        
        attn_out, _ = self.mha.forward(x, x, x, mask=mask)
        x = self.norm1(x + attn_out) ## Replaces _math.matrix_addition
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x