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


class Decoder(nn.Module):
    def __init__(self, input_size, n_heads, dim_ffn=1000):
        super(Decoder, self).__init__()
        
        self.self_attn = ATT.MultiHeadAttention(n_heads, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        
        self.cross_attn = ATT.MultiHeadAttention(n_heads, input_size)
        self.norm2 = nn.LayerNorm(input_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_size, dim_ffn),
            nn.ReLU(),
            nn.Linear(dim_ffn, input_size)
        )
        self.norm3 = nn.LayerNorm(input_size)
        
        self.output_layer = nn.Linear(input_size, input_size)
        self.final_norm = nn.LayerNorm(input_size)
        
        
    def forward(self, tgt, enc, tgt_mask=None):
        
        attn_self, _ = self.self_attn.forward(tgt, tgt, tgt, mask=tgt_mask)
        x = self.norm1(tgt + attn_self)
        
        attn_cross, _ = self.cross_attn(x, enc, enc)
        x = self.norm2(x + attn_cross)
        
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        output = self.output_layer(x)
        return self.final_norm(output)