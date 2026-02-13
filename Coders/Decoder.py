import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.NeuralNetwork as NN
import BaseFunctions.BaseTokenizer as Tokenizer
import Coders.FFN as FFN
import Coders.Attention as ATT
import Coders.Encoder as CE
import random
import math

_math = MATH.Math()
_mle = MLE.SMLE()

class Decoder(CE.Encoder):
    
    def __init__(self, seq_len, input_size, n_heads=2):
        self.self_attn = ATT.MultiHead(n_heads=n_heads, d_model=input_size)
        self.cross_attn = ATT.MultiHead(n_heads=n_heads, d_model=input_size)
        self.ffn = FFN.FeedForward(4000, input_size=input_size)
    
    def process(self, tgt_input, enc_output):
        # 1. Masked Self-Attention
        mask = _mle.generate_mask(len(tgt_input))
        x_s, _ = self.self_attn.forward(tgt_input, tgt_input, tgt_input, mask=mask)
        x_s = _math.layer_norm(_math.matrix_addition(x_s, tgt_input))

        # 2. Cross-Attention (The actual combination step)
        x_c, _ = self.cross_attn.forward(x_s, enc_output, enc_output)
        x_c = _math.layer_norm(_math.matrix_addition(x_c, x_s))
     
        ffn_out = self.ffn.forward(x_c)
        return _math.layer_norm(_math.matrix_addition(x_c, ffn_out))
    
    def backward(self, d_output, enc_output, lr):
        
        d_ffn_in = self.ffn.backward(d_output, lr)
        
        ##Cross Attention
        d_cross_q, d_enc_out = self.cross_attn.backward(
            d_ffn_in, 
            self.self_attn.cache['input_q'], 
            enc_output, 
            lr
        )
        
        d_self_in = _math.matrix_addition(d_ffn_in, d_cross_q)
        d_self_q, d_self_kv = self.self_attn.backward(d_self_in, 
                                                      self.self_attn.cache['input_q'], 
                                                      self.self_attn.cache['input_q'], 
                                                      lr)
        
        
        return d_enc_out
            
        