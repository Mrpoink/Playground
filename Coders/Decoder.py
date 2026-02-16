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
        self.ffn = FFN.FeedForward(400, input_size=input_size)
        
        self.Wo = _math.generate_he_matrix(input_size, input_size)
        self.cache = {}
    
    def process(self, tgt_input, enc_output):
        # 1. Masked Self-Attention
        mask = _mle.generate_mask(len(tgt_input))
        x_s, _ = self.self_attn.forward(tgt_input, tgt_input, tgt_input, mask=mask)
        x_s, ln_cache = _math.layer_norm_forward(_math.matrix_addition(x_s, tgt_input))
        self.cache['ln1_s'] = ln_cache

        # 2. Cross-Attention (The actual combination step)
        x_c, _ = self.cross_attn.forward(x_s, enc_output, enc_output)
        x_c, ln_cache = _math.layer_norm_forward(_math.matrix_addition(x_c, x_s))
        self.cache['ln1c'] = ln_cache
     
        ffn_out = self.ffn.forward(x_c)
        
        output = _math.dot_product(ffn_out, self.Wo)
        output, ln_cache = _math.layer_norm_forward(output)
        self.cache['ln1o'] = ln_cache
        
        self.cache['ffn_out'] = ffn_out
        self.cache['output'] = output
        
        return output
    
    def backward(self, d_output, enc_output, lr):
        
        d_out_ln, _, _ = _math.backward_layer_norm(d_output, self.cache['ln1o'])
        
        ffn_out = self.cache['ffn_out']
        Wo_t = _math.transpose(self.Wo)
        
        d_ffn_out = _math.dot_product(d_out_ln, Wo_t)
        dWo = _math.dot_product(_math.transpose(ffn_out), d_out_ln)
        
        dWo = _mle.gradient_clip(dWo, 1.0)
        
        for i in range(len(self.Wo)):
            for j in range(len(self.Wo[0])):
                self.Wo[i][j] -= lr * dWo[i][j]
                
        d_cross = self.ffn.backward(d_ffn_out, lr)
        
        d_xc_add, _, _ = _math.backward_layer_norm(d_cross, self.cache['ln1c'])
        
        d_xc = d_xc_add
        d_xs_from_residual = d_xc_add
        
        ##Cross Attention
        d_self, d_enc = self.cross_attn.backward(
            d_xc, 
            self.self_attn.cache['input_q'], 
            self.cross_attn.cache['input_k'],
            lr
        )
        
        d_xs_total = _math.matrix_addition(d_xs_from_residual, d_self)
        
        d_xs_add, _, _ = _math.backward_layer_norm(d_xs_total, self.cache['ln1_s'])
        
        d_self_attn = d_xs_add
        d_tgt_input = d_xs_add
        
        ## Self Attention
        d_self_q, d_self_kv = self.self_attn.backward(d_self_attn, 
                                                      self.self_attn.cache['input_q'], 
                                                      self.self_attn.cache['input_k'], 
                                                      lr)
        
        
        return d_enc
            
        