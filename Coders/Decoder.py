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
    
    def __init__(self, seq_len, input_size):
        # Weights for Masked Self-Attention
        self.Wq_self = _math.generate_he_matrix(input_size, input_size)
        self.Wk_self = _math.generate_he_matrix(input_size, input_size)
        self.Wv_self = _math.generate_he_matrix(input_size, input_size)
                
        # Weights for Cross-Attention (The Bridge)
        self.Wq_cross = _math.generate_he_matrix(input_size, input_size)
        self.Wk_cross = _math.generate_he_matrix(input_size, input_size)
        self.Wv_cross = _math.generate_he_matrix(input_size, input_size)
        
        self.ffn = FFN.FeedForward(layers=4000, input_size=input_size)
        self.att = ATT.Attention()
        self.cache = {}
    
    def process(self, tgt_input, enc_output):
        # 1. Masked Self-Attention
        mask = _mle.generate_mask(len(tgt_input))
        Q_s = _math.dot_product(tgt_input, self.Wq_self)
        K_s = _math.dot_product(tgt_input, self.Wk_self)
        V_s = _math.dot_product(tgt_input, self.Wv_self)
        
        x_s, self_scores = self.att.find_att(Q_s, K_s, V_s, mask=mask)
        x_s = _math.layer_norm(_math.matrix_addition(x_s, tgt_input))

        # 2. Cross-Attention (The actual combination step)
        Q_c = _math.dot_product(x_s, self.Wq_cross)
        K_c = _math.dot_product(enc_output, self.Wk_cross) # From Encoder
        V_c = _math.dot_product(enc_output, self.Wv_cross) # From Encoder
     
        attn_out, cross_scores = self.att.find_att(Q_c, K_c, V_c)
        
        x_l = _math.layer_norm(_math.matrix_addition(x_s, attn_out))
        
        self.cache['self_scores'] = self_scores
        self.cache['Q_s'] = Q_s; self.cache['K_s'] = K_s; self.cache['V_s'] = V_s
        self.cache['tgt_input'] = tgt_input
        
        self.cache['cross_scores'] = cross_scores
        self.cache['Q_c'] = Q_c; self.cache['K_c'] = K_c; self.cache['V_c'] = V_c
        self.cache['attn_out_self'] = x_s

        # 3. FFN
        ffn_out = self.ffn.forward(x_l)
        return _math.layer_norm(_math.matrix_addition(x_l, ffn_out))
    
    def backward(self, d_output, enc_output, lr):
        
        d_ffn_in = self.ffn.backward(d_output, lr)
        
        ##Cross Attention
        dWq_c, dWk_c, dWv_c, dQ_c, dK_c, dV_c = _mle.backprop_att(
            self.cache['cross_scores'],
            self.cache['Q_c'], self.cache['K_c'], self.cache['V_c'],
            d_ffn_in, self.cache['attn_out_self']
        )
        
        d_enc_out = _math.matrix_addition(
            _math.dot_product(dK_c, _math.transpose(self.Wk_cross)),
            _math.dot_product(dV_c, _math.transpose(self.Wv_cross))
        )
        
        
        # Update Cross-Weights
        self.Wq_cross = _math.matrix_subtraction(self.Wq_cross, _math.scalar_multiply(dWq_c, lr))
        self.Wk_cross = _math.matrix_subtraction(self.Wk_cross, _math.scalar_multiply(dWk_c, lr))
        self.Wv_cross = _math.matrix_subtraction(self.Wv_cross, _math.scalar_multiply(dWv_c, lr))
        
        
        d_self_path = _math.dot_product(dQ_c, _math.transpose(self.Wq_cross))
        d_attn_in = _math.matrix_addition(d_ffn_in, d_self_path)
        
        dWq_s, dWk_s, dWv_s, dQ_s, dK_s, dV_s = _mle.backprop_att(
            self.cache['self_scores'],
            self.cache['Q_s'], self.cache['K_s'], self.cache['V_s'],
            d_attn_in, self.cache['tgt_input']
        )
        
        ## Update Self-Weights
        self.Wq_self = _math.matrix_subtraction(self.Wq_self, _math.scalar_multiply(dWq_s, lr))
        self.Wk_self = _math.matrix_subtraction(self.Wk_self, _math.scalar_multiply(dWk_s, lr))
        self.Wv_self = _math.matrix_subtraction(self.Wv_self, _math.scalar_multiply(dWv_s, lr))
                
        return d_enc_out
            
        