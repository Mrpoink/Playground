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
import random
import math

_math = MATH.Math()
_mle = MLE.SMLE()

class Encoder:
    
    def __init__(self, seq_len, input_size, n_heads=2):
        
        ## 
        # INPUT SIZE IS THE SIZE OF THE EMBEDDINGS!!!
        # true input is sequence so it will be a matrix of dimension seq_len x input_size if input_size is the size of the embeddings
        
        self.mha = ATT.MultiHead(n_heads=n_heads, d_model=input_size)
        self.ffn = FFN.FeedForward(400, input_size=input_size)
        self.cache = {}
        
    def self_att(self, input):
        
        
        Q = _math.dot_product(input, self.Wq)
        K = _math.dot_product(input, self.Wk)
        V = _math.dot_product(input, self.Wv)
        
        # _math.print_matrix(input, "INPUT: ")
        # _math.print_matrix(Q, "Q: ")
        # _math.print_matrix(K, "K: ")
        # _math.print_matrix(V, "V: ")
        
        att, soft_scores = self.att.find_att(Q, K, V)
            
            # _math.print_matrix(att, "ATT: ")
        
        self.cache = {
            'input': input,
            'Q': Q, 'K': K, 'V': V,
            'soft_scores': soft_scores
        }
        
        layer_norm = _math.matrix_addition(att, input)
        
        x, _  = _math.layer_norm_forward(layer_norm)
        
        return x
    
    
    def process(self, input):
        
        pos_enc = _mle.position_wise(len(input), len(input[0]))
        
        x = _math.matrix_addition(input, pos_enc)
        
        attn_out, scores = self.mha.forward(x, x, x)
        
        x, _ = _math.layer_norm_forward(_math.matrix_addition(x, attn_out))
        ffn_out = self.ffn.forward(x)
        
        x, _ = _math.layer_norm_forward(_math.matrix_addition(x, ffn_out))
        
        return x
    
    def backward(self, d_output, lr):
        
        d_ffn_input = self.ffn.backward(d_output, lr)
        
        d_Xq, d_Xkv = self.mha.backward(d_ffn_input, self.mha.cache['input_q'], self.mha.cache['input_k'], lr)
        
        return _math.matrix_addition(d_Xq, d_Xkv)
        


    
    
        
        
        

    

                
