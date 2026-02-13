import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.NeuralNetwork as NN
import BaseFunctions.BaseTokenizer as Tokenizer
import random
import math

_math = MATH.Math()
_mle = MLE.SMLE()


class Attention:
    
    def __init__(self):
        pass
    
    def find_att(self, Q, K, V, mask=None):
        
        K_t = _math.transpose(K)
        
        qkt = _math.dot_product(Q, K_t)
        
        d_k = len(Q[0])
        
        scaled_qkt = _mle.softmax(_math.scalar_divide(qkt, math.sqrt(d_k)))
        
        
        if mask is not None:
            scaled_qkt = _math.matrix_addition(scaled_qkt, mask)
        
        return _math.dot_product(scaled_qkt, V), scaled_qkt
    
    
class MultiHead:
    
    def __init__(self, n_heads, d_model):
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.d_head = d_model // n_heads
        
        self.Wo = _math.generate_he_matrix(d_model, d_model)
        self.cache = {}
        
        
    def forward(self, Q, K, V, mask=None):
        
        qs = _mle.split_heads(Q, self.n_heads)
        ks = _mle.split_heads(K, self.n_heads)
        vs = _mle.split_heads(V, self.n_heads)
        
        head_outputs = []
        head_scores = []
        
        for i in range(self.n_heads):
            
            K_t = _math.transpose(ks[i])
            qkt = _math.dot_product(qs[i], K_t)
            
            scaled_qkt = _mle.softmax(qkt, math.sqrt(self.d_head))
            
            if mask is not None:
                scaled_qkt = _math.matrix_addition(scaled_qkt, mask)
                
            out = _math.dot_product(scaled_qkt, vs[i])
            
            head_outputs.append(out)
            head_scores.append(head_scores)
            
        merged = _math.vert_concat(head_outputs)
        
        output = _math.dot_product(merged, self.Wo)
        
        
        self.cache = {
            'head_qs': qs, 'head_ks': ks, 'head_vs': vs,
            'head_scores': head_scores,
            'merged': merged
        }
        
        return output, head_scores
    
    
    def backward(self, d_output, X_q, X_kv, lr):
        
        dWo = _math.dot_product(_math.transpose(self.cache['merged']), d_output)
        
        d_merged = _math.dot_product(d_output, _math.transpose(self.Wo))
        
        d_heads = _mle.split_heads(d_merged, self.n_heads)
        
        total_dWq = _math.get_blank_matrix(self.d_model, self.d_model)
        total_dWk = _math.get_blank_matrix(self.d_model, self.d_model)
        total_dWv = _math.get_blank_matrix(self.d_model, self.d_model)
        
        
        for i in range(self.n_heads):
            dWq_h, dWk_h, dWv_h, dQ_h, dK_h, dV_h = _mle.backprop_att(
                self.cache['head_scores'][i],
                self.cache['head_qs'][i], self.cache['head_ks'][i], self.cache['head_vs'][i],
                d_heads[i], 
                _math.split_heads(X_q, self.n_heads)[i],
                _math.split_heads(X_kv, self.n_heads)[i]
            )
            
        self.Wo = _math.matrix_subtraction(self.Wo, _math.scalar_multiply(dWo, lr))
        
        return d_enc_out