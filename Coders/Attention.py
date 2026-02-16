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
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.d_head = d_model // n_heads
        
        
        self.cache = {}
        
        self.Wo = _math.generate_he_matrix(d_model, d_model)
        self.Wq = _math.generate_he_matrix(d_model, d_model)
        self.Wk = _math.generate_he_matrix(d_model, d_model)
        self.Wv = _math.generate_he_matrix(d_model, d_model)
        self.Wo = _math.generate_he_matrix(d_model, d_model)
        
        
    def forward(self, Q, K, V, mask=None):
        
        self.cache['input_q'] = Q
        self.cache['input_k'] = K
        self.cache['input_v'] = V
        
        Q_proj = _math.dot_product(Q, self.Wq)
        K_proj = _math.dot_product(K, self.Wk)
        V_proj = _math.dot_product(V, self.Wv)
        
        qs = _mle.split_heads(Q_proj, self.n_heads)
        ks = _mle.split_heads(K_proj, self.n_heads)
        vs = _mle.split_heads(V_proj, self.n_heads)
        
        head_outputs = []
        head_scores = []
        
        for i in range(self.n_heads):
            
            curr_k = ks[i]
            if not isinstance(curr_k[0], list): 
                # safeguard if split_heads returned 1D lists for some reason
                curr_k = [curr_k]
            
            K_t = _math.transpose(ks[i])
            
            # print(f"d_model: {self.d_model}\nn_heads: {self.n_heads}\nd_head: {self.d_head}")
            # _math.print_matrix(Q_proj, "Q Projection: ")
            # _math.print_matrix(qs, "qs: ")
            # _math.print_matrix(qs[i], "qs[i]: ")
            # _math.print_matrix(K_t, "K_t: ")
            
            qkt = _math.dot_product(qs[i], K_t)
            
            
            scaled_qkt = _mle.softmax(_math.scalar_divide(qkt, math.sqrt(self.d_head)))
            
            if mask is not None:
                scaled_qkt = _math.matrix_addition(scaled_qkt, mask)
                
            out = _math.dot_product(scaled_qkt, vs[i])
            
            head_outputs.append(out)
            head_scores.append(scaled_qkt)
            
        merged = _math.concate_heads(head_outputs)
        
        output = _math.dot_product(merged, self.Wo)
        
        
        self.cache = {
            'input_q': Q, 'input_k': K, 'input_v': V,
            'head_qs': qs, 'head_ks': ks, 'head_vs': vs,
            'head_scores': head_scores,
            'merged': merged,
            'output': output
            }
        
        return output, head_scores
    
    
    def backward(self, d_output, X_q, X_kv, lr):
        
        d_merged = _math.dot_product(d_output, _math.transpose(self.Wo))
        d_heads = _mle.split_heads(d_merged, self.n_heads)
        
        dQ_heads, dK_heads, dV_heads = [], [], []
        
        for i in range(self.n_heads):
            
            _, _, _, dQ_h, dK_h, dV_h = _mle.backprop_att(
                self.cache['head_scores'][i],
                self.cache['head_qs'][i], self.cache['head_ks'][i], self.cache['head_vs'][i],
                d_heads[i], 
                _mle.split_heads(X_q, self.n_heads)[i],
                _mle.split_heads(X_kv, self.n_heads)[i]
            )
            dQ_heads.append(dQ_h); dK_heads.append(dK_h); dV_heads.append(dV_h)
            
        dQ_full = _math.concate_heads(dQ_heads)
        dK_full = _math.concate_heads(dK_heads)
        dV_full = _math.concate_heads(dV_heads)
        
        d_Xq = _math.dot_product(dQ_full, _math.transpose(self.Wq))
        d_Xkv = _math.matrix_addition(
            _math.dot_product(dK_full, _math.transpose(self.Wk)),
            _math.dot_product(dV_full, _math.transpose(self.Wv))
        )
        
        dWq = _math.dot_product(_math.transpose(X_q), dQ_full)
        self.Wq = _math.matrix_subtraction(self.Wq, _math.scalar_multiply(dWq, lr))
        
        dWk = _math.dot_product(_math.transpose(X_kv), dK_full)
        self.Wk = _math.matrix_subtraction(self.Wk, _math.scalar_multiply(dWk, lr))
        
        dWv = _math.dot_product(_math.transpose(X_kv), dV_full)
        self.Wv = _math.matrix_subtraction(self.Wv, _math.scalar_multiply(dWv, lr))
        
        dWo = _math.dot_product(_math.transpose(self.cache['merged']), d_output)
        self.Wo = _math.matrix_subtraction(self.Wo, _math.scalar_multiply(dWo, lr))
        
        return d_Xq, d_Xkv