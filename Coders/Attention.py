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
    
    def find_att(self, Q, K, V):
        
        K_t = _math.transpose(K)
        
        qkt = _math.dot_product(Q, K_t)
        
        d_k = len(Q[0])
        
        scaled_qkt = _mle.softmax(_math.scalar_divide(qkt, math.sqrt(d_k)))
        
        return _math.dot_product(scaled_qkt, V)
        
    