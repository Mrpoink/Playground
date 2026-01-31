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
    
    def find_attention(Q, K, V):
        
        
        qkt = [[0.0 for _ in range(len(Q))] for _ in range(len(K))]
        
        for i in range(len(Q)):
            for j in range(len(V)):
                
                qkt[i][j]