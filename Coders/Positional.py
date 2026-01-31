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

class PE:
    
    def __init__(self, d_model):
        r'''
        d_model is the size of the dense vector of embeddings'''
        
        self.size = d_model
        self.matrix = [0.0 for _ in range(d_model)]
        
    def create_pe(self, embeddings):
        
        for i in range(len(embeddings)):
            for j in range(len(embeddings[0])):
                
                if ((2 * j) + 1) < len(embeddings[0]):
                
                    self.matrix[i][2 * j] = math.sin(embeddings[i][j] / (10000 ** ((2 * i) / self.size)))
                    self.matrix[i][(2 * j) + 1] = math.cos(embeddings[i][j] / (10000 ** ((2 * i) / self.size)))
                
                else:
                    return self.matrix