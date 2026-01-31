import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.NeuralNetwork as NN
import BaseFunctions.BaseTokenizer as Tokenizer
import Coders.Attention
import random
import math

_math = MATH.Math()
_mle = MLE.SMLE()

class Encoder:
    
    def __init__(self, Q, K):
        
        ## Main Equation:
        # Q is the query, the positional embedding
        # K is the key, the size of the input
        # V is the value, I think the actual value....
        #
        self.key = [0.0 for _ in range(len(K))]
        self.q = [0.0 for _ in range(len(Q))]
        
    def read(self, input):
        
        attention = 
    
    def process_sequence(self, input_sequence):
        
        all_states = []
        
        for x in input_sequence:
            state = self.forward(x)
            all_states.append(state)
            
        return all_states
    
    
        
        
        

    

                
