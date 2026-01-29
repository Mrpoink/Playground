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

class Encoder:
    
    def __init__(self, hidden_size, input_size):
        
        ## Main Equation:
        # h_t = tanh((W * x_t) + (U * h_t-1) + b)
        # W = input_weight
        # U = hidden_weight
        # b = bias
        #
        #
        self.hidden_weight = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.weight = [[random.uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(hidden_size)]
        self.bias = [[0.0] for _ in range(hidden_size)]
        
        self.input = None
        self.output = None
        self.state = None
        
    def forward(self, x):
        ## This is a basic forward pass function
        ## Such as the one in the OG RNN class
        
        if self.state is None:
            self.state = [[0.0] for _ in range(len(self.hidden_weight))]
        
        wx = _math.dot_product(self.weight, x)
        uh = _math.dot_product(self.hidden_weight, self.state)
        
        htan = _math.matrix_addition(_math.matrix_addition(wx, uh), self.bias)
        
        ht = _math.matrix_tanh(htan)
        
        self.state = ht
        
        return ht
    
    def process_sequence(self, input_sequence):
        
        all_states = []
        
        for x in input_sequence:
            state = self.forward(x)
            all_states.append(state)
            
        return all_states
    
    
        
        
        

    

                
