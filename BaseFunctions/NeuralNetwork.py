import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import random

_math = MATH.Math()

class Neuron:
    
    def __init__(self, input_size=3):
        self.input_size = input_size
        
        #self.a = _math.normalize(self._generate_random_matrix(hidden_size, hidden_size))
        
        self.weight = [[random.uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(input_size)]
        self.bias = [[0.0] for _ in range(input_size)]
        
        self.output = None
        self.input = None
        self.state = None
        self.z = None
        
        
    def get_weight(self):
        return self.weight
    
    def activate(self, input : list):
        
        self.input = input
        # W: (H x I), a: (I x B)
        z_linear = _math.matrix_multiply(self.weight, input)  # (H x B)
        
        B = len(z_linear[0])
        
        b_tiled = [row * B for row in self.bias]  # (H x B)

        self.z = _math.matrix_addition(z_linear, b_tiled)
        
        self.a = input
        
        return self.z
    
    def activation_at_time(self, input : list, prev_state):
        #_math.print_matrix(input, "activation input")
        z = self.rnn_z(self.weight, self.hidden_weight, self.bias, input, prev_state)
        self.state = _math.matrix_tanh(z)
         
        return  self.state
    
    def get_activation(self):
        return self.a
    
    def get_bias(self):
        return self.bias
        
        
        
    