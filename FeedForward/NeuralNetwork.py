import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import MathStuff.MathEquations as MATH
import random

_math = MATH.Math()

class Neuron:
    
    def __init__(self):
        self.weight = _math.normalize([[random.random(), random.random(), random.random()], \
                       [random.random(), random.random(), random.random()], \
                       [random.random(), random.random(), random.random()]])
        self.a = _math.normalize([[random.random(), random.random(), random.random()], \
                  [random.random(), random.random(), random.random()], \
                  [random.random(), random.random(), random.random()]])
        self.bias = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.output = None
        self.input = None
        
    def get_weight(self):
        return self.weight
    
    def activate(self, input : list):
        self.input = input
        self.a = _math.dot_product(self.weight, input)
        output = _math.matrix_addition(self.a, self.bias)
        return output
    
    def get_activation(self):
        return self.a
    
    def get_bias(self):
        return self.bias
    