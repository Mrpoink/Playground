import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.NeuralNetwork as NN
import BaseFunctions.AVS as AVS
import random
from tqdm import tqdm

_math = MATH.Math()
_mle = MLE.SMLE()


class FeedForward:
    
    def __init__(self, layers : int, input_size = 3):
        self.layers = [NN.Neuron(input_size=input_size) for _ in range(layers)]
        self.best_layers = []
        
    def forward(self, input : list):
            
        a = input
        
        for i in range(len(self.layers)):
            
            self.layers[i].input = a
                
            z = self.layers[i].activate(a)
            a = _math.relu(z)
            
            self.layers[i].output = a
            self.layers[i].z = z
            
        return a
    
    def backward(self, d_output, lr):
        
        d_a = d_output
        
        for i in reversed(range(len(self.layers))):
            
            layer = self.layers[i]
            
            d_z = _math.hadamard_product(d_a, _math.relu_der(layer.z))
            
            d_w = _math.dot_product(_math.transpose(layer.input), d_z)
            
            d_b = [[sum(col) for col in _math.transpose(d_z)]]
            d_b = _math.transpose(d_b)
            
            d_a = _math.dot_product(d_z, _math.transpose(layer.weight))
            
            update = _math.scalar_multiply(d_w, lr)
            
            layer.weight = _math.matrix_subtraction(layer.weight, update)
            layer.bias = _math.matrix_subtraction(layer.bias, _math.scalar_multiply(d_b, lr))
            
        return d_a
            
            