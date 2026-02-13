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
                
            z = self.layers[i].activate(a)
            a = _math.relu(z)
            
            self.layers[i].output = a
            
        return a