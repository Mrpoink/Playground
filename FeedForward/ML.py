import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import MathStuff.MathEquations as MATH
import MathStuff.StandardMLEquations as MLE
import NeuralNetwork as NN
import random

_math = MATH.Math()
_mle = MLE.SMLE()


class FeedForward:
    
    def __init__(self, layers : int):
        self.layers = [NN.Neuron() for _ in range(layers)]
        
    def get_pred(self, input : list):
        
        print("\n\n Input: \n")
        for row in input:
            print(row)
            
        a = input
        
        for layer in self.layers:
                
            z = layer.activate(a)
            a = _math.sigmoid(z)
            
            layer.output = a
            
        return a
    

ff = FeedForward(3)
input = _math.normalize([[random.random(), random.random(), random.random()], \
        [random.random(), random.random(), random.random()], \
        [random.random(), random.random(), random.random()]])
ground_truth = _math.normalize([[random.random(), random.random(), random.random()], \
                [random.random(), random.random(), random.random()], \
                [random.random(), random.random(), random.random()]])
test_nn = ff.get_pred(input)
loss = _mle.CCEL(test_nn, ground_truth)
print("\n\n--------------\n\nInput: \n")
for row in input:
    print(row)
print("\n\n--------------\nGround Truth: \n")
for row in ground_truth:
    print(row)
print("\n\n--------------\nOutput: \n")
for row in test_nn:
    print(row)
print("\n\n--------------\nLoss: \n")
print(loss)
