import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import MathStuff.MathEquations as MATH
import MathStuff.StandardMLEquations as MLE
import NeuralNetwork as NN

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
        
        for row in self.layers:
            for col in self.layers:
                
                z = col.activate(a)
                a = _math.sigmoid(z)
                
                col.output = a
            
        return a
    

ff = FeedForward(3)
test_nn = ff.get_pred([[0.231, 0.562, 0.893], [0.894, 0.565, 0.236], [0.568, 0.899, 0.2304]])
print("\n\n--------------\nOutput: \n")
for row in test_nn:
    print(row)