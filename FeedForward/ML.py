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
        
    def forward(self, input : list):
            
        a = input
        
        for layer in self.layers:
                
            z = layer.activate(a)
            a = _math.sigmoid(z)
            
            layer.output = a
            
        return a
    
    def train(self, train : list, test : list, epochs : int, learning_rate):
        
        for epoch in range(epochs):
            
            y_hat = self.forward(train)
            
            if epoch % 10 == 0:
                current_loss = _mle.CCEL(y_hat, test)
                print(f"Epoch: {epoch} Loss: {current_loss}")
                
            error = _mle.cce_deriv(y_hat, test)
            
            for layer in reversed(self.layers):
                
                error = _mle.BackPropogation_Step(layer, error, learning_rate)
    

ff = FeedForward(33)


input = _math.normalize([[random.random(), random.random(), random.random()], \
        [random.random(), random.random(), random.random()], \
        [random.random(), random.random(), random.random()]])
ground_truth = _math.normalize([[random.random(), random.random(), random.random()], \
                [random.random(), random.random(), random.random()], \
                [random.random(), random.random(), random.random()]])

print("Trying to train....")
ff.train(input, ground_truth, epochs=45, learning_rate=0.1)



# print("\n\n--------------\n\nInput: \n")
# for row in input:
#     print(row)
# print("\n\n--------------\nGround Truth: \n")
# for row in ground_truth:
#     print(row)
# print("\n\n--------------\nLayers: \n")
# print(ff.layers)
# num_layers = 0

# comparitive_weight_matrix_1 = []
# comparitive_activation_1 = []



# for item in ff.layers:
#     print(f"\n\n--------\nLayer {num_layers + 1}: \n")
#     print(f"\nACTIVATION: \n")
#     for row in item.a:
#         comparitive_activation_1 = item.a
#         print(row)
    
#     print(f"\nWEIGHT: \n")
#     for row in item.weight:
#         comparitive_weight_matrix_1 = item.weight
#         print(row)
    
#     print(f"\nBIAS: \n")
#     for row in item.bias:
#         print(row)

#     print(f"\nOUTPUT: \n")
#     for row in item.output:
#         print(row)
        
#     num_layers += 1
# print("\n\n--------------\nOutput: \n")
# for row in ff.layers[0].output:
#     print(row)

# print("\n\n--------------\nBackward Pass: \n")
# print("\n\n--------------\nLayers: \n")
# print(backward_pass)
# num_layers = 0

# comparitive_weight_matrix_2 = []
# comparitive_activation_2 = []

# for item in backward_pass:
#     print(f"\n\n--------\nLayer {num_layers + 1}: \n")
#     print(f"\nACTIVATION: \n")
#     for row in item.a:
#         comparitive_activation_2 = item.a
#         print(row)
    
#     print(f"\nWEIGHT: \n")
#     for row in item.weight:
#         comparitive_weight_matrix_2 = item.weight
#         print(row)
    
#     print(f"\nBIAS: \n")
#     for row in item.bias:
#         print(row)

#     print(f"\nOUTPUT: \n")
#     for row in item.output:
#         print(row)
        
#     num_layers += 1
    
# print(f"\n\n\n-----------\nComparitive Analysis: ")
# print("\n------\nBefore: \n")
# print("Activation: \n")
# for row in comparitive_activation_1:
#     print(row)

# print("Weight: \n")
# for row in comparitive_weight_matrix_1:
#     print(row)
    
# print("\n------\nAfter: \n")
# print("Activation: \n")
# for row in comparitive_activation_2:
#     print(row)

# print("Weight: \n")
# for row in comparitive_weight_matrix_2:
#     print(row)