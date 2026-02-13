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
        
        for layer in self.layers:
                
            z = layer.activate(a)
            a = _math.sigmoid(z)
            
            layer.output = a
            
        return a
    
    def train(self, train : list, test : list, epochs : int, learning_rate):
        
        best_loss = 50000.0
        best_epoch = 0
        current_loss = 49999
        
        lrs = AVS.AV_Scheduler(learning_rate)
        
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                
                y_hat = self.forward(train)
                    
                error = _mle.MSE_loss_der(y_hat, test)
                current_loss = _math.frobenius_norm(error)
                lrs.step(current_loss)
                
                if epoch % 1 == 0:
                    if (current_loss < best_loss):
                        best_loss = current_loss 
                        best_epoch = epoch
                        self.best_layers = self.layers
                    print(f"Epoch: {epoch} Loss: {current_loss}")
                
                
                i = len(self.layers) - 1
                while i > 0:
                    is_final = (i == 0)
                    error = _mle.gradient_clip(error, threshold=1.0)
                    error = _mle.BackPropagation_Step(self.layers[i], 
                                                      error,
                                                      lrs.lr, 
                                                      is_final=is_final
                                                      )
                    i-=1
                    
                pbar.update(1)
        print("Best loss: ", best_loss)
        print("Best epoch: ", best_epoch)
    




# input = _math.normalize([[random.random(), random.random(), random.random()], \
#         [random.random(), random.random(), random.random()], \
#         [random.random(), random.random(), random.random()],
#         [random.random(), random.random(), random.random()], \
#         [random.random(), random.random(), random.random()], \
#         [random.random(), random.random(), random.random()]])
# ground_truth = _math.normalize([[random.random(), random.random(), random.random()], \
#         [random.random(), random.random(), random.random()], \
#         [random.random(), random.random(), random.random()],
#         [random.random(), random.random(), random.random()], \
#         [random.random(), random.random(), random.random()], \
#         [random.random(), random.random(), random.random()]])

# ff = FeedForward(32, input_size=len(input))

# print("Trying to train....")
# ff.train(input, ground_truth, epochs=500, learning_rate=0.01)

# ff.layers = ff.best_layers

# print(ff.forward(input))

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