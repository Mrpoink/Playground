import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.NeuralNetwork as NN
import BaseFunctions.BaseTokenizer as Tokenizer
import Coders.FFN as FFN
import Coders.Attention as ATT
import random
import math

_math = MATH.Math()
_mle = MLE.SMLE()

class Encoder:
    
    def __init__(self, seq_len, input_size):
        
        ## 
        # INPUT SIZE IS THE SIZE OF THE EMBEDDINGS!!!
        # true input is sequence so it will be a matrix of dimension seq_len x input_size if input_size is the size of the embeddings
        
        self.Wq = [[random.uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(input_size)]
        self.Wk = [[random.uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(input_size)]
        self.Wv = [[random.uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(input_size)]
        self.ffn = FFN.FeedForward(4, input_size = input_size)
        self.b = [[0.0] for _ in range(input_size)]
        ## Please remember that the input size if the dimension of the vectors in the layers
        ## The amount of layers is 1 here, however it would be better to probably match the maximum amount of values
        ## In the context. Maybe??
        
        self.att = ATT.Attention()
        
    def self_att(self, input):
        
        
        Q = _math.dot_product(input, self.Wq)
        K = _math.dot_product(input, self.Wk)
        V = _math.dot_product(input, self.Wv)
        
        # _math.print_matrix(input, "INPUT: ")
        # _math.print_matrix(Q, "Q: ")
        # _math.print_matrix(K, "K: ")
        # _math.print_matrix(V, "V: ")
        
        att = self.att.find_att(Q, K, V)
        
        # _math.print_matrix(att, "ATT: ")
        
        layer_norm = _math.matrix_addition(att, input)
        
        x  = _math.layer_norm(layer_norm)
        
        return x
    
    
    def process(self, input):
        
        pos_enc = _mle.position_wise(len(input), len(input[0]))
        
        input = _math.matrix_addition(input, pos_enc)
        
        attn = self.self_att(input)
        
        x = _math.matrix_addition(input, attn)
        
        ffn_out = self.ffn.forward(x)
        
        x = _math.matrix_addition(x, ffn_out)
        
        x = _math.layer_norm(x)
        
        return x
    


test_input = [
    [10.0, 10.0, 10.0, 10.0], 
    [0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1]
]

# Run your self_att
encoder = Encoder(len(test_input), len(test_input[0]))
output = encoder.process(test_input)

_math.print_matrix(output, "OUTPUT: ")
    
    
        
        
        

    

                
