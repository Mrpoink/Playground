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


class Attention:
    
    def __init__(self):
        pass
    
    def find_attention(self, v_a, prev_state, weight, hidden_weight, all_states):
        
        ## Must run over all states individually instead of
        ## All at once
        
        ## Dimension setups:
        # hidden_weights = H x H
        # weights = I x I
        # bias = 1 x I
        # all_states = T x H (where t = input_seq_length)
        # prev_state = I x I (maybe?)
        
        # W_a * s{i-1}
        decoder_portion = _math.dot_product(weight, prev_state)
        
        z = []
        
        for h_j in all_states:
            
            # U_a * h_j
            encoder_portion = _math.dot_product(hidden_weight, h_j)
            
            #Tanh(W_a*s + U_a*h)
            activation = _math.matrix_addition(decoder_portion, encoder_portion)
            activation_tanh = _math.matrix_tanh(activation)
            
            #v_a * Tanh(activation_tanh) -> scalar score
            # Energy; e_ij
            activation_tanh = _math.transpose(activation_tanh)
            score = _math.dot_product(activation_tanh, v_a)
            
            if isinstance(score, list):
                score = score[0] #Extract scalar
                
            z.append(score)
            
        # Return probabilities in list, not just one score
        return _mle.softmax(z)
    
    def find_context(self, attention, all_states):
        ## c_j = sum(aij * hj)
        # Attention: 1 x T
        # all_states = H x T
        # return: 1 x H
        
        hidden_size = len(all_states[0])
        context = [0.0] * hidden_size
        
        
        for j in range(len(all_states)):
            
            scalar_weight = attention[j]
            
            h_j = all_states[j]
            
            for k in range(hidden_size):
                context[k] += (scalar_weight * h_j[k][0])
                
        return context
        