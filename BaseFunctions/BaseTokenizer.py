import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from BaseFunctions.StandardMLEquations import SMLE
from BaseFunctions.MathEquations import Math
import random

_mle = SMLE()
_math = Math()

class GLoVE_Tokenizer:
    
    def __init__(self):
        
        self.embeddings_dict = {}
        
        with open('GLoVE/GLoVEVocab.txt', 'r') as file:
            for line in file:
                values = line.split()
                word = values[0]
                try:
                    # Attempt to convert; if '1/2' appears, this will trigger the except block
                    vector = [float(x) for x in values[1:]]
                    self.embeddings_dict[word] = vector
                except ValueError:
                    # Optional: Handle specific cases like '1/2' by splitting/dividing
                    # For now, skipping malformed lines is safer for your Transformer
                    continue
                
    def tokenize(self, input):
        
        final_embeddings = []
        
        words = input.lower().split()
        
        for word in words:
            if word in self.embeddings_dict:
                final_embeddings.append(self.embeddings_dict[word])
            else:
                final_embeddings.append([0.0] * 100)
            
            
        return final_embeddings