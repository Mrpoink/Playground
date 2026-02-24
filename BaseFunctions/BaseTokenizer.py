import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from BaseFunctions.StandardMLEquations import SMLE
from BaseFunctions.MathEquations import Math
import random
import torch

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
        
        self.dim =  len(next(iter(self.embeddings_dict.values())))
        
        if '<SOS>' not in self.embeddings_dict:
            self.embeddings_dict['<SOS>'] = [random.uniform(-0.1, 0.1) for _ in range(self.dim)]
            self.embeddings_dict['<EOS>'] = [random.uniform(-0.1, 0.1) for _ in range(self.dim)]
                
    def tokenize(self, input, add_special = True):
        
        final_embeddings = []
        
        words = input.lower().split()
        
        if add_special:
            words = ['<SOS>'] + words + ['<EOS>']
        
        
        for word in words:
            if word in self.embeddings_dict:
                final_embeddings.append(self.embeddings_dict[word])
            else:
                final_embeddings.append([0.0] * self.dim)
            
            
        return final_embeddings
    
    def detokenize(self, vector):
        """Finds the word in the dictionary closest to the given vector."""
        best_word = "<UNK>"
        min_dist = float('inf')
        
        # If input is a tensor, convert to list for comparison
        if isinstance(vector, torch.Tensor):
            vector = vector.tolist()

        for word, emb in self.embeddings_dict.items():
            # Euclidean distance calculation
            dist = sum((v - e) ** 2 for v, e in zip(vector, emb))
            if dist < min_dist:
                min_dist = dist
                best_word = word
        return best_word
    
    def to_tensor(self, input):
        
        return torch.tensor(input, dtype=torch.float32).unsqueeze(0)