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

class Tokenizer:
    def __init__(self, embedding_dim = 16):
        self.vocab_size = 0
        self.embedding_dim = embedding_dim

    def train(self, text: str):
        # 1. Find all unique characters in your text
        
        unique_chars = sorted(list(set(text)))
        self.vocab_size = len(unique_chars)
        self.char_to_int = {char: i for i, char in enumerate(unique_chars)}
        self.int_to_char = {i: char for i, char in enumerate(unique_chars)}

        raw_table = [[random.uniform(-0.1, 0.1) for _ in range(self.embedding_dim)] for _ in range(self.vocab_size)]
        self.lookup_table = raw_table
        
        
        print(f"Vocab Size: {self.vocab_size}")

    def encode(self, text: str):
        """Converts string to vectors"""
        
        return [self.char_to_int[ch] for ch in text if ch in self.char_to_int]

    def decode(self, vectors: list):
        """Converts list of integers -> token_ids"""
        
        output = ""
        for item in vectors:
            output = output + " " + self.int_to_char[item]          
        return output
    
    def lookup(self, token_id):
        if token_id < 0 or token_id >= self.vocab_size:
            raise IndexError(f"token id {token_id} out of range 0..{self.vocab_size-1}")
        return self.lookup_table[token_id]

class Embedding(Tokenizer):
    def __init__(self, embedding_dim = 16):
        ## Create embeddings, and train them ironically
        self.embedding_dim = embedding_dim
        pass
       
    def get_embedding(self, token_id):
        
        ## Looks up the embedding in the existing table
        _math.print_matrix([self.lookup_table[token_id]], "Embedding")
        return [self.lookup_table[token_id]]
    
    def update_embedding(self, token_id, error, learning_rate):
        
        ## Updates like a weight does
        error = [row[0] for row in error]
        
        
        for i in range(self.embedding_dim):
                
            self.lookup_table[token_id][i] -= (error[i] * learning_rate)
                
    def create_embeddings(self, text):
        
        ids = self.encode(text)
        
        return self.get_embedding(ids)
                
