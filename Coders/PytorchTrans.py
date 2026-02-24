import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.NeuralNetwork as NN
import BaseFunctions.BaseTokenizer as Tokenizer
import BaseFunctions.BaseTokenizer as BT
import Coders.PytorchAtt as ATT
import Coders.PytorchEncoder as ENC
import Coders.PytorchDecoder as DEC
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math

_math = MATH.Math()
_mle = MLE.SMLE()


class Transformer(nn.Module):
    def __init__(self, input_size, n_heads):
        super().__init__()
        
        self.encoder = ENC.Encoder(input_size, n_heads)
        self.decoder = DEC.Decoder(input_size, n_heads)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, mask=src_mask)
        
        final_out = self.decoder(tgt, enc_out, tgt_mask=tgt_mask)
        
        return final_out
    
class Utilities:
    def __init__(self, model : Transformer, optimizer, critereon):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_func = critereon
        
        
    def train(self, epochs, src, tgt):
        
        self.model.train()
        
        
        for epoch in range(epochs):
            
            self.optimizer.zero_grad()
            
            prediction = self.model(src, tgt)
            
            loss = self.loss_func(prediction, tgt)
            
            loss.backward()
            
            self.optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                
    def inference(self, tokenizer, src : torch.tensor, max_len=5):
        self.model.eval()
        
        
        size = self.model.encoder.mha.d_model
        current_tgt = torch.zeros((1, 1, size)) 
        
        output_words = []

        with torch.no_grad():
            for _ in range(max_len):
                # Forward pass: Encoder looks at src, Decoder looks at what we've built so far
                prediction = self.model(src, current_tgt)
                
                # Take the LAST predicted vector in the sequence
                next_token_vec = prediction[0, -1, :]
                
                # Detokenize to get the word
                word = tokenizer.detokenize(next_token_vec)
                output_words.append(word)
                
                # Append the new vector to the decoder input for the next iteration
                next_token_vec = next_token_vec.unsqueeze(0).unsqueeze(0)
                current_tgt = torch.cat([current_tgt, next_token_vec], dim=1)
                
                # Break if you implement an <EOS> token or stop condition
                if word == ".": break 

        return " ".join(output_words)


tokenizer = BT.GLoVE_Tokenizer()

src = tokenizer.tokenize('I am')
tgt = tokenizer.tokenize('Me, and you are you, and we are we.')

src_tensor = tokenizer.to_tensor(src)
tgt_tensor = tokenizer.to_tensor(tgt)      
                
batch_size = 1
seq_len = 5
input_dim = len(src[0])

model = Transformer(input_size=input_dim, n_heads=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

trainer = Utilities(model, optimizer, criterion)
trainer.train(epochs=50000, src=src_tensor, tgt=tgt_tensor)

pred = trainer.inference(tokenizer, src_tensor)

print(pred)

