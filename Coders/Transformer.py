import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.AVS as AVS
import BaseFunctions.BaseTokenizer as BT

_math = MATH.Math()
_mle = MLE.SMLE()

import Coders.Encoder as CE
import Coders.Decoder as CD

class Transformer:
    def __init__(self, seq_len, input_size, n_heads=2):
        self.encoder = CE.Encoder(seq_len, input_size, n_heads=n_heads)
        self.decoder = CD.Decoder(seq_len, input_size, n_heads=n_heads)
        
    def forward(self, src_input, tgt_input):
        # Encoder processes the source
        enc_out = self.encoder.process(src_input)
        
        # Decoder processes the target while looking at the source
        final_output = self.decoder.process(tgt_input, enc_out)
        
        return final_output
    
    def train(self, src, tgt, expected_output, epochs, max_lr):
        
        best_loss = 50000.0
        best_epoch = 0
        current_loss = 49999
        
        min_lr = max_lr / epochs
        
        for epoch in range(epochs):
            
            lr = _mle.cosine_scheduler(epoch, epochs, max_lr, min_lr)
            
            enc_out = self.encoder.process(src) 
            prediction = self.decoder.process(tgt, enc_out)
            
            current_loss = _mle.MSE_loss(prediction, expected_output)
                
            if epoch % 1 == 0:
                if (current_loss < best_loss):
                    best_loss = current_loss 
                    best_epoch = epoch
                print(f"Epoch: {epoch} Loss: {current_loss}")
            
            d_output = _mle.MSE_loss_der(prediction, expected_output)
            
            d_encoder_error = self.decoder.backward(d_output, enc_out, lr)
            
            self.encoder.backward(d_encoder_error, lr)
            
        print("Best loss: ", best_loss)
        print("Best epoch: ", best_epoch)
        
    



tokenizer = BT.GLoVE_Tokenizer()

src = tokenizer.tokenize('I am')
tgt = tokenizer.tokenize('Me')
input_dim = len(src[0])
seq_len = len(src)

model = Transformer(seq_len, input_dim, n_heads=50)
# 2. Dummy Data

epochs = 10000
lr = 0.001

model.train(src, tgt, tgt, epochs, lr)

final = model.forward(src, tgt)

_math.print_matrix(final, "FINAL: ")