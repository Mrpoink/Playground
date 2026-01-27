import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE
import BaseFunctions.NeuralNetwork as NN
import random
from tqdm import tqdm

_math = MATH.Math()
_mle = MLE.SMLE()

class RNN:
    
    def __init__(self, hidden_layers : int, input_size : int):
        self.neuron = NN.Neuron(input_size=input_size, hidden_size=hidden_layers)
        self.LSTM_Neuron = NN.LSTM(input_size=input_size, hidden_size=hidden_layers)
        
        self.current_hidden_state = _mle.get_blank_matrix(hidden_layers, 1, 0.0)
        self.current_memory = _mle.get_blank_matrix(hidden_layers, 1, 0.0)
        
        self.final_state = []
        
    def forward(self, input : list):
        history = []
        
        # print("input: ", len(input), len(input[0]))
        
        for time in input:
            
            # print(len(time))
            new_state = self.neuron.activation_at_time(time, self.current_hidden_state)
            
            self.current_hidden_state = new_state
            
            self.final_state = new_state
            
            history.append(new_state)
                
        return self.current_hidden_state, history
    
    def LSTM_forward(self, input : list):
        
        history = []
        caches = []
        
        for x in input:
            
            # print("X: ", x)
            
            new_state, new_memory, cache = self.LSTM_Neuron.activate_step(x, self.current_hidden_state, self.current_memory)
            
            self.current_hidden_state = new_state
            self.current_memory = new_memory
            
            self.final_state = new_state
            
            history.append(new_state)
            caches.append(cache)
            

        return self.current_hidden_state, history, caches
   
    def train_LSTM(self, train : list, test : list, epochs : int, learning_rate = 1e-5):
        
        for epoch in range(epochs):
            
            self.current_hidden_state = _mle.get_blank_matrix(self.LSTM_Neuron.hidden_size, 1, 0.0)
            self.current_memory = _mle.get_blank_matrix(self.LSTM_Neuron.hidden_size, 1, 0.0)
            state, history, caches = self.LSTM_forward(train)
            
            loss = _mle.MSE_loss(state, test)
            dh_n1 = _math.matrix_subtraction(state, test)
            dC_n1 = _mle.get_blank_matrix(1, self.LSTM_Neuron.hidden_size, 0)
            
            T = len(train)
            
            for t in reversed(range(T)):
                cache = caches[t]
                
                # Get gradients for this step
                grads, dh_prev, dC_prev = self.LSTM_Neuron.BackPropagation_Step_RNN_LSTM(dh_n1, dC_n1, cache)
                
                # Apply Updates Immediately (Stochastic) or Accumulate 
                # Here we Apply Immediately for simplicity
                def apply(W, U, b, g_tuple):
                    dW, dU, db = g_tuple
                    W_new = _math.matrix_subtraction(W, _math.scalar_multiply(dW, learning_rate))
                    U_new = _math.matrix_subtraction(U, _math.scalar_multiply(dU, learning_rate))
                    b_new = _math.matrix_subtraction(b, _math.scalar_multiply(db, learning_rate))
                    return W_new, U_new, b_new

                # Update Forget Gate
                self.LSTM_Neuron.F_w, self.LSTM_Neuron.F_h, self.LSTM_Neuron.F_b = \
                    apply(self.LSTM_Neuron.F_w, self.LSTM_Neuron.F_h, self.LSTM_Neuron.F_b, grads["F"])
                
                # Update Input Gate
                self.LSTM_Neuron.I_w, self.LSTM_Neuron.I_h, self.LSTM_Neuron.I_b = \
                    apply(self.LSTM_Neuron.I_w, self.LSTM_Neuron.I_h, self.LSTM_Neuron.I_b, grads["I"])

                # Update Candidate Gate
                self.LSTM_Neuron.C_w, self.LSTM_Neuron.C_h, self.LSTM_Neuron.C_b = \
                    apply(self.LSTM_Neuron.C_w, self.LSTM_Neuron.C_h, self.LSTM_Neuron.C_b, grads["C"])

                # Update Output Gate
                self.LSTM_Neuron.O_w, self.LSTM_Neuron.O_h, self.LSTM_Neuron.O_b = \
                    apply(self.LSTM_Neuron.O_w, self.LSTM_Neuron.O_h, self.LSTM_Neuron.O_b, grads["O"])
                
                # Pass error back
                dh_n1 = dh_prev
                dC_n1 = dC_prev
                
            print(f"Epoch {epoch} LSTM Loss: {loss}")
                
    
    def train_forward(self, train : list, test : list, epochs : int, learning_rate = 1e-5):
        
        for epoch in range(epochs):
        
            state, history = self.forward(train)
            
            pre_state = _mle.get_blank_matrix(1, self.neuron.hidden_size, 0)
            history = [pre_state] + history
            
            error = _mle.MSE_loss_matrix(state, test)
            
            grad_w = [[0] for _ in range(self.neuron.hidden_size)]
            grad_h = _mle.get_blank_matrix(self.neuron.hidden_size, self.neuron.hidden_size, 0)
            grad_b = [[0] for _ in range(self.neuron.hidden_size)]
            
            T = len(train)
            
            for t in reversed(range(T)):
                
                input_at_time = train[t]
                output_at_time = history[t+1]
                h_t_1 = history[t]
                
                gw, gh, gb, dh_t_1 = _mle.BackPropagation_Step_RNN(
                    error,
                    input_at_time,
                    h_t_1,
                    output_at_time,
                    self.neuron.hidden_weight
                )
                
                grad_w = _math.matrix_addition(grad_w, gw)
                grad_h = _math.matrix_addition(grad_h, gh)
                grad_b = _math.matrix_addition(grad_b, gb)
                
                error = dh_t_1
                
            step_w = _math.scalar_multiply(grad_w, learning_rate)
            self.neuron.weight = _math.matrix_subtraction(self.neuron.weight, step_w)
            
            step_h = _math.scalar_multiply(grad_h, learning_rate)
            self.neuron.hidden_weight = _math.matrix_subtraction(self.neuron.hidden_weight, step_h)
            
            step_b = _math.scalar_multiply(grad_b, learning_rate)
            self.neuron.bias = _math.matrix_subtraction(self.neuron.bias, step_b)
            
            overall_error = _mle.MSE_loss(state, test)
            print(f"Epoch {epoch} Error: {overall_error}")
    


## I want you to run this to show the effects of explosive gradients
## We are achieving this result due to the constant multiplication
## And running off the previous weights. 
## Take a look at the tanh equation and usage in the MATH class
## We are adding and subtracting numbers yes; however,
## As we multiply the positive weight with the positive inputs
## We get a positive output, that is then added to the previous step
## (See NeuralNetwork.activate_at_time)
## And we continue to do this through the chain of neurons
## Hence, an explosive gradient

rnn_forward = RNN(input_size=1, hidden_layers=3)

input = [
    [0.25], 
    [0.75],
    [0.65]
    ]

output = [
    [0.75],
    [0.25],
    [0.25]
]

final_state, history = rnn_forward.forward(input)

print(f"\n--------\nFinal State: \n")
for row in final_state:
    print(row)

print(f"\n--------\nHistory: \n")
for item in history:
    print(item)


## This is the logic for the Long Short Term Memory as designed
## The design is drawn in my notes regarding Attention is All You Need
## I think you can guess what my next implementation will be (after propogation)


print(f"\n\n----------\nLSTM Implementation: \n")
    
rnn_lstm = RNN(input_size=1, hidden_layers=3)

final_state, history, caches = rnn_lstm.LSTM_forward(input)

print(f"\n--------\nFinal State: \n")
for row in final_state:
    print(row)

print(f"\n--------\nHistory: \n")
for item in history:
    print(item)
    
print("\n\n-----------\nTraining (Standard Implementation): \n")
rnn_forward.train_forward(input, output, 200, 0.1)

final_state_forward, history = rnn_forward.forward(input)



print("\n\n-----------\nTraining (LSTM Implementation): \n")
rnn_lstm.train_LSTM(input, output, 1600, 0.1)

final_state_lstm, history, caches = rnn_lstm.LSTM_forward(input)


timesteps = random.randint(1, 16)  # avoid 0-length

long_input = [[random.uniform(-1.0, 1.0)] for _ in range(timesteps)]
long_output = [[random.uniform(-1.0, 1.0)] for _ in range(timesteps)]

print(len(long_input), len(long_input[0]))  # T, 1
print(len(long_output), len(long_output[0]))

sorted_input = sorted(input, key=lambda r: r[0])

# Model dims: input_size=1 (feature size), hidden_layers=H
H = 3
rnn_lstm_long = RNN(input_size=1, hidden_layers=len(long_input))
rnn_long = RNN(input_size = 1, hidden_layers=len(long_input))

# train_LSTM expects test to be H x 1; provide a target vector
target = _mle.get_blank_matrix(1, H, 0.0)  # e.g., zeros

rnn_lstm_long.train_LSTM(long_input, long_output, 600, 0.01)
rnn_long.train_forward(long_input, long_output, 600, 0.01)

rnn_lstm_long_final_state, history, caches = rnn_lstm_long.LSTM_forward(long_input)

rnn_long_final_state, history = rnn_long.forward(long_input)

print(f"\n--------\nStandard Test:")
print("\nStandard: \n")
for row in output:
    print(row)

print(f"\n--------\nFinal State (Standard): \n")
for row in final_state_forward:
    print(row)
    
print("(Standard) Forward Loss: ", _mle.MSE_loss(final_state_forward, output))

print(f"\n--------\nFinal State (LSTM): \n")
for row in final_state_lstm:
    print(row)

print("(LSTM) Forward Loss: ", _mle.MSE_loss(final_state_lstm, output))
    
print("\n---------\nLong Test: \n")
for row in long_output:
    print(row)
    
print(f"\n--------\nFinal State (LSTM-Long Context): \n")
for row in rnn_lstm_long_final_state:
    print(row)
    
print("LSTM Long Loss (Final): ", _mle.MSE_loss(rnn_lstm_long_final_state, long_output))
    
print(f"\n--------\nFinal State (Standard-Long Context): \n")
for row in rnn_long_final_state:
    print(row) 
print("Standard Long Loss (Final): ",_mle.MSE_loss(rnn_long_final_state, long_output))