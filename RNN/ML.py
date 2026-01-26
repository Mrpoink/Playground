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
        
        self.current_hidden_state = _mle.get_blank_matrix(1, hidden_layers, 0)
        self.current_memory = _mle.get_blank_matrix(1, hidden_layers, 0)
        
    def forward(self, input : list):
        history = []
        
        for time in input:
            
            new_state = self.neuron.activation_at_time(time, self.current_hidden_state)
            
            self.current_hidden_state = new_state
            
            history.append(new_state)
                
        return self.current_hidden_state, history
    
    def LSTM_forward(self, input : list):
        
        history = []
        
        for x in input:
            
            new_state, new_memory = self.LSTM_Neuron.activate_step(x, self.current_hidden_state, self.current_memory)
            
            self.current_hidden_state = new_state
            self.current_memory = new_memory
            
            history.append(new_state)

        return self.current_hidden_state, history
   
    def train_LSTM(self, train : list, test : list, epochs : int, learning_rate = 1e-5):
        best_loss = 5000000.0
        best_epoch = 0
        
        state_history = []
        
        for epoch in epochs:
            
            y_hat, _ = self.LSTM_forward(train)
            
            state_history.append(y_hat)
            
            error = _mle.cce_deriv(y_hat, test)
            
            
            
        pass
    
    def train_forward(self, train : list, test : list, epochs : int, learning_rate = 1e-5):
        
        for epoch in range(epochs):
        
            state, history = self.forward(train)
            
            pre_state = _mle.get_blank_matrix(1, 3, 0)
            history = [pre_state] + history
            
            error = _math.matrix_subtraction(state, test)
            
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
            print(f"Epoch {epoch} Error: {overall_error}\n")
    


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

rnn = RNN(input_size=1, hidden_layers=3)

input = [
    [[0.25]], 
    [[0.75]],
    [[0.65]]
    ]

output = [
    [0.75],
    [0.25],
    [0.25]
]

final_state, history = rnn.forward(input)

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
    
rnn = RNN(input_size=1, hidden_layers=3)

final_state, history = rnn.LSTM_forward(input)

print(f"\n--------\nFinal State: \n")
for row in final_state:
    print(row)

print(f"\n--------\nHistory: \n")
for item in history:
    print(item)
    
print("\n\n-----------\nTraining: \n")
rnn.train_forward(input, output, 9, 0.1)

