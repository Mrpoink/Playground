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
from tqdm import tqdm

_math = MATH.Math()
_mle = MLE.SMLE()

tokenizer = Tokenizer.Embedding()

class RNN:
    
    def __init__(self, hidden_layers : int, input_size : int, embedding_dim = 16):
        self.neuron = NN.Neuron(input_size=embedding_dim, hidden_size=hidden_layers)
        self.LSTM_Neuron = NN.LSTM(input_size=embedding_dim, hidden_size=hidden_layers)
        
        self.current_hidden_state = _mle.get_blank_matrix(hidden_layers, 1, 0.0)
        self.current_memory = _mle.get_blank_matrix(hidden_layers, 1, 0.0)
        
        self.tokenizer = tokenizer
        self.tokenizer.embedding_dim = embedding_dim
        
        self.decoder = [[[0.0] for _ in range(hidden_layers)] for _ in range(input_size)]
        self.output = []
        
        self.Y_w = [[random.uniform(-1.0, 1.0)for _ in range(hidden_layers)] for _ in range(input_size)]
        self.Y_b = [[0.0] for _ in range(input_size)]
        
    def forward(self, input : list):
        token_ids = self.tokenizer.encode(input)
        # print(f"history dimensions: {len(token_ids)}, 1")

        sequence = []
        for tid in token_ids:
            emb_row = self.tokenizer.lookup(tid)           # [D]
            x_t = _math.as_col(emb_row)                    # D x 1
            sequence.append(x_t)

        # print(f"seq dimensions: {len(sequence)}, {len(sequence[0])}")
        # print(sequence[0])

        state_history = []
        for time in sequence:
            # print(f"time dimensions: {len(time), len(time[0])}")
            # print(f"\nhidden state dimensions: {len(self.current_hidden_state)}, {len(self.current_hidden_state[0])}")
            new_state = self.neuron.activation_at_time(time, self.current_hidden_state)
            # print(f"New_state dimensions: {len(new_state)}, {len(new_state[0])}")
            self.current_hidden_state = new_state
            state_history.append(new_state)
        return self.current_hidden_state, state_history
    
    def LSTM_forward(self, input : list):
        train_embedding = self.tokenizer.encode(input)
        sequence = []
        for token_id in train_embedding:
            emb_row = self.tokenizer.lookup(token_id)     # [D]
            x_t = _math.as_col(emb_row)                   # D x 1
            sequence.append(x_t)
        # print(f"seq dimensions: {len(sequence)}, {len(sequence[0])}")
        # print(sequence[0])

        state_history = []
        caches = []
        for time in sequence:
            new_state, new_memory, cache = self.LSTM_Neuron.activate_step(time, self.current_hidden_state, self.current_memory)
            self.current_hidden_state = new_state
            self.current_memory = new_memory
            state_history.append(new_state)
            caches.append(cache)

        return self.current_hidden_state, state_history, caches
   
    def train_LSTM(self, train : list, test : list, epochs : int, learning_rate = 1e-5):
        V = self.tokenizer.vocab_size
        H = self.LSTM_Neuron.hidden_size

        test_ids = self.tokenizer.encode(test)
        if not test_ids:
            raise ValueError("Empty test_ids; tokenizer may not contain test chars")

        # Re-sync decoder to current V,H if mismatched
        if len(self.Y_w) != V or len(self.Y_w[0]) != H:
            # print("Decoder mismatch; resetting Y_w/Y_b")
            self.Y_w = [[random.uniform(-0.1, 0.1) for _ in range(H)] for _ in range(V)]
            self.Y_b = [[0.0] for _ in range(V)]
            # print("Y_b, Y_w: ", len(self.Y_b), len(self.Y_b[0]), len(self.Y_w), len(self.Y_w[0]))

        # Determine zero/one-based tokenizer ids safely
        max_id_map = max(self.tokenizer.char_to_int.values()) if self.tokenizer.char_to_int else 0
        is_one_based = (max_id_map == V)

        for epoch in range(epochs):
            # Reset states
            self.current_hidden_state = _mle.get_blank_matrix(H, 1, 0.0)
            self.current_memory = _mle.get_blank_matrix(H, 1, 0.0)

            # Forward through LSTM on train text
            state, history, caches = self.LSTM_forward(train)

            # Build one-hot target (V x 1) for last test token
            target_id_raw = test_ids[-1]
            idx = target_id_raw - 1 if is_one_based else target_id_raw

            # print("Vocab size: ", V)
            # print("Hidden size: ", H)
            # print("Target raw/id: ", target_id_raw, idx)

            if idx < 0 or idx >= V:
                raise ValueError(f"Token index {idx} out of range for vocab size {V}")

            # Decoder forward: logits V x 1
            # print("State: ", len(state), len(state[0]))
            # print("Y_b, Y_w: ", len(self.Y_b), len(self.Y_b[0]), len(self.Y_w), len(self.Y_w[0]))

            logits = _math.matrix_addition(_math.dot_product(self.Y_w, state), self.Y_b)
            # print("Logits: ", len(logits), len(logits[0]))

            probabilities = _mle.softmax(logits)
            print("Probabilities: ", len(probabilities), len(probabilities[0]))
            self.output = probabilities
            print("Probabilities: ", probabilities)

            # Cross-entropy loss and dlogits (V x 1)
            loss = _mle.vector_CCEL(probabilities, idx)
            # print("Loss: ", loss)
            dlogits = _mle.vector_cce_der(probabilities, idx)
            
            # Decoder grads
            dY_w = _math.dot_product(dlogits, _math.transpose(state))          # V x H
            dY_b = dlogits                                                      # V x 1
            dh_n1 = _math.dot_product(_math.transpose(self.Y_w), dlogits)       # H x 1
            dC_n1 = _mle.get_blank_matrix(H, 1, 0.0)                            # H x 1

            # Clip + update decoder
            dY_w = _mle.gradient_clip(dY_w, 1.0)
            dY_b = _mle.gradient_clip(dY_b, 1.0)
            self.Y_w = _math.matrix_subtraction(self.Y_w, _math.scalar_multiply(dY_w, learning_rate))
            self.Y_b = _math.matrix_subtraction(self.Y_b, _math.scalar_multiply(dY_b, learning_rate))

            # Backprop through time
            T = len(caches)
            for t in reversed(range(T)):
                cache = caches[t]
                # print("\nHidden dimensions: ", len(dh_n1), len(dh_n1[0]))
                # print("Memory dimensions: ", len(dC_n1), len(dC_n1[0]))

                grads, dh_prev, dC_prev = self.LSTM_Neuron.BackPropagation_Step_RNN_LSTM(dh_n1, dC_n1, cache)

                def update(W, U, b, g_tuple):
                    dW, dU, db = g_tuple
                    dW = _mle.gradient_clip(dW, 1.0)
                    dU = _mle.gradient_clip(dU, 1.0)
                    db = _mle.gradient_clip(db, 1.0)
                    W_new = _math.matrix_subtraction(W, _math.scalar_multiply(dW, learning_rate))
                    U_new = _math.matrix_subtraction(U, _math.scalar_multiply(dU, learning_rate))
                    b_new = _math.matrix_subtraction(b, _math.scalar_multiply(db, learning_rate))
                    return W_new, U_new, b_new

                self.LSTM_Neuron.F_w, self.LSTM_Neuron.F_h, self.LSTM_Neuron.F_b = update(self.LSTM_Neuron.F_w, self.LSTM_Neuron.F_h, self.LSTM_Neuron.F_b, grads["F"])
                self.LSTM_Neuron.I_w, self.LSTM_Neuron.I_h, self.LSTM_Neuron.I_b = update(self.LSTM_Neuron.I_w, self.LSTM_Neuron.I_h, self.LSTM_Neuron.I_b, grads["I"])
                self.LSTM_Neuron.C_w, self.LSTM_Neuron.C_h, self.LSTM_Neuron.C_b = update(self.LSTM_Neuron.C_w, self.LSTM_Neuron.C_h, self.LSTM_Neuron.C_b, grads["C"])
                self.LSTM_Neuron.O_w, self.LSTM_Neuron.O_h, self.LSTM_Neuron.O_b = update(self.LSTM_Neuron.O_w, self.LSTM_Neuron.O_h, self.LSTM_Neuron.O_b, grads["O"])

                dh_n1 = dh_prev
                dC_n1 = dC_prev

            print(f"Epoch {epoch} LSTM Loss: {loss}")
                
    
    def train_forward(self, train : list, test : list, epochs : int, learning_rate = 1e-5):
        train_ids = self.tokenizer.encode(train)
        test_ids = self.tokenizer.encode(test)
        # print(f"\nDimensions of test_embedding: {len(test_ids)}, 1")
        # print(f"\nDimensions of train_embedding: {len(train_ids)}, 1")

        for epoch in range(epochs):
            state, history = self.forward(train)

            target_id = test_ids[-1]
            target_vec = _math.as_col(self.tokenizer.lookup(target_id))  # H x 1

            # print("\nNew state dimensions: ", len(state), len(state[0]))
            # print("Dimensions of target: ", len(target_vec), len(target_vec[0]))

            error = _math.matrix_subtraction(state, target_vec)          # H x 1
            # print("\n----------------\nError dims: ", len(error), len(error[0]))

            grad_w = _mle.get_blank_matrix(len(self.neuron.weight), len(self.neuron.weight[0]), 0)        # H x I
            grad_h = _mle.get_blank_matrix(len(self.neuron.hidden_weight), len(self.neuron.hidden_weight[0]), 0)  # H x H
            grad_b = _mle.get_blank_matrix(len(self.neuron.bias), len(self.neuron.bias[0]), 0)            # H x 1

            T = len(train_ids)
            for t in reversed(range(T - 1)):
                x_row = self.tokenizer.lookup(train_ids[t])              # [I]
                input_at_time = _math.as_col(x_row)                      # I x 1
                output_at_time = history[t+1]                            # H x 1
                h_t_1 = history[t]                                       # H x 1

                # print("\nBackprop (Standard) inputs: ")
                # print("error: ", len(error), len(error[0]))
                # print("input_at_time: ", len(input_at_time), len(input_at_time[0]))
                # print("h_t_1: ", len(h_t_1), len(h_t_1[0]))
                # print("output_at_time: ", len(output_at_time), len(output_at_time[0]))
                # print("hidden_weight: ", len(self.neuron.hidden_weight), len(self.neuron.hidden_weight[0]))

                gw, gh, gb, dh_t_1 = _mle.BackPropagation_Step_RNN(
                    error, input_at_time, h_t_1, output_at_time, self.neuron.hidden_weight
                )
                # print("gb: ", len(gb), len(gb[0]), "\ngrad_b: ", len(grad_b), len(grad_b[0]))
                
                if len(grad_b) != len(gb):
                    grad_b = _math.transpose(grad_b)

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
            
            
            
            overall_error = _mle.MSE_loss_matrix(state, target_vec)
            print(f"Epoch {epoch} Error: {overall_error}")
    


## Here I will implement the tokenizer to create mock input/output data:

training_data = "I want you to run this to show the effects of explosive gradients"
test_data = "We are achieving this result due to the constant multiplication"
tokenizer.train("## I want you to run this to show the effects of explosive gradients \
                ## We are achieving this result due to the constant multiplication \
                ## And running off the previous weights. \
                ## Take a look at the tanh equation and usage in the MATH class \
                ## We are adding and subtracting numbers yes; however, \
                ## As we multiply the positive weight with the positive inputs \
                ## We get a positive output, that is then added to the previous step \
                ## (See NeuralNetwork.activate_at_time) \
                ## And we continue to do this through the chain of neurons \
                ## Hence, an explosive gradient")




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

rnn = RNN(input_size=tokenizer.vocab_size, hidden_layers=tokenizer.embedding_dim)


final_state, history = rnn.forward(training_data)

# print(f"\n--------\nFinal State: \n")
# for row in final_state:
#     print(row)

# print(f"\n--------\nHistory: \n")
# for item in history:
#     print(item)


## This is the logic for the Long Short Term Memory as designed
## The design is drawn in my notes regarding Attention is All You Need
## I think you can guess what my next implementation will be (after propogation)


print(f"\n\n----------\nLSTM Implementation: \n")
    
rnn = RNN(input_size=tokenizer.vocab_size, hidden_layers=tokenizer.embedding_dim)

final_state, history, caches = rnn.LSTM_forward(training_data)

print(f"\n--------\nFinal State: \n")
for row in final_state:
    print(row)

print(f"\n--------\nHistory: \n")
for item in history:
    print(item)
    
print("\n\n-----------\nTraining (Standard Implementation): \n")
rnn.train_forward(training_data, test_data, 30, 0.1)

print("\n\n-----------\nTraining (LSTM Implementation): \n")

rnn.train_LSTM(training_data, test_data, 300, 1e-2)
# print("\n------------\nTest: \n")

# print("\n------------\nRaw State: \n")
# print(rnn.output)
# print(f"Actual Output: {test_data}")