import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import random

_math = MATH.Math()

class Neuron:
    
    def __init__(self, input_size=3, hidden_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # self.a = _math.normalize(self._generate_random_matrix(hidden_size, hidden_size))
        
        self.weight = [[random.uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(hidden_size)]
        self.hidden_weight = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.bias = [[0.0] for _ in range(hidden_size)]
        
        self.output = None
        self.input = None
        self.state = None
        
        
    def get_weight(self):
        return self.weight
    
    def activate(self, input : list):
        self.input = input
        self.a = _math.dot_product(self.weight, input)
        self.state = _math.matrix_addition(self.a, self.bias)
        self.ouptut = self.state
        
        return self.state
    
    def activation_at_time(self, input : list, prev_state):
        _math.print_matrix(input, "activation input")
        z = self.rnn_z(self.weight, self.hidden_weight, self.bias, input, prev_state)
        self.state = _math.matrix_tanh(z)
         
        return  self.state
    
    def rnn_z(self, weight, hidden, bias, input, prev_state):
        
        #print(input)
       # Coerce to column vectors
        x = _math.as_col(input)
        h_1 = _math.as_col(prev_state)

        _math.print_matrix(weight, "weight")
        _math.print_matrix(hidden, "hidden")
        _math.print_matrix(bias, "bias")
        _math.print_matrix(x, "x for z")
        _math.print_matrix(h_1, "h_1")

        
        wx = _math.dot_product(weight, x)      # (H x I) · (I x 1) -> (H x 1)
        uh = _math.dot_product(hidden, h_1)    # (H x H) · (H x 1) -> (H x 1)
        _math.print_matrix(wx, "wx")
        _math.print_matrix(uh, "uh")
        z = _math.matrix_addition(_math.matrix_addition(wx, uh), bias)  # (H x 1)

        _math.print_matrix(z, "z")
        return z
    
    def get_activation(self):
        return self.a
    
    def get_bias(self):
        return self.bias





class LSTM(Neuron):
    
    def __init__(self, input_size = 1, hidden_size = 3):
        super().__init__(input_size, hidden_size)
        H, I = hidden_size, input_size
        
        # Forget Gate
        self.F_w = [[random.uniform(-0.01, 0.01) for _ in range(I)] for _ in range(H)]
        self.F_h = [[random.uniform(-0.01, 0.01) for _ in range(H)] for _ in range(H)]
        self.F_b = [[0.0] for _ in range(H)]
        
        # Input Gate
        self.I_w = [[random.uniform(-0.01, 0.01) for _ in range(I)] for _ in range(H)]
        self.I_h = [[random.uniform(-0.01, 0.01) for _ in range(H)] for _ in range(H)]
        self.I_b = [[0.0] for _ in range(H)]
        
        # Candidate (Memory) Weights
        self.C_w = [[random.uniform(-0.01, 0.01) for _ in range(I)] for _ in range(H)]
        self.C_h = [[random.uniform(-0.01, 0.01) for _ in range(H)] for _ in range(H)]
        self.C_b = [[0.0] for _ in range(H)]
        
        # Output Weights
        self.O_w = [[random.uniform(-0.01, 0.01) for _ in range(I)] for _ in range(H)]
        self.O_h = [[random.uniform(-0.01, 0.01) for _ in range(H)] for _ in range(H)]
        self.O_b = [[0.0] for _ in range(H)]
        
    def activate_step(self, input_at_time, prev_state, prev_mem):
        
        
        
        ## Defining the actual variables named in the LSTM diagram and equations
        x = input_at_time
        h = _math.reshape(prev_state)
        c = _math.reshape(prev_mem)
        
        _math.print_matrix(x, "x in step")
        _math.print_matrix(h, "h in step")
        _math.print_matrix(c, "c in step")
        
        ## Define the different gates at this current time
        f_t = _math.sigmoid(self.rnn_z(self.F_w, self.F_h, self.F_b, x, h))        
        i_t = _math.sigmoid(self.rnn_z(self.I_w, self.I_h, self.I_b, x, h))        
        c_t = _math.matrix_tanh(self.rnn_z(self.C_w, self.C_h, self.C_b, x, h))
        o_t = _math.sigmoid(self.rnn_z(self.O_w, self.O_h, self.O_b, x, h))
        
        ## Update them now  
        forget = _math.hadamard_product(f_t, c)
        input = _math.hadamard_product(i_t, c_t)
        c_t_1 = _math.matrix_addition(forget, input)
        
        ## To keep stability, we will copy the memory so tanh doesn't ruin everything
        c_temp = [row[:] for row in c_t_1]
        c_tanh = _math.matrix_tanh(c_temp)
        
        h_new = _math.hadamard_product(o_t, c_tanh)
        
        cache = {
            "f" : f_t,
            "i" : i_t,
            "c_t" : c_t,
            "c_prev" : c ,
            "c_new" : c_t_1,
            "c_tanh" : c_tanh,
            "x" : x,
            "h_prev" : h,
            "o" : o_t
        }
        
        return h_new, c_t_1, cache
    
    def BackPropagation_Step_RNN_LSTM(self, dh, dc, cache):
        # Unpack cache and coerce to column vectors
        # print(cache.keys())
        x, h_prev, c_prev, f, i, c_hat, o, c_t = cache["x"], cache["h_prev"], cache["c_prev"], cache["f"], \
            cache["i"], cache["c_new"], cache["o"], cache["c_t"]
        x = _math.as_col(x)
        h_prev = _math.as_col(h_prev)
        c_prev = _math.as_col(c_prev)
        f = _math.as_col(f)
        i = _math.as_col(i)
        c_hat = _math.as_col(c_hat)
        o = _math.as_col(o)
        c_t = _math.as_col(c_t)
        dh = _math.as_col(dh)
        dc = _math.as_col(dc)

        # dC total
        dC = _math.matrix_addition(dc, _math.hadamard_product(_math.hadamard_product(dh, o), _math.matrix_tanh_derivative(c_t)))

        # Gate derivatives
        d_o = _math.hadamard_product(_math.hadamard_product(dh, _math.matrix_tanh(c_t)), _math.sigmoid_der(o))
        d_f = _math.hadamard_product(_math.hadamard_product(dC, c_prev), _math.sigmoid_der(f))
        d_i = _math.hadamard_product(_math.hadamard_product(dC, c_hat), _math.sigmoid_der(i))
        d_c_hat = _math.hadamard_product(_math.hadamard_product(dC, i), _math.matrix_tanh_derivative(c_hat))

        # Gradients
        x_T = _math.transpose(x)          # 1 x I
        h_T = _math.transpose(h_prev)     # 1 x H

        dF_w = _math.dot_product(d_f, x_T)       # H x I
        dF_h = _math.dot_product(d_f, h_T)       # H x H
        dF_b = d_f                                # H x 1

        dI_w = _math.dot_product(d_i, x_T)
        dI_h = _math.dot_product(d_i, h_T)
        dI_b = d_i

        dC_w = _math.dot_product(d_c_hat, x_T)
        dC_h = _math.dot_product(d_c_hat, h_T)
        dC_b = d_c_hat

        dO_w = _math.dot_product(d_o, x_T)
        dO_h = _math.dot_product(d_o, h_T)
        dO_b = d_o

        # Backprop to previous hidden and cell
        dh_prev = _math.matrix_addition(
            _math.matrix_addition(_math.dot_product(_math.transpose(self.F_h), d_f),
                                  _math.dot_product(_math.transpose(self.I_h), d_i)),
            _math.matrix_addition(_math.dot_product(_math.transpose(self.C_h), d_c_hat),
                                  _math.dot_product(_math.transpose(self.O_h), d_o))
        )
        dC_prev = _math.hadamard_product(dC, f)

        grads = {
            "F": (dF_w, dF_h, dF_b),
            "I": (dI_w, dI_h, dI_b),
            "C": (dC_w, dC_h, dC_b),
            "O": (dO_w, dO_h, dO_b),
        }
        return grads, dh_prev, dC_prev
        
        
        
        
        
    