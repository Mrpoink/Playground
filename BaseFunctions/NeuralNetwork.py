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
        self.weight = self._generate_random_matrix(hidden_size, input_size)
        
        # self.a = _math.normalize(self._generate_random_matrix(hidden_size, hidden_size))
        
        self.bias = [[0] for _ in range(hidden_size)]
        
        self.hidden_weight = self._generate_random_matrix(hidden_size, hidden_size)
        
        self.output = None
        self.input = None
        self.state = None
        
    
    def _generate_random_matrix(self, rows : int, col : int):
        
        return [[random.random() for _ in range(col)] for _ in range(rows)]
        
    def get_weight(self):
        return self.weight
    
    def activate(self, input : list):
        self.input = input
        self.a = _math.dot_product(self.weight, input)
        self.state = _math.matrix_addition(self.a, self.bias)
        self.ouptut = self.state
        
        return self.state
    
    def activation_at_time(self, input : list, prev_state):
        
        z = self.rnn_z(self.weight, self.hidden_weight, self.bias, input, prev_state)
        self.state = _math.matrix_tanh(z)
         
        return  self.state
    
    def rnn_z(self, weight, hidden, bias, input, prev_state):
        
        w = weight
        h = hidden
        h_1 = prev_state
        b = bias
        x = input
        
        wx = _math.dot_product(w, x)
        uh = _math.dot_product(h, h_1)
        sum_val = _math.matrix_addition(wx, uh)
        return _math.matrix_addition(sum_val, b)
    
    def get_activation(self):
        return self.a
    
    def get_bias(self):
        return self.bias





class LSTM(Neuron):
    
    def __init__(self, input_size = 1, hidden_size = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Forget Gate
        self.F_w = self._generate_random_matrix(self.hidden_size, self.input_size)
        self.F_h = self._generate_random_matrix(self.hidden_size, self.hidden_size)
        self.F_b = [[0] for _ in range(hidden_size)]
        
        # Input Gate
        self.I_w = self._generate_random_matrix(self.hidden_size, self.input_size)
        self.I_h = self._generate_random_matrix(self.hidden_size, self.hidden_size)
        self.I_b = [[0] for _ in range(hidden_size)]
        
        # Candidate (Memory) Weights
        self.C_w = self._generate_random_matrix(self.hidden_size, self.input_size)
        self.C_h = self._generate_random_matrix(self.hidden_size, self.hidden_size)
        self.C_b = [[0] for _ in range(hidden_size)]
        
        # Output Weights
        self.O_w = self._generate_random_matrix(self.hidden_size, self.input_size)
        self.O_h = self._generate_random_matrix(self.hidden_size, self.hidden_size)
        self.O_b = [[0] for _ in range(hidden_size)]
        
    def activate_step(self, input_at_time, prev_state, prev_mem):
        
        ## Defining the actual variables named in the LSTM diagram and equations
        x = input_at_time
        h = prev_state
        c = prev_mem
        
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
        
        f, i, c_t, o = cache["f"], cache["i"], cache["c_t"], cache["o"]
        c_prev, c_tanh = cache["c_prev"], cache["c_tanh"]
        x, h_prev = cache["x"], cache["h_prev"]
        
        ## Calculate gate gradients
        
        ## Output
        grad_o = _math.hadamard_product(dh, c_tanh)
        d_o = _math.hadamard_product(grad_o, _math.sigmoid_der(o))
        
        ## States
        grad_c = _math.hadamard_product(dh, o)
        grad_c = _math.hadamard_product(grad_c, _math.matrix_tanh_derivative(c_tanh))
        dC = _math.matrix_addition(dc, grad_c)
        
        # Memory
        grad_ct = _math.hadamard_product(dC, i)
        d_ct = _math.hadamard_product(grad_c, _math.matrix_tanh_derivative(c_t))
        
        # Input
        grad_i = _math.hadamard_product(dC, c_t)
        d_i = _math.hadamard_product(grad_i, _math.sigmoid_der(i))
        
        # Forget
        grad_f = _math.hadamard_product(dC, c_prev)
        d_f = _math.hadamard_product(grad_f, _math.sigmoid_der(f))
        
        # Time
        x_T = _math.transpose(x)
        h_T = _math.transpose(h_prev)
        
        
        
        ## I don't really want to make this into it's own ath function unless 
        ## it comes in handy later
        
        def compute_grad(gate):
            d_w = _math.dot_product(gate, x_T)
            d_u = _math.dot_product(gate, h_T)
            d_b = gate
            return d_w, d_u, d_b
        
        def project(gate, hidden_weight):
            return _math.dot_product(_math.transpose(hidden_weight), gate)
        
        grads = {
            "F" : compute_grad(d_f),
            "I" : compute_grad(d_i),
            "C" : compute_grad(d_ct),
            "O" : compute_grad(d_o)
        }
        
        ## Time to actually propagate back over the course of 4 NEURONS
        ## Man, I need soemthing better to do with my time
        
        dh_prev = dh_prev = [[0 for _ in range(len(h_prev[0]))] for _ in range(len(h_prev))]
        dh_prev = _math.matrix_addition(dh_prev, project(d_f, self.F_h))
        dh_prev = _math.matrix_addition(dh_prev, project(d_i, self.I_h))
        dh_prev = _math.matrix_addition(dh_prev, project(d_ct, self.C_h))
        dh_prev = _math.matrix_addition(dh_prev, project(d_o, self.O_h))
        
        dC_prev = _math.hadamard_product(dC, f)
        
        return grads, dh_prev, dC_prev
        
        
        
        
        
    