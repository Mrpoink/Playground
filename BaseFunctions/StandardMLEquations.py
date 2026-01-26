import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import math
import random
import BaseFunctions.MathEquations as MATH
import BaseFunctions.NeuralNetwork as NN

class SMLE:
    
    def __init__(self):
        self._math = MATH.Math()
        pass
    
    def get_blank_matrix(self, N : int, M : int, number : int = None):
        matrix = [[_ for _ in range(N)] for _ in range(M)]
      
        if number is not None:
            for i in range(M):
                for j in range(N):
                    matrix[i][j] = number
        else:
            for i in range(M):
                for j in range(N):
                    matrix[i][j] = 0
        return matrix
    
    def LSE(self, input : list):
        ## This is used to calculate log(softmax(z))
        ## We will be using a variant of this algorithm to prevent overflow
        ## xi subtracted by the maximum value of x will not change the relative properties
        ## and keeps numbers small (relatively)
        ## This algorithm is best for categorical loss applications
        
        ## M for max
        M = max(input)
        
        ## find sum of the e's in the row
        exp_sum = sum([math.exp(value - M) for value in input])
        
        return M + math.log(exp_sum)
    
    def log_softmax(self, z : list):
        ## This will compute the log_softmax efficiently if the input is a matrix
        
        matrix = []
        
        for row in z:
            
            lse = self.LSE(row)
            
            log_probs = [value - lse for value in row]
            matrix.append(log_probs)
            
            
        return matrix
            
    def project(self, gate, hidden_weight):
        return self._math.dot_product(self._math.transpose(hidden_weight), gate)
        
    
    def softmax(self, z : list):
        r'''
        see my example under "Kulkarni's Equations" and "Backpropogation"
        input: z as an array with NxM dimensions
        output: an array with softmax calculated
        '''
        matrix = []
        
        ## Keep in mind that a matrix looks like this:
        ## [[1,2],[2,1]]
        ## We compute row by row
        for i in range(len(z)):
            
            M = max(z[i])
            ## Stability shift so that we get the most accurate softmax
            ## Think about the CNN paper, they wanted to reduce the variance
            
            numerators = [math.exp(value - M) for value in z[i]]
            denominator = sum(numerators)
            matrix.append([numerator / denominator for numerator in numerators])
     
        return matrix
    
    def CCEL(self, y_hat, y : list):
        
        softmax_matrix = self.log_softmax(y_hat)
        
        N = len(softmax_matrix)
        total_loss = 0
        
        for i in range(len(softmax_matrix)):
            loss = 0
            for j in range(len(y[i])):
                
                loss += softmax_matrix[i][j] * y[i][j]
                
            total_loss += loss
                
        return -1 * (total_loss / N)
    
    def cce_deriv(self, pred_matrix : list, true_matrix : list):
        
        N = len(pred_matrix)
        grad_matrix = []
        
        for i in range(N):
            gradient_row = []
            for j in range(len(true_matrix[i])):
                
                derivative = -1 * (true_matrix[i][j] / (pred_matrix[i][j] + 1e-9))
                gradient_row.append(derivative)
                
            grad_matrix.append(gradient_row)
        
        return grad_matrix
    
    def MSE_loss(self, y_hat : list, y : list):
        
        total_loss = 0
        
        for i in range(len(y_hat)):
            for j in range(len(y[0])):
                
                total_loss += 0.5 * ((y_hat[i][j] - y[i][j]) * (y_hat[i][j] - y[i][j]))
                
        return total_loss
        
        
    def MSE_loss_der(self, y_hat : list, y : list):
        
        return sum(y_hat[i][j] - y[i][j] for i, j in (range(len(y_hat)), range(len(y))))     
        
    
    def sigmoid_derivative(self, matrix : list):
        
        blank = self.get_blank_matrix(len(matrix), len(matrix[0]), 1)
        sub = self._math.matrix_subtraction(blank, matrix)
        
        return self._math.hadamard_product(matrix, sub)
    
    
    
    def BackPropagation_Step(self, output_node : NN.Neuron, loss : list, learning_rate : float):
        
        sig_der = self.sigmoid_derivative(output_node.output)
        
        delta = self._math.hadamard_product(loss, sig_der)
        
        weight_t = self._math.transpose(output_node.weight)
        next_error = self._math.dot_product(weight_t, delta)
        
        input_t = self._math.transpose(output_node.input)
        weight_gradient = self._math.dot_product(delta, input_t)
        
        for i in range(len(output_node.weight)):
            for j in range(len(output_node.weight[0])):
                change = learning_rate * weight_gradient[i][j]
                output_node.weight[i][j] = output_node.weight[i][j] - change
                
        for i in range(len(output_node.bias)):
            for j in range(len(output_node.bias[0])):
                output_node.bias[i][j] = output_node.bias[i][j] - (learning_rate * delta[i][j])
                
        return next_error
    
    def BackPropagation_Step_RNN(self, dh, x_t, h_t_1, h_t, hw):
        
        dtanh = self._math.matrix_tanh_derivative(h_t)
        
        dz = self._math.hadamard_product(dh, dtanh)
        
        T_input = self._math.transpose(x_t)
        T_h_t_1 = self._math.transpose(h_t_1)
        
        grad_w = self._math.dot_product(dz, T_input)
        
        grad_h = self._math.dot_product(dz, T_h_t_1)
        
        grad_b = dz
        
        w_hh_T = self._math.transpose(hw)
        dh_t_1 = self._math.dot_product(w_hh_T, dz)
        
        return grad_w, grad_h, grad_b, dh_t_1
    
        
        
        
    
    
    