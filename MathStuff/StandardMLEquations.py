import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import math
import random
import MathStuff.MathEquations as MATH

class SMLE:
    
    def __init__(self):
        self._math = MATH.Math()
        pass
    
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
    
    
    