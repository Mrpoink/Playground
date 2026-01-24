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