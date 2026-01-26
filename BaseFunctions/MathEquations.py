import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import math
import random
class Math:
    
    def __init__(self):
        pass
    
    def normalize(self, matrix : list):
        r'''
        Normalizes the rows in the matrix so the sum equals 1
        We may be able to use the detriment to normalize
        The entire matrix rather than a row
        '''
        
        normalized_matrix = []
        
        for row in matrix:
            row_sum = sum(row)
            
            normalized_matrix.append([value / row_sum for value in row])
            
        return normalized_matrix 
        
    
    def sigmoid(self, input : list):
        
        ## Element-wise, we parse through the existing matrix and find the sigmoid of that value
        
        return [[1.0 / (1.0 + math.exp(-value)) for value in row] for row in input]
    
    def sigmoid_der(self, input : list):
        
        return [[val * (1.0 - val) for val in row] for row in input]
    
    
    def dot_product(self, matrix_1 : list, matrix_2 : list):
        r'''
        IMPORTANT: These two matrices MUST be the same size
        parameters : two matrices of equal size
        returns : a single matrix
        '''
        
        row_1 = len(matrix_1)
        col_1 = len(matrix_1[0])
        
        
        ## Second matrix
        
        row_2 = len(matrix_2)
        col_2 = len(matrix_2[0])
        
        if col_1 != row_2:
            print(f"Matrix 1: {row_1}x{col_1}\nMatrix 2: {row_2}x{col_2}")
            raise ValueError("Incompatible dimensions")
        
        final_matrix = [[0 for _ in range(col_2)] for _ in range(row_1)]
        
        for i in range(row_1):
            for k in range(col_2):
                point = 0
                
                for j in range(col_1):
                    
                    point += matrix_1[i][j] * matrix_2[j][k]
                    
                final_matrix[i][k] = point
            
        return final_matrix
    
    
    def matrix_addition(self, matrix_1 : list, matrix_2 : list):
        
        N_1, M_1 = len(matrix_1), len(matrix_1[0])
        N_2, M_2 = len(matrix_2), len(matrix_2[0])
        
        if (N_1 != N_2) or (M_1 != M_2):
            raise ValueError("For some reason, you can't add these, check the dimensions")
        
        final_matrix = [[0 for _ in range(M_2)] for _ in range(N_1)]
        
        for i in range(N_1):
            for j in range(M_1):
                
                final_matrix[i][j] = matrix_1[i][j] + matrix_2[i][j]
                
        return final_matrix
    
    def matrix_subtraction(self, matrix_1 : list, matrix_2 : list):
        
        N_1, M_1 = len(matrix_1), len(matrix_1[0])
        N_2, M_2 = len(matrix_2), len(matrix_2[0])
        
        if (N_1 != N_2) or (M_1 != M_2):
            raise ValueError("For some reason, you can't add these, check the dimensions")
        
        final_matrix = [[0 for _ in range(M_2)] for _ in range(N_1)]
        
        for i in range(N_1):
            for j in range(M_1):
                
                final_matrix[i][j] = matrix_1[i][j] - matrix_2[i][j]
                
        return final_matrix
    
    def find_determinate(self, matrix : list):
    
        n = len(matrix)
        
        if n > 2:
            while n > 2:
                
                rows = len(matrix)
                cols = len(matrix[0])
                
                col_d = random.random(n)
                row_d = random.random(n)
                
                d_point_r = matrix[row_d]
                d_point_c = matrix[col_d]
                
                matrix.remove(d_point_r)
                
                for i in range(rows):
                    i.remove(d_point_c)
                
        
        determinate = (matrix[0][cols] * matrix[rows][0]) - (matrix[0][0] * matrix[rows][cols])
        
        return determinate
    
    def hadamard_product(self, matrix_1 : list, matrix_2 : list):
        
        row_1 = len(matrix_1)
        col_1 = len(matrix_1[0])
        row_2 = len(matrix_2)
        col_2 = len(matrix_2[0])
        
        if (row_1 != row_2) or (col_1 != col_2):
            raise ValueError("Cannot multiply multiple matrices")
        
        product = [[_ for _ in range(col_1)] for _ in range(row_1)]
        
        for i in range(row_1):
            for j in range(col_1):
                product[i][j] = matrix_1[i][j] * matrix_2[i][j]
                
        return product
    
    
    def transpose(self, matrix : list):
        
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    def scalar_multiply(self, input : list, scalar):
        
        for i in range(len(input)):
            for j in range(len(input[0])):
                
                input[i][j] = input[i][j] * scalar   
                
        return input     
    
    
    def tanh(self, number):
        
        return (math.exp(number) - math.exp(-number)) / (math.exp(number) + math.exp(-number))
  
    def matrix_tanh(self, matrix : list):
        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                
                matrix[i][j] = self.tanh(matrix[i][j])
                
        return matrix
    
    def matrix_subtraction_scalar(self, scalar : float, input : list):
        
        rows = len(input)
        cols = len(input[0])
        
        matrix = [[0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = scalar - input[i][j]
                
        return matrix
    
    def matrix_tanh_derivative(self, input : list):
        
        squared = self.hadamard_product(input, input)
        
        return self.matrix_subtraction_scalar(1.0, squared)
        
                
        
    # def find_inverse(self, matrix : list):
        
    #     rows = len(matrix)
    #     cols = len(matrix[0])
        
    #     if rows != cols:
    #         raise ValueError("Matrix must be square to find inverse")
        
    #     determinate = self.find_determinate(matrix)
        
    #     if determinate < 0:
    #         raise ValueError("Determinate is too low, cannot continue")
        
    #     for i in rows:
    #         for j in cols:
    #             matrix[i][j] = matrix[j][i]
                
        
        
        
                                                      
        
    
        
        
        
        

# #print(Math.softmax([[1.5, 0.5, 0.5], [0.5, 1.5, 0.5]]))
# #dot product test:
# matrix_1 = [[2, 1], [4, 3]]
# matrix_2 = [[5, 0], [-1, 2]]

# print(Math.matrix_addition(matrix_1, matrix_2))       
        

                    