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
            row_max = max(row)
            
            normalized_matrix.append([value / row_max for value in row])
            
        return normalized_matrix 
    
    def std_dev(self, input : list, sample = False):
        
        total = 0
        n = len(input)
        mean = sum(input) / n
        
        for point in input:
            total += (point - mean) ** 2
        
        if sample:
            return math.sqrt(total / n)
        else:
            return math.sqrt(total / (n-1))
        
        
        
        for item in input:
            total += item
            
        
        
        
    
    def print_matrix(self, input, name):
        
        H = len(input)
        I = len(input[0])
        
        print(f"{name} Size (H x I): {H}, {I}")
        for i in range(H):
            print(f"{i}.")
            for j in range(I):
                print(f"{input[i][j]}")
    
    def vector_norm(self, v):
        return math.sqrt(sum(x**2 for x in v))
    
    def multiply_with_broadcasting(self, matrix, vector):
        rows = len(matrix)
        cols = len(matrix[0])
        v_rows = len(vector)
        
        if rows != v_rows:
            raise ValueError("Dimensions mismatch for broadcasting.")

        # Create a 16x16 result
        new_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                # vector[i][0] applies the same vector element to the entire row
                new_matrix[i][j] = matrix[i][j] * vector[i][0]
                
        return new_matrix
    
    def qr_decomposition_raw(self, A):
        n = len(A)
        m = len(A[0])
        
        # Transpose to work with columns easily
        columns = self.transpose(A)
        Q_cols = []
        R = [[0.0 for _ in range(m)] for _ in range(m)]

        for j in range(m):
            v = list(columns[j])
            for i in range(j):
                # Calculate projection
                R[i][j] = self.dot_product_vectors(Q_cols[i], columns[j])
                # Subtract projection to make v orthogonal
                v = [v_k - R[i][j] * Q_cols[i][k] for k, v_k in enumerate(v)]
            
            # Normalize the resulting vector
            norm_v = self.vector_norm(v) # Pass the whole vector
            R[j][j] = norm_v
            
            if abs(norm_v) > 1e-10:
                Q_cols.append([v_k / norm_v for v_k in v])
            else:
                Q_cols.append([0.0 for _ in v])
        
        Q = self.transpose(Q_cols)
        return Q, R
    def dot_product_vectors(self, v1, v2):
        """Returns a single float: sum of v1[i] * v2[i]"""
        return sum(x * y for x, y in zip(v1, v2))

    def matrix_multiply(self, A, B): # Added self
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def get_eigenvalues_raw(self, A, iterations=50):
        Ak = [row[:] for row in A]
        for _ in range(iterations):
            Q, R = self.qr_decomposition_raw(Ak)
            # In the QR algorithm, Ak+1 = R * Q
            Ak = self.matrix_multiply(R, Q)
        
        return [Ak[i][i] for i in range(len(Ak))]
    
    def solve_eigenvector(A, eigenvalue):
        n = len(A)
        # 1. Create (A - lambda*I)
        M = [row[:] for row in A]
        for i in range(n):
            M[i][i] -= eigenvalue

        # 2. Forward Elimination (Transform to Upper Triangular)
        for i in range(n):
            # Pivot selection
            pivot = M[i][i]
            if abs(pivot) < 1e-10: # Handle near-zero pivots
                pivot = 1e-10
                
            for j in range(i + 1, n):
                factor = M[j][i] / pivot
                for k in range(i, n):
                    M[j][k] -= factor * M[i][k]

        # 3. Back-substitution 
        # Since (A-lambdaI) is singular, we set the last element to 1 
        # and solve for the others relative to it.
        v = [0 for _ in range(n)]
        v[n-1] = 1.0 
        
        for i in range(n-2, -1, -1):
            sum_val = sum(M[i][j] * v[j] for j in range(i + 1, n))
            if abs(M[i][i]) > 1e-10:
                v[i] = -sum_val / M[i][i]
            else:
                v[i] = 0

        # 4. Normalize the vector (so its length is 1)
        mag = math.sqrt(sum(x**2 for x in v))
        return [x / mag for x in v]
        
    def shape(self, matrix : list):
        return (len(matrix), len(matrix[0]))
    
    def reshape(self, matrix : list, N = 1, M = 1):
        """Only does N columns at the moment, will update for rows

        Args:
            matrix (list): matrix to be reshaped
            N (int, optional): columns. Defaults to 1.
            M (int, optional): rows. Defaults to 1.

        Returns:
            _type_: matrix
        """
        
        try:
            if len(matrix) == N:
                return self.transpose(matrix)
            if all(len(r) == N for r in matrix):
                return matrix
            
            return matrix
        except Exception as e:
            print("Error in reshaping\n", e)
    
    def generate_random_matrix(self, rows : int, col : int):
        
        return [[random.uniform(-0.01, 0.01) for _ in range(col)] for _ in range(rows)]
    
    def sigmoid_scalar(self, value):
        ## This function is designed for the sigmoid function below
        ## We are trying to avoid overflow errors
        ## So we go element wise and see if it's possible to do our
        ## Computations, and if not then we just use default values
        ## This can cause a bit of bias towards some vectors over others
        ## However, since we are dealing with values between -1.0 and 1.0
        ## This is negligable and works a lot like the stuff in the CNN paper
        ## I can't remember the name of it at this moment
        
        
        ## TODO: define the thing in the CNN paper that relates to this
        
        try:
            if value >= 0:
                z = math.exp(-value)
                return 1.0 / (1.0 + z)
            else:
                z = math.exp(value)
                return z / (1.0 + z)
        except OverflowError:
            return 1.0 if value > 0 else 0.0
        
    
    def sigmoid(self, input : list):
        
        ## Element-wise, we parse through the existing matrix and find the sigmoid of that value
        
        return [[self.sigmoid_scalar(value) for value in row] for row in input]
    
    def sigmoid_der(self, input : list):
        
        return [[val * (1.0 - val) for val in row] for row in input]
    
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
    
    
    def matrix_addition(self, matrix_1 : list, matrix_2 : list):
        
        N_1, M_1 = len(matrix_1), len(matrix_1[0])
        N_2, M_2 = len(matrix_2), len(matrix_2[0])
        
        if (N_1 != N_2) or (M_1 != M_2):
            self.print_matrix([[N_1, M_1]], "Addition lhs dims")
            self.print_matrix([[N_2, M_2]], "Addition rhs dims")
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
            self.print_matrix([[N_1, M_1]], "Subtraction lhs dims")
            self.print_matrix([[N_2, M_2]], "Subtraction rhs dims")
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
    
    def as_col(self, mat):
        # Convert 1 x N or flat list to N x 1; leave matrices as-is
        if isinstance(mat, list):
            if not mat:
                return [[]]
            if isinstance(mat[0], list):
                # 1 x N row -> N x 1 col
                if len(mat) == 1:
                    return self.transpose(mat)
                # Already N x 1
                if all(len(r) == 1 for r in mat):
                    return mat
                return mat
            # Flat list -> N x 1
            return [[v] for v in mat]
        # Scalar -> 1 x 1
        return [[mat]]
    
    def hadamard_product(self, A, B):
        # Elementwise product; coerce 1xN vs Nx1
        if not A or not B or not isinstance(A[0], list) or not isinstance(B[0], list):
            raise ValueError("Inputs must be 2D lists")
        rA, cA = len(A), len(A[0])
        rB, cB = len(B), len(B[0])
        # Allow (r x 1) * (1 x r)
        if rA == cB and cA == 1 and rB == 1:
            B = self.transpose(B)
            rB, cB = len(B), len(B[0])
        if rB == cA and cB == 1 and rA == 1:
            A = self.transpose(A)
            rA, cA = len(A), len(A[0])
        if rA != rB or cA != cB:
            raise ValueError("Shapes must match for hadamard_product")
        return [[A[i][j] * B[i][j] for j in range(cA)] for i in range(rA)]
    
    
    def get_matrix_means(self, input : list):
        
        rows, cols = len(input), len(input[0])
        means = []
        
        for j in range(cols):
            col_sum = sum(input[i][j] for i in range(rows))
        return means
    
    
    def PCA(self, input, N = None, M = None):
        
        standardized_matrix = self.standardization(input)
        
        #print("Standardized dimension : ", len(standardized_matrix), len(standardized_matrix[0]))
        
        cov = self.covariance(standardized_matrix)
        
        #print(f"dimension of conv: ", len(cov), len(cov[0]))
        
     
        eigen_value, principle_component = self.qr_decomposition_raw(cov)
        
        return eigen_value, principle_component
        
    def _unwrap_scalar(self, x):
        # If element is a single-item list, unwrap to scalar
        if isinstance(x, list):
            if len(x) == 1 and not isinstance(x[0], list):
                return x[0]
        return x

    def dot_product(self, matrix_1, matrix_2):
        # (A: m x n) · (B: n x p) => (m x p)
        #print(f"dimensions from dot product: {len(matrix_1)}, {len(matrix_1[0])}    {len(matrix_2)}, {len(matrix_2[0])}")
        if not matrix_1 or not matrix_2 or not isinstance(matrix_1[0], list) or not isinstance(matrix_2[0], list):
            raise ValueError("Inputs must be 2D lists")
        n1 = len(matrix_1[0])
        m = len(matrix_1)
        n2 = len(matrix_2)
        p = len(matrix_2[0])
        if n1 != n2:
            raise ValueError("Incompatible dimensions")
        result = [[0.0 for _ in range(p)] for _ in range(m)]
        for i in range(m):
            for k in range(p):
                s = 0.0
                for j in range(n1):
                    a = self._unwrap_scalar(matrix_1[i][j])
                    b = self._unwrap_scalar(matrix_2[j][k])
                    s += a * b
                result[i][k] = s
        return result
        
        
    def covariance(self, centered_matrix):
        n = len(centered_matrix)
        cols = len(centered_matrix[0])
        # Resulting matrix is cols x cols (16x16)
        cov = [[0.0 for _ in range(cols)] for _ in range(cols)]
        for i in range(cols):
            for j in range(cols):
                # Dot product of column i and column j
                dot = sum(centered_matrix[k][i] * centered_matrix[k][j] for k in range(n))
                cov[i][j] = dot / (n - 1)
        return cov
        
    def standardization(self, input):
        
        rows, cols = len(input), len(input[0])
        means = []
        for j in range(cols):
            col_sum = sum(input[i][j] for i in range(rows))
            means.append(col_sum / rows)
            
        return [[val - means[j] for j, val in enumerate(row)] for row in input]
    
    def frobenius_norm(self, input : list):
        ## Computes the euclidean distance in a matrix to make a single value
        ## Used mostly for loss when the loss for backprop is actually a matrix
        
        total = 0
        
        for row in input:
            for element in row:
                total += element ** 2
                
        return math.sqrt(total)
    
    def transpose(self, matrix : list):
        
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    def scalar_multiply(self, input : list, scalar):
        
        for i in range(len(input)):
            for j in range(len(input[0])):
                
                input[i][j] = input[i][j] * scalar   
                
        return input     
    
    
    def tanh(self, number):
        try:
            return (math.exp(number) - math.exp(-number)) / (math.exp(number) + math.exp(-number))
        except OverflowError:
            return 1.0 if number > 0 else -1.0
        
  
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
        

                    