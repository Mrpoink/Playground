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
    
    def softmax_vector(self, logits):
        v = self._to_vec(logits)
        if not v:
            return [[0.0]]
        m = max(v)
        exps = [math.exp(x - m) for x in v]
        s = sum(exps)
        probs = [e / s for e in exps]
        return self._to_col(probs)
    
    def _to_col(self, vec):
        # [N] -> N x 1
        return [[v] for v in vec]
    
    def _to_vec(self, mat):
        # Flatten 1xN or Nx1 to [N]
        if isinstance(mat, list):
            if not mat:
                return []
            if isinstance(mat[0], list):
                if len(mat) == 1:
                    return mat[0]
                if all(len(r) == 1 for r in mat):
                    return [r[0] for r in mat]
                raise ValueError("CCE expects a vector, got a 2D matrix")
            return mat
        return [mat]
    
    def vector_CCEL(self, p, y):
        
        pv = self._to_vec(p)
        if isinstance(y, int):
            if y < 0 or y >= len(pv):
                raise ValueError(f"class index {y} out of range 0..{len(pv)-1}")
            eps = 1e-12
            return -math.log(max(pv[y], eps))
        
        yv = self._to_vec(y)
        if len(pv) != len(yv):
            raise ValueError(f"CCE shapes mismatch: p={len(pv)}, y={len(yv)}")
        
        eps = 1e-12
        
        return -sum(yv[i] * math.log(max(pv[i], eps)) for i in range(len(pv)))
    
    def vector_cce_der(self, p, y):
        
        pv = self._to_vec(p)
        if isinstance(y, int):
            yv = [0.0] * len(pv)
            yv[y] = 1.0
        else:
            yv = self._to_vec(y)
            if len(pv) != len(yv):
                raise ValueError(f"CCE der mismatch: p={len(pv)}, y={len(yv)}")
        diff = [pv[i] - yv[i] for i in range(len(pv))]
        return self._to_col(diff)

    
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
        
        diff = self._math.matrix_subtraction(y_hat, y)
        squared_diff = self._math.hadamard_product(diff, diff)
        
        total_loss = 0
        for row in squared_diff:
            total_loss += sum(row)
                    
        return total_loss / len(y_hat)
    
    def MSE_loss_matrix(self, y_hat: list, y : list):
        
        diff = self._math.matrix_subtraction(y_hat, y)
        squared_diff = self._math.hadamard_product(diff, diff)
        
        new_matrix = []
        
        for row in squared_diff:
            row_sum = sum(row)
            new_matrix.append([row_sum / len(y_hat)])
        
        return new_matrix
        
        
    def MSE_loss_der(self, y_hat : list, y : list):
        
        return self._math.matrix_subtraction(y_hat, y)    
        
    
    def sigmoid_derivative(self, matrix : list):
        
        blank = self.get_blank_matrix(len(matrix), len(matrix[0]), 1)
        sub = self._math.matrix_subtraction(blank, matrix)
        
        return self._math.hadamard_product(matrix, sub)
    
    
    
    def BackPropagation_Step(self, output_node : NN.Neuron, loss : list, learning_rate : float, is_final = False):
        
        a_prev = output_node.input          # (I x B)
        y_hat = output_node.output          # (H x B)
        z = output_node.z                   # (H x B)
        
        H = len(y_hat)
        B = len(y_hat[0])
        
        if is_final:
            delta = loss
            
        else:
            
            # sigmoid` (z) = y_hat * (1 - y_hat)
            sigma_prime = [[y_hat[i][j] * (1.0 - y_hat[i][j]) for j in range(B)] for i in range(H)]
            # This is the delta (change) constant for the weights given the error
            delta = self._math.hadamard_product(loss, sigma_prime) # (H x B)
        
        # Transpose previous input
        a_prev_t = self._math.transpose(a_prev)     # (B x I)
        # Find delta for weights
        dW = self._math.matrix_multiply(delta, a_prev_t)    # (H x I)
        
        # db = sum over batch (Bias update)
        db = [[sum(delta[i])] for i in range(H)]        # (H x 1)
        
        # prevent exploding gradients
        dW = self.gradient_clip(dW, threshold=1.0)
        db = self.gradient_clip(db, threshold=1.0)
        
        # Now to start updating the weights with thier corresponding changes
        output_node.weight = self._math.matrix_subtraction(
            output_node.weight,
            self._math.scalar_multiply(dW, learning_rate)
            )
        
        output_node.bias = self._math.matrix_subtraction(
            output_node.bias,
            self._math.scalar_multiply(db, learning_rate)
        )
        
        # Propagate error: error_prev = W^T @ delta
        W_T = self._math.transpose(output_node.weight)  # (I x H)
        error_prev = self._math.matrix_multiply(W_T, delta)  # (I x B)
        
        return error_prev
        
        
        
    
    def BackPropagation_Step_RNN(self, dh, x_t, h_prev, h_t, U):
        x_t = self._math.as_col(x_t)
        h_prev = self._math.as_col(h_prev)
        h_t = self._math.as_col(h_t)
        dh = self._math.as_col(dh)

        dtanh = self._math.matrix_tanh_derivative(h_t)     # H x 1
        dz = self._math.hadamard_product(dh, dtanh)        # H x 1
        
        # print(f"Dimensions for dtanh and dh: {len(dtanh)}, {len(dtanh[0])}   {len(dh)}, {len(dh[0])}")

        T_input = self._math.transpose(x_t)                # 1 x I
        T_hprev = self._math.transpose(h_prev)             # 1 x H

        grad_w = self._math.dot_product(dz, T_input)       # H x I
        grad_h = self._math.dot_product(dz, T_hprev)       # H x H
        grad_b = dz                                        # H x 1

        dh_prev = self._math.dot_product(self._math.transpose(U), dz)  # H x 1
        return grad_w, grad_h, grad_b, dh_prev

    def flatten_output(self, history_states):
        r'''
        Best used for RNNs'''
        predicted_indices = []
        
        for state_matrix in history_states:
            
            flat_vector = [row[0] for row in state_matrix]
            
            
            best_index = flat_vector.index(max(flat_vector))
            
            predicted_indices.append(best_index)
            
        return predicted_indices
    
    def cosine_scheduler(self, epoch, total_epochs, max, min):
        r'''
        epoch: current epoch
        total_epochs: total epochs
        max: Max learning rate
        min: minimum learning rate
        returns: a float for a learning rate'''
        
        fraction = epoch / total_epochs
        return min + 0.5 * (max - min) * (1 + math.cos(fraction * math.pi))
    
    def gradient_clip(self, matrix: list, threshold=1.0):
        """Designed to reduce the explosive or vanshing gradients

        Args:
            matrix (list): gradient
            threshold (float, optional): Determines range. Defaults to 1.0.

        Returns:
            matrix (list): clipped gradient
        """
        return [[max(min(val, threshold), -threshold) for val in row] for row in matrix]
    
    def generate_text(self, history, tokenizer):
        
        
        predicted_text = ""
        for state in history:
            best_char = "?"
            min_dist = float('inf')
            
            for char, idx in tokenizer.char_to_int.items():
                emb = [[val] for val in tokenizer.lookup_table[idx]]
                # Use your MSE_loss to find the closest match
                dist = self.MSE_loss(state, emb)
                if dist < min_dist:
                    min_dist = dist
                    best_char = char
            predicted_text += best_char
        return predicted_text
    
    def position_wise(self, seq_len, d_model):
        
        pe = [[0.0 for _ in range(d_model)] for _ in range(seq_len)]
        
        for position in range(seq_len):
            for i in range(0, d_model, 2):
                
                denominator = math.pow(10000, (2 * i) / d_model)
                
                pe[position][i] = math.sin(position / denominator)
                
                if i + 1 < d_model:
                    pe[position][i + 1] = math.cos(position / denominator)
                    
        return pe
    
    
    def generate_mask(self, seq_len):
        
        mask = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
        
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                mask[i][j] = -1e9
                
        return mask
    
    def backprop_att(self, soft_scores, Q, K, V, dZ, X_q, X_kv=None):
        
        if X_kv is None:
            X_kv = X_q
        
        dV = self._math.dot_product(self._math.transpose(soft_scores), dZ)
        
        dWv = self._math.dot_product(self._math.transpose(X_kv), dV)
        dWeights = self._math.dot_product(dZ, self._math.transpose(V))
        
        dScore = [[0.0 for _ in range(len(soft_scores[0]))] for _ in range(len(soft_scores))]
        num = 0
        for num, row in enumerate(soft_scores):
            weighted_gradient = self._math.dot_product_vectors(soft_scores[num], dWeights[num])
            for i in range(len(row)):
                dScore[num][i] = row[i] * (dWeights[num][i] - weighted_gradient)
                
                
                
        dQ = self._math.dot_product(dScore, K)
        dK = self._math.dot_product(self._math.transpose(dScore), Q)
        
        d_k = math.sqrt(len(K[0]))
        
        dWq = self._math.dot_product(self._math.transpose(X_q), self._math.scalar_divide(dQ, d_k))
        dWk = self._math.dot_product(self._math.transpose(X_kv), self._math.scalar_divide(dK, d_k))
        
        dWq = self.gradient_clip(dWq, 0.1)
        dWk = self.gradient_clip(dWk, 0.1)
        
        return dWq, dWk, dWv, dQ, dK, dV
    
    def split_heads(self, matrix, heads):
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        d_head = cols // heads
        
        heads = []
        for h in range(heads):
            
            start = h * d_head
            end = start * d_head
            
            head_mat = [row[start:end] for row in matrix]
            heads.append(head_mat)
            
        return heads                
    
