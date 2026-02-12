import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import BaseFunctions.MathEquations as MATH
import BaseFunctions.StandardMLEquations as MLE

_math = MATH.Math()
_mle = MLE.SMLE()

import math


class AV_Scheduler:
    
    def __init__(self, lr, window_size=3, threshold = 0.8, factor = 0.5, boost = 1.2):
        r'''
        windows_size: how long you want the queue to be before the learning rate changes
        threshold: the maximum values required to "kick" the learning rate higher
        factor: Regular decay rate if everything is going swimmingly
        boost: How much you want it to be "kicked"
        '''
        self.window_size = window_size
        self.threshold = threshold
        self.factor = factor
        self.boost = boost
        self.history = []       ### Empty queue
        self.lr = lr
        
    def step(self, loss):
        if math.isnan(loss):
            print("NaN detected! Reducing LR significantly and skipping step.")
            self._update_lr(0.1) # Aggressive reduction
            return
        
        self.history.append(loss)
        
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
            volitility = _math.std_dev(self.history) ## If you make this into a matrix thing, change this
            
            if volitility < self.threshold:
                
                self.lr *= self.boost
                
            else:
                
                self.lr *= self.factor
                
    def _update_lr(self, multiplier):
        self.lr *= multiplier