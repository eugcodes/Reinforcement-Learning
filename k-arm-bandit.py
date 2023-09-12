import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# define Arm class
class Arm:
    def __init__(self):
        self.q_star = np.random.normal(0, 1)
        self.samples = []
        
    def gen_samples(self, sample_size=2000):
        self.samples = [np.random.normal(self.q_star, 1, self.sample_size)]
      
# define k-arm bandit class as an iterable  
class k_arm_bandit:
    def __init__(self, k=10):
        self.k = k
        self.arms = [Arm() for i in range(self.k)]
        self.current_index = 0 # current index for iteration
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= self.k:
            self.current_index = 0 # Reset index for future iterations
            raise StopIteration
        arm = self.arms[self.current_index]
        self.current_index += 1
        return arm

class testbed:
    def __init__(self, epsilon = [0, 0.01, 0.1, 1], k=10, sample_size=2000):
        self.epsilon = epsilon
        zeroes = [0]*len(epsilon)
       
        self.df = pd.DataFrame({'epsilon': epsilon, 'total_reward': zeroes, 'steps': zeroes, 'avg_reward': zeroes})
        
        self.k = k
        self.sample_size = sample_size
        
        self.bandit = k_arm_bandit(self.k)
        self.current_index = 0 # current index for iteration
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= len(self.epsilon):
            self.current_index = 0  # Reset index for future iterations
            raise StopIteration
        eps = self.epsilon[self.current_index]
        self.current_index += 1
        return eps 
 
tb = testbed()

# for each epsilon, perform a run


# test
for eps in tb:
    print(eps)
    
for arm in tb.bandit:
    print(arm.q_star)
    print(arm.samples)
    
print(tb.df)


