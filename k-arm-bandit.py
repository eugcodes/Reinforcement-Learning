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
    def __init__(self, epsilon = [0.0, 0.01, 0.1, 1.0], k=10, run_size=2000):
        self.epsilon = epsilon
        zeroes = [0]*len(epsilon)
       
        self.dfs = {'epsilon:' + str(ep) : pd.DataFrame(columns = ['Run', 'Arm', 'Reward', 'Total_Reward', 'Avg Reward']) for ep in epsilon}
        
        self.k = k
        self.run_size = run_size
        
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

# for each epsilon, perform a run of run_size

# Run # | Explore | Arm | Reward | Total reward | Avg Reward


# test
# for eps in tb:
#     print(eps)
    
# for arm in tb.bandit:
#    print(arm.q_star)
#    print(arm.samples)
    
print(tb.dfs)

# Choose first run randomly
run_df = pd.DataFrame({
    'Run': range(1, tb.run_size + 1), 
    'Random value': [0] + list(np.random.uniform(0, 1, tb.run_size-1))
})

run_df['Explore?'] = (run_df['Random value'] < tb.epsilon[2]).astype(int) 

'''
run_df['Arm'] = np.random.randint(0, tb.k - 1, tb.run_size)

'''
print(run_df)
# do a single run: choose arm, calculate reward, update q_star for selected arm




