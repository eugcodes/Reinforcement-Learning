import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# define Arm class
class Arm:
    def __init__(self, *args):
        if len(args) == 1:
            self.q_star = args[0]
        else:
            self.q_star = np.random.normal(0, 1)
        self.samples = []
    
    def gen_samples(self, sample_size=2000):
        self.samples = np.random.normal(self.q_star, 1,sample_size)
        return self.samples
      
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

class agent:
    def __init__(self, k=10):
        self.k = k
        # assume initial q_star estimates are 0
        self.q_star_estimates = [0]*k

class testbed:
    def __init__(self, agent_class, k=10, epsilon = [0.0, 0.01, 0.5, 1.0], run_size=2000):
        self.agent = agent_class(k)
        
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


tb = testbed(agent,5)

# for each epsilon, perform a run of run_size

# Run # | Explore | Arm | Reward | Total reward | Avg Reward


# test
# for eps in tb:
#     print(eps)
    
# for arm in tb.bandit:
#    print(arm.q_star)
#    print(arm.samples)
    
print(tb.dfs)

run_df = pd.DataFrame(columns = ['Run', 'Random value', 'Explore?', 'Arm', 'Reward', 'Total_Reward', 'Avg Reward'])
    
# Choose first run randomly
initial_arm = np.random.randint(0, tb.k - 1)
initial_reward = tb.bandit.arms[initial_arm].gen_samples(1)[0]
rewards_dict = {
    'Run': 1,
    'Random value': 0,
    'Explore?': True,
    'Arm': initial_arm, 
    'Reward': initial_reward, 
    'Total_Reward': initial_reward, 
    'Avg Reward': initial_reward}

run_df = pd.concat([run_df, pd.DataFrame(rewards_dict, index=[0])], ignore_index=True)

arm_set = set(range(tb.k))

for i in range(2, tb.run_size + 1):
    random_value = np.random.uniform(0, 1)
    explore = (random_value < tb.epsilon[2])
    if explore:
        # exclude greedy arm from exploration
        excluded_arm = np.argmax([arm.q_star for arm in tb.bandit])
        avail_arms = list(arm_set - set([excluded_arm]))
        arm = np.random.choice(avail_arms)
    else:
        arm = np.argmax([arm.q_star for arm in tb.bandit])
    reward = tb.bandit.arms[arm].gen_samples(1)[0]
    
    # update q_star estimate
    total_reward = rewards_dict['Total_Reward'] + reward
    avg_reward = total_reward / i
    rewards_dict = {
        'Run': i,
        'Random value': random_value,
        'Explore?': explore,
        'Arm': arm, 
        'Reward': reward, 
        'Total_Reward': total_reward, 
        'Avg Reward': avg_reward}
    
    run_df = pd.concat([run_df, pd.DataFrame(rewards_dict, index=[0])], ignore_index=True)
    

print(initial_reward)
print(run_df)
print(arm_set)
print(excluded_arm)
print(avail_arms)
print(arm)

