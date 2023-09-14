import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# define Arm class
class Arm:
    def __init__(self, q_star=None):
        self.q_star = q_star if q_star is not None else np.random.normal(0, 1)
        self.samples = []
        self.current_index = 0 # current index for iteration
    
    def sample(self, sample_size=1):
        self.samples = np.random.normal(self.q_star, 1, sample_size)
        if sample_size == 1:
            return self.samples[0]
        else:
            return self.samples
     
    def __next__(self):
        if self.current_index >= len(self.samples):
            self.current_index = 0 # Reset index for future iterations
            raise StopIteration
        sample = self.samples[self.current_index]
        self.current_index += 1
        return sample
      
# define k-arm bandit class 
class KArmBandit:
    def __init__(self, k=10):
        self.k = k
        self.arms = [Arm() for _ in range(self.k)]
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

class Agent:
    def __init__(self, k=10, epsilon=0.1, run_size=1000):
        self.k = k
        self.epsilon = epsilon
        self.run_size = run_size
        self.first_step = True

        # assume initial q_star estimates are 0
        self.model = pd.DataFrame({
            'arm': range(self.k),
            'q_star': [0] * self.k,
            'Total Reward': [0] * self.k,
            'n': [0] * self.k
        })
    
    def choose_arm(self):
        # Explore if first run iteration or epsilon-greedy condition is met
        if self.first_step or (np.random.uniform(0, 1) < self.epsilon): 
            self.first_step = False
            return np.random.randint(self.k) 
        # Else Exploit
        else: 
            return self.model['q_star'].idxmax()
    
    def update_model(self, arm, reward):
        self.model.loc[arm, 'n'] += 1
        self.model.loc[arm, 'Total Reward'] += reward
        self.model.loc[arm, 'q_star'] = self.model.loc[arm, 'Total Reward'] / self.model.loc[arm, 'n']
    
    def run(self, environment): 
        total_reward = 0
        run_results = []
        
        # remaining runs based on epsilon-greedy selection method
        for i in range(self.run_size):
            action = self.choose_arm()
            reward = environment.arms[action].sample()
            total_reward += reward
            self.update_model(action, reward)
            
            # update run_data
            row_data = [i+1, action, reward, total_reward, total_reward / (i+1)] + list(self.model['q_star'])
            run_results.append(row_data)
        
        columns = ['Step', 'Arm', 'Reward', 'Total_Reward', 'Avg Reward'] + ['Q_star estimate' + str(_) for _ in range(self.k)]
        self.run_data = pd.DataFrame(run_results, columns=columns)
        
        self.first_step = True # reset first_step flag for next run
        
        return self.run_data
    
# Create the TestBed and an Agent
n_arms = 10
test_bed = KArmBandit(n_arms)
agent = Agent(n_arms, 0.1, 1000) 
runs = 200

# Run Agent in TestBed environment n times and average the results at each step

exp_results = []
for i in range(runs):
    current_run = agent.run(test_bed)
    exp_results.append(current_run) # array of df's
    # print(current_run)

# Average the Rewards at each step
exp_results = np.array(exp_results)
avg_rewards = np.mean(exp_results[:, :, 4], axis=0) # average of avg rewards at each step

# print(avg_rewards)
# print(len(avg_rewards))

# Plot average rewards at each step
plt.plot(avg_rewards)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Steps')
plt.show()


cols = ['Step', 'Total_Reward', 'Avg Reward']
#exp_results = pd.DataFrame(________, columns=cols)

""" r1 = agent.run(test_bed)
true_q = [arm.q_star for arm in test_bed.arms]
est_q = list(agent.model['q_star'])

print(r1)
print(true_q) 
print(est_q)
print(np.array(true_q) - np.array(est_q))


r2 = agent.run(test_bed)

# print(agent.run(test_bed))

true_q = [arm.q_star for arm in test_bed.arms]
est_q = list(agent.model['q_star'])

print(r2)
print(true_q) 
print(est_q)
print(np.array(true_q) - np.array(est_q))
 """

