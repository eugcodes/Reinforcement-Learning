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

class Agent:
    def __init__(self, k=10, epsilon=0.1, runs=1, run_size=1000):
        self.k = k
        self.epsilon = epsilon
        self.runs = runs
        self.run_size = run_size
        self.explore_flag = None
        self.first_run = True

        # assume initial q_star estimates are 0
        self.q_star_estimates = [{'arm': i, 'q_star': 0, 'Total Reward': 0, 'n': 0} for i in range(self.k)]
        self.avail_arms = set(range(self.k))

        # create a list of data frames, each for a run
        #self.run_data = [pd.DataFrame(columns = ['Step', 'Reward', 'Total_Reward', 'Avg Reward'])] * self.runs
        self.run_data = pd.DataFrame(columns = ['Step', 'Arm', 'Reward', 'Total_Reward', 'Avg Reward'] +['Q_star estimate' + str(arm) for arm in range(self.k)])

        self.current_index = 0 # current index for iteration
            
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= self.runs:
            self.current_index = 0
            raise StopIteration
        run_data = self.run_data[self.current_index]
        self.current_index += 1
        return run_data
    
    def choose_arm(self):
        if self.first_run == True: # first run
            self.explore_flag = True
            self.first_run = False
            selected_arm = np.random.choice(range(self.k)) # choose a random arm
        elif (np.random.uniform(0, 1) < self.epsilon): # Explore if epsilon-greedy condition is met
            self.explore_flag = True
            selected_arm = np.random.choice(range(self.k))  # choose a random arm
        else: #Exploit
            self.explore_flag = False
            selected_arm = np.argmax([self.q_star_estimates[i]['q_star'] for i in range(self.k)])
        return selected_arm
    
    def update_q_star_estimate(self, arm, reward):
        self.q_star_estimates[arm]['n'] += 1
        self.q_star_estimates[arm]['Total Reward'] += reward
        self.q_star_estimates[arm]['q_star'] = self.q_star_estimates[arm]['Total Reward'] / self.q_star_estimates[arm]['n']
        return self.q_star_estimates[arm]['q_star']
    
    def run(self, environment): 
        total_reward = 0
        # remaining runs based on epsilon-greedy selection method
        for i in range(self.run_size):
            action = self.choose_arm()
            reward = environment.arms[action].sample()
            total_reward += reward
            self.update_q_star_estimate(action, reward)
            # update run_data
            rewards_dict = {
                'Step': i+1,
                'Arm': action,
                'Reward': reward,   
                'Total_Reward': total_reward, 
                'Avg Reward': total_reward / (i+1)} | {
                'Q_star estimate' + str(arm) : self.q_star_estimates[arm]['q_star'] for arm in range(self.k)
                }
            self.run_data.loc[i] = [i, action, reward, total_reward, total_reward / (i+1)] + [ self.q_star_estimates[arm]['q_star'] for arm in range(self.k)]
        
        self.first_run = True # reset first_run flag for next run
        self.greedy_arm = -1 # reset greedy_arm for next run
        
        return self.run_data
    
# Create the TestBed and an Agent
n_arms = 10
test_bed = KArmBandit(n_arms)
agent = Agent(n_arms, 0.1, 1, 2000) # later, create 4 agents with 4 epsilons

#print(agent.run(test_bed).head(500))
print(agent.run(test_bed))

true_q =[]
for arm in test_bed:
    true_q.append(arm.q_star)

est_q = []
for arm in range(n_arms):
    est_q.append(agent.q_star_estimates[arm]['q_star'])

print(true_q) 
print(est_q)
print(np.array(true_q) - np.array(est_q))

#for i in agent:
#    agent.run(test_bed)
