import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import time

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
        self.q_star = np.zeros(k)
        self.total_reward = np.zeros(k)
        self.n = np.zeros(k, dtype=int)
    
    def choose_arm(self):
        if self.first_step:  # initial step is exploration
            self.first_step = False
            return np.random.randint(self.k)
        elif (np.random.uniform(0, 1) < self.epsilon): # explore
            return np.random.randint(self.k) # assumption: exploration includes greedy arm
        else: # Else Exploit
            return np.argmax(self.q_star)
    
    def update_model(self, arm, reward):
        self.n[arm] += 1
        self.total_reward[arm] += reward
        self.q_star[arm] = self.total_reward[arm] / self.n[arm]

    def run(self, environment): 
        total_rewards = np.zeros(self.run_size)
        self.first_step = True # reset first_step flag for each run
        
        for i in range(self.run_size):
            chosen_arm = self.choose_arm()
            reward = environment.arms[chosen_arm].sample()
            self.update_model(chosen_arm, reward)
            
            # update run_data
            #row_data = [i+1, action, reward, total_reward, total_reward / (i+1)] + list(self.model['q_star'])
            #run_results.append(row_data)
            
            # only produce an array of average reward at each step to optimize performance
            total_rewards[i] = reward
            
        return np.cumsum(total_rewards) / (np.arange(1, self.run_size + 1)) # average reward at each step
            
        # columns = ['Step', 'Arm', 'Reward', 'Total_Reward', 'Avg Reward'] + ['Q_star estimate' + str(_) for _ in range(self.k)]
        # self.run_data = pd.DataFrame(run_results, columns=columns)
        
        #return self.run_data

# Execute multiple runs of each Agent in the TestBed environment and average the results at each step
n_arms = 10
eps = [0, 0.01, 0.1]
runs = 2000
steps = 1000

agents = [Agent(n_arms, epsilon, steps) for epsilon in eps]
avg_rewards_dict = {}

# create a test bed instance for each run
start_time = time.time()

# test_bed is an array of size runs of k-Arm Bandits 
test_bed = [KArmBandit(n_arms) for _ in range(runs)]

# For each epsilon agent, create 2-D arrays of avg rewards x runs, then average the runs to create a 1-D array of avg rewards
for agent in agents:
    avg_rewards = np.zeros(steps) # initialize array of avg rewards for this epsilon agent
    
    #for _ in range(runs):
    for bandits in test_bed:
        avg_rewards += agent.run(bandits)  # sum of avg rewards across all runs for this epsilon agent
    
    avg_rewards /= runs # average of avg rewards across all runs for this epsilon agent
    avg_rewards_dict['epilson = ' + str(agent.epsilon)] = avg_rewards # add to avg rewards table
    
end_time = time.time()
elapsed_time = end_time - start_time
print (f"Elapsed time: {elapsed_time} seconds")
        
""" for 
        current_run = agent.run(test_bed)
        #current_run = [agent.epsilon]*agent.run_size # testing
        exp_results.append(current_run) # array of arrays

    # Average the Rewards at each step
     """

# Average the Rewards at each step
#exp_results = np.array(exp_results)
#avg_rewards = np.mean(exp_results[:, :, 4], axis=0) # average of avg rewards at each step
""" 
print(all_avg_rewards[0])
print(all_avg_rewards[1])
print(all_avg_rewards[2]) """

# print(len(avg_rewards))

# Plot average rewards at each step
#for avg_rewards, epsilon in zip(all_avg_rewards, eps):
for epsilon, avg_rewards in avg_rewards_dict.items():
    plt.plot(avg_rewards, label=f"ε={epsilon}")


plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Steps')
plt.legend()
plt.show()

#cols = ['Step', 'Total_Reward', 'Avg Reward']
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

