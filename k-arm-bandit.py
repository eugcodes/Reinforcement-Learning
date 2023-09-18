import numpy as np
import matplotlib.pyplot as plt
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
        self.q_star[arm] += (1/self.n[arm]) * (reward - self.q_star[arm])

    def run(self, environment): 
        # Reset model for each run
        self.q_star = np.zeros(self.k)
        self.total_reward = np.zeros(self.k)
        self.n = np.zeros(self.k, dtype=int)
        
        run_rewards = np.zeros(self.run_size)
        self.first_step = True # reset first_step flag for each run
        
        for i in range(self.run_size):
            chosen_arm = self.choose_arm()
            reward = environment.arms[chosen_arm].sample()
            self.update_model(chosen_arm, reward)
            run_rewards[i] = reward
            
        return run_rewards

 
# Execute multiple runs of each Agent in the TestBed environment and average the results at each step
n_arms = 10
eps = [0, 0.01, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
runs = 20000
steps = 1000

agents = [Agent(n_arms, epsilon, steps) for epsilon in eps]
avg_rewards_dict = {}

# create a test bed instance for each run
start_time = time.time()

# test_bed is an array of size runs of k-Arm Bandits 
test_bed = [KArmBandit(n_arms) for _ in range(runs)]

# For each epsilon agent, create 2-D arrays of rewards x runs, then average the runs to create a 1-D array of avg rewards
for agent in agents:
    rewards = [] # initialize array of rewards for this epsilon agent
    
    for bandits in test_bed:
        rewards.append(agent.run(bandits))  # rewards for each run for this epsilon agent
    
    avg_rewards = np.mean(rewards, axis=0) # average of avg rewards at each step
    
    avg_rewards_dict['epilson = ' + str(agent.epsilon)] = avg_rewards # add to avg rewards table
    
end_time = time.time()
elapsed_time = end_time - start_time
print (f"Elapsed time: {elapsed_time} seconds")

plt.ion()
# Plot average rewards at each step
plt.figure()
for epsilon, avg_rewards in avg_rewards_dict.items():
    plt.plot(avg_rewards, label=f"ε={epsilon}")
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Steps')
plt.legend()
plt.draw()
plt.pause(0) 
    
'''
# Violin plot Testbed Arms q_star
testbed_q_stars = []
for bandit in test_bed:
    testbed_q_stars.append([arm.q_star for arm in bandit.arms])
testbed_q_stars = np.array(testbed_q_stars)

plt.figure()
plt.violinplot(testbed_q_stars, showmeans=True)
plt.xlabel('Arm')
plt.ylabel('q_star')
plt.title('Distribution of q_star values for each arm')
plt.xticks(range(1, testbed_q_stars.shape[1] + 1))  # Set x-ticks to represent arm numbers
plt.grid(True, axis='y')
plt.draw()
plt.pause(0) 

# Violin plot of 2000 samples for each arm in one instance of a bandit
bandit_instance_samples = []
for i in range(n_arms):
    bandit_instance_samples.append(test_bed[0].arms[i].sample(2000)) 
bandit_instance_samples = np.array(bandit_instance_samples).T

plt.figure()
plt.violinplot(bandit_instance_samples, showmeans=True)
plt.xlabel('Arm')
plt.ylabel('q_star')
plt.title('Distribution of samples for each arm of an instance of a k-arm bandit')
plt.xticks(range(1, n_arms + 1))  # Set x-ticks to represent arm numbers

plt.grid(True, axis='y')
plt.draw()
plt.pause(0)
'''