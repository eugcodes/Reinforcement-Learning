import numpy as np
import matplotlib.pyplot as plt
import random

# define the number of arms and actual rewards
k = 10
r = 2000

# define a list of length k to store the true value q*(a) of each action according to a normal distribution w/ mean 0 and unit variance
q_star = np.random.normal(0, 1, k)

# generate actual rewards for each action w/ mean q*(a), unit variance, normal distribution
rewards=[]
for a in range(0, k):
    rewards.append(np.random.normal(q_star[a], 1, r))
    
# visualize actual rewards as a violin plot
plt.violinplot(rewards)
plt.show()

# Evaluate performance of greedy and epsilon-greedy methods for different values of epsilon

epsilon = [0, 0.01, 0.1, 0.5, 1]
steps = 1000
runs = 2000

# assume initial action-value estimates are zero
q_estimate = np.zeros(k)

# single run
# generate a list of random numbers between 0 and 1 
choice = []
choice.append(random.sample(range((0,1),1000)))

avg_reward = []
avg_reward.append(0)

for i in range(0, steps):
    if choice[i] < epsilon:
        a = random.randint(0, k-1) # explore
    else:
        a = np.argmax(q_estimate) # exploit
    
    # update action-value estimates
    q_estimate[a] = q_estimate[a] + (1/(i+1))*(rewards[a][i] - q_estimate[a])
    
    # update avg-reward
    avg_reward = avg_reward + rewards[a][i]
    
    # (1/(i+1))*(rewards[a][i] - avg_reward)





