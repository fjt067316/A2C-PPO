import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """Policy Network (Actor)"""
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)

        )
    
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class ValueNetwork(nn.Module):
    """Value Network (Critic)"""
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)


import torch

class A2C:
    """A2C (Advantage Actor-Critic) algorithm for continuous action spaces"""
    
    def __init__(self, obs_space_dims: int, action_space_dims: int, lr: float = 5e-4, gamma: float = 0.97):
        self.gamma = gamma
        self.action_dim = action_space_dims
        self.policy_net = PolicyNetwork(obs_space_dims, action_space_dims)
        self.value_net = ValueNetwork(obs_space_dims)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Member variables to store episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
    
    def sample_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(np.array([state]))
        action_q_vals= self.policy_net(state)
        value_pred = self.value_net(state)
        
        dist = torch.distributions.Categorical(action_q_vals)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        self.log_probs.append(log_prob)
        self.values.append(value_pred)
        
        action = action.numpy()
        return action.item()
    
    def update(self):
        """Updates the policy network's weights."""
        # Compute Q values (returns)
        running_g = 0
        gs = []
        rewards = np.array(self.rewards)
        # mean = rewards.mean()
        # std = rewards.std()
        # rewards = (rewards - mean) / std

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        # Convert lists to tensors
        gs = torch.tensor(gs, dtype=torch.float32)
        values = torch.cat(self.values).squeeze()
        log_probs = torch.cat(self.log_probs)

        # Compute advantages
        # advantages = [gs[i] + values[i+1]*self.gamma - values[i] for i in range(len(gs)-1)]
        # advantages.append(gs[-1]-values[-1])
        # advantages = torch.tensor(advantages)
        advantages = gs - values
        advantages = (advantages - advantages.mean()) / (advantages.std()+ 1e-10)

        # Compute losses
        policy_loss = - (log_probs * advantages.detach()).mean()
        value_loss = torch.mean((gs-values)**2)                     #F.smooth_l1_loss(values, gs)

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.policy_optimizer.step()

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1)
        self.value_optimizer.step()

        # Clear episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []


'''
Lunar Lander Discrete
'''
lunar_lander_params = {'continuous': False, 'gravity': -10.0, 'enable_wind': True, 'wind_power': 15.0, 'turbulence_power':1.5}
env = gym.make( "LunarLander-v2", **lunar_lander_params)
env_human = gym.make( "LunarLander-v2", render_mode="human", **lunar_lander_params)
title="LunarLander-v2"

# env = gym.make("Humanoid-v4")
# env_human = gym.make("Humanoid-v4", render_mode="human")
# title = "Humanoid-v4"

# TODO can I add other information in here to make viz better?
wrapped_env_data = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
wrapped_env_human = gym.wrappers.RecordEpisodeStatistics(env_human, 50)  # Records episode-reward


# TODO : implement a record video wrapper
# TODO ; implement SB3 solvers


total_num_episodes = int(1_000) #5e3 # Total number of episodes
printEveryNEps = 100 #50 
showViewHuman = True
viewEveryNEps = 200 # lander 500 # penduluam 200 #humanoid ??
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = 4
rewards_over_seeds = []

# seed_list = [1, 2, 3, 5, 8] # Fibonacci seeds
# see 7 seems easier to start with
seed_list = [7]#,5, 8, 9, 13] 
# seed_list = [7,5] 

for s, seed in enumerate(seed_list):
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = A2C(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        if showViewHuman & (episode % viewEveryNEps == 0): 
            wrapped_env = wrapped_env_human
        else:
            wrapped_env = wrapped_env_data

        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        episode_rewards = []
        while not done:
            # Collect current state and action
            action = agent.sample_action(obs)

            # Step in the environment
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            # Store the data
            agent.states.append(obs)
            agent.actions.append(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            done = terminated or truncated

        # Append reward information if needed
        reward_over_episodes.append(wrapped_env.return_queue[-1])

        # Update the agent after the episode ends
        agent.update()

        if episode % printEveryNEps == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print(f"Seed: {seed} ({s+1} of {len(seed_list)}) Episode: {episode} Average Reward: {avg_reward}")

    rewards_over_seeds.append(reward_over_episodes)

rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt() #what does melt do?
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title=f"A2C for {title}"
)
plt.show()
