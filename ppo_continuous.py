import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)


class Value_Network(nn.Module):
    def __init__(self, obs_space_dims: int):
        super().__init__()

        hidden_1 = 512
        hidden_2 = 256
        dropout = 0.1
        
        self.net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_1),
            nn.Tanh(),
            nn.Linear(hidden_1, hidden_2),
            nn.Tanh(),
            nn.Linear(hidden_2, 1)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 256 
        hidden_space2 = 64 
        hidden_3 = 128

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_3),
            nn.Tanh(),
            nn.Linear(hidden_3, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, 32),
            nn.Linear(32, action_space_dims)

        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, 32),
            nn.Linear(32, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        # if the nn shared features are f(x)
        # then action_means are mu(f(x))
        action_means = self.policy_mean_net(shared_features)

        # then the action_stddevs are std(f(x)) = log(1+ e^{f(x)})
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs

class PPO:
    """PPO algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via PPO algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        # hyperparameters
        self.eps = 1e-8  # small number for mathematical stability
        self.learning_rate = 1e-4  # learning rate for policy optimization
        self.gamma = 0.97  # discount factor
        self.eps_clip = 0.2  # clipping epsilon for PPO

        self.probs = []  # stores probability values of the sampled action
        self.values = []  # stores the value function estimates
        self.rewards = []  # stores the corresponding rewards
        self.actions = []
        self.states = []

        self.policy_net = Policy_Network(obs_space_dims, action_space_dims)
        self.value_net = Value_Network(obs_space_dims)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]), dtype=torch.float32)
        self.states.append(state)
        action_means, action_stddevs = self.policy_net(state)
        value = self.value_net(state)

        # create a normal distribution from the predicted
        # mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        self.actions.append(action)
        self.probs.append(prob.mean())
        self.values.append(value) # append values 

        return action.numpy()

    def update(self):
        
        gs = []
        states = torch.stack(self.states).detach()
        
        for ep_rewards in self.rewards:
            running_g = 0
            ep_gs = []
            # normalize rewards
            rewards = np.array(ep_rewards)
            # mean = rewards.mean()
            # std = rewards.std()
            # rewards = (rewards - mean) / std
            
            # Calculate discounted returns
            for R in rewards[::-1]:
                running_g = R + self.gamma * running_g
                ep_gs.insert(0, running_g)
            gs.extend(ep_gs)
    
        gs = torch.tensor(gs, dtype=torch.float32)
        values = torch.cat(self.values).squeeze()
        advantages = gs - values
        advantages = advantages.detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std()+ 1e-10)
        
        probs = torch.stack(self.probs).detach()
        actions_old = torch.stack(self.actions).detach()
        actions_old = actions_old.unsqueeze(1)
        
        for _ in range(5): # ppo steps
            # log prob of current new actions
            action_means, action_stddevs = self.policy_net(states)
            value_pred = self.value_net(states).squeeze()
              
            distrib = Normal(action_means + self.eps, action_stddevs + self.eps)

            new_log_prob_actions = distrib.log_prob(actions_old)
            new_log_prob_actions = new_log_prob_actions.squeeze(1).mean(dim=1)

            policy_ratio = (new_log_prob_actions - probs).exp() # substracting log probs is division

            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.eps_clip, max = 1.0 + self.eps_clip) * advantages
            policy_loss = - torch.mean(torch.min(policy_loss_1, policy_loss_2))
            
            value_loss = torch.mean((gs-value_pred)**2)
            loss = 0.5*(policy_loss + value_loss)
            
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1)

            self.policy_optimizer.step()
            self.value_optimizer.step()
    
    def clear_data(self):
        # Clear stored information
        self.probs = []
        self.values = []
        self.rewards = []
        self.states = []
        self.actions = []



# lunar_lander_params = {'continuous': True, 'gravity': -10.0, 'enable_wind': True, 'wind_power': 15.0, 'turbulence_power':1.5}
# env = gym.make( "LunarLander-v2", **lunar_lander_params)
# env_human = gym.make( "LunarLander-v2", render_mode="human", **lunar_lander_params)
# title="LunarLander-v2"

env_human = gym.make( "Humanoid-v4", render_mode="human")
env = gym.make( "Humanoid-v4")
env_human = gym.make( "Humanoid-v4", render_mode="human")
title="Humanoid-v4"
# print(env.action_space)


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
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []
lr_decay_t = 50
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
    agent = PPO(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        if False and showViewHuman & (episode % viewEveryNEps == 0): 
            wrapped_env = wrapped_env_human
        else:
            wrapped_env = wrapped_env_data

        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        
        if (episode+1) % 10 == 0:
            agent.clear_data()
        ep_rewards = []
        while not done:
            action = agent.sample_action(obs)


            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            ep_rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        # if not render_only:
        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.rewards.append(ep_rewards)
        agent.update()
        
        # if episode % lr_decay_t == 0:
        #     agent.learning_rate = max(agent.learning_rate * 0.9, 5e-5)

        if episode % printEveryNEps == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print(f"Seed: {seed} ({s+1} of {len(seed_list)}) Episode: {episode} Average Reward: {avg_reward}")

    rewards_over_seeds.append(reward_over_episodes)

rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt() #what does melt do?
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title=f"PPO for {title} (no reward norm)"
)
plt.show()
