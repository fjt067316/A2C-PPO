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

        hidden_1 = 256
        hidden_2 = 128
        dropout = 0.1
        
        self.net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_1),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            # nn.Dropout(dropout),
            nn.ReLU(),
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

        hidden_space1 = 32 
        hidden_space2 = 64 

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, 16),
            nn.Linear(16, action_space_dims)

        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, 16),
            nn.Linear(16, action_space_dims)
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


# %%
# Building an agent
# ~~~~~~~~~~~~~~~~~
#
# .. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig3.jpeg
#
# Now that we are done building the policy, let us develop **REINFORCE** which gives life to the policy network.
# The algorithm of REINFORCE could be found above. As mentioned before, REINFORCE aims to maximize the Monte-Carlo returns.
#
# Fun Fact: REINFROCE is an acronym for " 'RE'ward 'I'ncrement 'N'on-negative 'F'actor times 'O'ffset 'R'einforcement times 'C'haracteristic 'E'ligibility
#
# Note: The choice of hyperparameters is to train a decently performing agent. No extensive hyperparameter
# tuning was done.
#
import torch.nn.functional as F
import math

class PPO:
    """PPO algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via PPO algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        # Hyperparameters
        self.eps = 1e-8  # small number for mathematical stability
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.95  # Discount factor
        self.eps_clip = 0.2  # Clipping epsilon for PPO
        # self.critic_discount = 0.5  # Discount for critic loss
        # self.entropy_beta = 0.01  # Coefficient for entropy bonus
        # self.critic_loss_coeff = 0.5  # Coefficient for critic loss

        self.probs = []  # Stores probability values of the sampled action
        self.values = []  # Stores the value function estimates
        self.rewards = []  # Stores the corresponding rewards
        self.actions = []
        self.states = []

        self.policy_net = Policy_Network(obs_space_dims, action_space_dims)
        self.value_net = Value_Network(obs_space_dims)
        # self.optimizer = torch.optim.Adam(
        #     list(self.policy_net.parameters()) + list(self.value_net.parameters()),
        #     lr=self.learning_rate
        # )
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
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        # prob = self.calc_logprob(action_means, action_stddevs, value)
        prob = distrib.log_prob(action)
        # action = action.numpy()
        # action = np.clip(action, -1.0, 1.0)
        self.actions.append(action)
        # print(prob)
        self.probs.append(prob.sum())
        self.values.append(value) # append values 

        return action.numpy()

    def update(self):
        running_g = 0
        gs = []
        states = torch.stack(self.states).detach()
        # normalize rewards
        rewards = np.array(self.rewards)
        mean = rewards.mean()
        std = rewards.std()
        rewards = (rewards - mean) / std
        # Calculate discounted returns
        for R in rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)
    
        gs = torch.tensor(gs, dtype=torch.float32)
        values = torch.cat(self.values).squeeze()
        deltas = gs - values
        # advantages = (deltas - deltas.mean()) / (deltas.std() + 1e-10) # normalized
        advantages = deltas.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std()+ 1e-10)
        # gs = gs.detach()
        # actions = torch.stack(self.actions).detach()
        probs = torch.stack(self.probs).detach()
        actions_old = torch.stack(self.actions).detach()
        actions_old = actions_old.unsqueeze(1)

        
        policy_losses = []
        value_losses = []
        
        for _ in range(5): # ppo steps
            # log prob of current new actions
            action_means, action_stddevs = self.policy_net(states)
            value_pred = self.value_net(states).squeeze()
              
            distrib = Normal(action_means + self.eps, action_stddevs + self.eps)
            actions = distrib.sample()
            # print(actions.shape)
            new_log_prob_actions = distrib.log_prob(actions_old)#distrib.log_prob(actions)
            # print(new_log_prob_actions.shape)
            new_log_prob_actions = new_log_prob_actions.squeeze(1).sum(dim=1)
            
            # # new log_prob of old actions
            # # print(actions_old.shape)
            # old_log_prob_actions = distrib.log_prob(actions_old)
            # # print(old_log_prob_actions.shape)
            # old_log_prob_actions = old_log_prob_actions.squeeze(1).sum(dim=1)
            
            # print(old_log_prob_actions.shape)
            # print(new_log_prob_actions.shape)
            # print(self.probs.shape)
            # print(new_log_prob_actions[0])
            # print(self.probs[0])
            policy_ratio = (new_log_prob_actions - probs).exp()
            # print(policy_ratio.shape)
            # print(advantages.shape)
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.eps_clip, max = 1.0 + self.eps_clip) * advantages
            policy_loss = - torch.mean(torch.min(policy_loss_1, policy_loss_2))
            
            # print(gs.shape)
            # print(value_pred.shape)
            value_loss = torch.mean((gs-value_pred)**2)#F.smooth_l1_loss(gs, value_pred).mean()
            # print(value_loss)
            # print(policy_loss)
            loss = policy_loss + value_loss
            
            # print(f"value loss {value_loss} policy loss {policy_loss}")
            
            # self.optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            # policy_loss.backward()
            # value_loss.backward()
            
            # self.optimizer.step()
            self.policy_optimizer.step()
            self.value_optimizer.step()
        # for log_prob, value, advantage, reward in zip(self.probs, self.values, advantages, gs):
        #     ratio = torch.exp(log_prob - torch.mean(log_prob))  # Ensure you are using the correct old_log_prob if needed
        #     surr1 = ratio * advantage
        #     surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            
        #     policy_loss = -torch.min(surr1, surr2).mean()
        #     value_loss = self.critic_loss_coeff * F.mse_loss(value, reward)
            
        #     # Compute entropy loss for exploration
        #     entropy = -(log_prob * torch.exp(log_prob)).sum()
        #     entropy_loss = -self.entropy_beta * entropy
    
        #     policy_losses.append(policy_loss)
        #     value_losses.append(value_loss)
        #     entropy_losses.append(entropy_loss)
    
        # loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean() + torch.stack(entropy_losses).mean()
    def clear_data(self):
        # Clear stored information
        self.probs = []
        self.values = []
        self.rewards = []
        self.states = []
        self.actions = []


# Create and wrap the environment
# env = gym.make("InvertedPendulum-v4")
# env_human = gym.make("InvertedPendulum-v4", render_mode="human")
# title = "invertedpendulum-v4"

# lunar_lander_params = {'continuous': True, 'gravity': -10.0, 'enable_wind': True, 'wind_power': 15.0, 'turbulence_power':1.5}
# env = gym.make( "LunarLander-v2", **lunar_lander_params)
# env_human = gym.make( "LunarLander-v2", render_mode="human", **lunar_lander_params)


env_human = gym.make( "Humanoid-v4", render_mode="human")
env = gym.make( "Humanoid-v4")
env_human = gym.make( "Humanoid-v4", render_mode="human")
title="Humanoid-v4"
print(env.action_space)
# print(env.action_space.sample())
# env = gym.make("Humanoid-v4")
# env_human = gym.make("Humanoid-v4", render_mode="human")
# title = "Humanoid-v4"

# TODO can I add other information in here to make viz better?
wrapped_env_data = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
wrapped_env_human = gym.wrappers.RecordEpisodeStatistics(env_human, 50)  # Records episode-reward


# TODO : implement a record video wrapper
# TODO ; implement SB3 solvers


total_num_episodes = int(1_000) #5e3 # Total number of episodes
printEveryNEps = 1000 #50 
showViewHuman = True
viewEveryNEps = 200 # lander 500 # penduluam 200 #humanoid ??
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []

# seed_list = [1, 2, 3, 5, 8] # Fibonacci seeds
# see 7 seems easier to start with
seed_list = [7,5, 8, 9, 13] 
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
        if showViewHuman & (episode % viewEveryNEps == 0): 
            wrapped_env = wrapped_env_human
        else:
            wrapped_env = wrapped_env_data

        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        
        if (episode+1) % 10 == 0:
            agent.clear_data()

        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            # if render_only:
                # obs, reward, terminated, truncated, info = env_human.step(action)
            # else:
            # print(action)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        # if not render_only:
        reward_over_episodes.append(wrapped_env.return_queue[-1])
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
    title=f"PPO for {title}"
)
plt.show()
