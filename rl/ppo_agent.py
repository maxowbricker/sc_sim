"""
Proximal Policy Optimization (PPO) agent for dynamic weight adaptation.

Implements PPO to learn optimal λ1, λ2, λ3 weights for composite scoring function
based on real-time system state and performance feedback.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random


class PolicyNetwork(nn.Module):
    """Neural network for PPO policy."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        # Shared feature layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (outputs mean and std for continuous actions)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_std = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        features = self.shared(state)
        
        # Policy outputs
        mean = torch.tanh(self.policy_mean(features))  # Bounded to [-1, 1]
        std = F.softplus(self.policy_std(features)) + 1e-6  # Ensure positive
        
        # Value output
        value = self.value(features)
        
        return mean, std, value
    
    def get_action(self, state):
        """Sample action from policy."""
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob, value
    
    def evaluate_action(self, state, action):
        """Evaluate action probability and value."""
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return log_prob, value, entropy


class PPOAgent:
    """PPO agent for learning optimal weight parameters."""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, entropy_coef=0.01, value_coef=0.5):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space (system metrics)
            action_dim: Dimension of action space (λ1, λ2, λ3)
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of epochs per update
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Neural networks
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Experience buffer
        self.memory = PPOMemory()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        
    def get_action(self, state):
        """Get action from current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in memory."""
        self.memory.store(state, action, reward, next_state, done, log_prob, value)
    
    def update(self):
        """Update policy using PPO."""
        if len(self.memory) < 1:
            return
        
        # Get data from memory
        states, actions, rewards, next_states, dones, old_log_probs, values = self.memory.get_all()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # Calculate advantages and returns
        advantages, returns = self._calculate_advantages(rewards, values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Evaluate current policy
            log_probs, current_values, entropy = self.policy.evaluate_action(states, actions)
            
            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(current_values.squeeze(), returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.memory.clear()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _calculate_advantages(self, rewards, values, dones):
        """Calculate advantages using GAE."""
        advantages = []
        returns = []
        
        # Convert to numpy for easier calculation
        rewards = np.array(rewards)
        values = values.detach().cpu().numpy()
        dones = np.array(dones)
        
        # Calculate returns and advantages
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae  # GAE with λ=0.95
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def save(self, path):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class PPOMemory:
    """Memory buffer for PPO."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get_all(self):
        return (self.states, self.actions, self.rewards, self.next_states, 
                self.dones, self.log_probs, self.values)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)


def transform_action_to_weights(action):
    """
    Transform PPO action ([-1, 1]^3) to valid lambda weights.
    
    Maps:
    - action[0] -> λ1 (fairness weight): [0.1, 3.0]
    - action[1] -> λ2 (starvation weight): [0.1, 3.0] 
    - action[2] -> λ3 (utility weight): [0.1, 2.0]
    """
    lambda1 = 0.1 + (action[0] + 1) / 2 * 2.9  # Map [-1,1] to [0.1, 3.0]
    lambda2 = 0.1 + (action[1] + 1) / 2 * 2.9  # Map [-1,1] to [0.1, 3.0]
    lambda3 = 0.1 + (action[2] + 1) / 2 * 1.9  # Map [-1,1] to [0.1, 2.0]
    
    return lambda1, lambda2, lambda3


def extract_state_features(simulation_state, recent_history=None):
    """
    Extract state features for PPO from simulation state.
    
    State features include:
    - Current backlog (active + deferred tasks)
    - Worker availability ratio
    - Recent assignment success rate
    - Average wait time trend
    - Fairness distribution metrics
    - System load indicators
    """
    features = []
    
    # Basic system state
    total_workers = len(simulation_state.all_workers_map)
    available_workers = len(simulation_state.available_workers)
    active_tasks = len(simulation_state.active_tasks)
    deferred_tasks = len(simulation_state.deferred_tasks)
    
    # Normalized features
    features.extend([
        available_workers / max(1, total_workers),  # Worker availability ratio
        active_tasks / max(1, active_tasks + deferred_tasks + 1),  # Active task ratio
        deferred_tasks / max(1, active_tasks + deferred_tasks + 1),  # Deferred task ratio
        min(1.0, (active_tasks + deferred_tasks) / max(1, available_workers)),  # Load ratio
    ])
    
    # Worker fairness distribution
    if simulation_state.available_workers:
        fairness_scores = [w.fairness_ewma for w in simulation_state.available_workers]
        completed_tasks = [w.completed_tasks for w in simulation_state.available_workers]
        
        features.extend([
            np.mean(fairness_scores),
            np.std(fairness_scores) if len(fairness_scores) > 1 else 0,
            np.mean(completed_tasks),
            np.std(completed_tasks) if len(completed_tasks) > 1 else 0,
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # Recent performance (if history provided)
    if recent_history:
        features.extend([
            recent_history.get('recent_assignment_rate', 0),
            recent_history.get('recent_avg_wait', 0),
            recent_history.get('recent_fairness_trend', 0),
        ])
    else:
        features.extend([0, 0, 0])
    
    return np.array(features, dtype=np.float32)