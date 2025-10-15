import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from rocket_env import SimpleRocketEnv

# ==== Hyperparameters ====
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 1000
MAX_EPISODES = 500
MAX_STEPS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Q-Network ====
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ==== Training ====
env = SimpleRocketEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_SIZE)

epsilon = EPSILON_START

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, done, truncated, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train
        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            q_values = policy_net(states).gather(1, actions).squeeze()
            next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Epsilon decay
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Update target network
    if episode % (TARGET_UPDATE // 10) == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode+1}/{MAX_EPISODES} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")

# ==== Save model ====
torch.save(policy_net.state_dict(), "dqn_torch.pth")
print("✅ Model DQN (PyTorch) berhasil disimpan sebagai dqn_torch.pth")
