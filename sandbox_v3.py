import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random
import torch
import torch.nn as nn
from collections import deque


# System parameters
M = 1.0   # Cart mass (kg)
m = 0.1   # Pole mass (kg)
l = 0.5   # Pole length (m)
g = 9.81  # Gravity (m/s²)
dt = 0.02 # Time step (s)
total_time = 10.0 # Simulation duration (s)
# Time steps for the animation
time_steps = 250

# Controller gains
Kp = 150.0  # Proportional gain for balancing
Kd = 30.0   # Derivative gain for balancing
K_swing = 15.0  # Gain for swing-up

# Set up the figure and axis for the animation
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)  # Limit x-axis from -2m to 2m
ax.set_ylim(-1, 1)  # Limit y-axis from -1m to 1m
ax.set_aspect('equal')

# Create the cart (a simple rectangle) and pole (a line)
cart, = ax.plot([], [], 'k-', lw=6)  # Cart rectangle (black)
pole, = ax.plot([], [], 'r-', lw=3)  # Pole line (red)
tracking_line, = ax.plot([], [], 'b--', lw=1)  # Tracking line (blue dashed)
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

def compute_accelerations(theta, theta_dot, F):
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    denominator = (M + m) - (m * cos_theta**2) / 2
    numerator_x = F + (m/2)*g*sin_theta*cos_theta - m*l*theta_dot**2*sin_theta
    x_ddot = numerator_x / denominator

    theta_ddot = (x_ddot * cos_theta + g * sin_theta) / (2 * l)

    return x_ddot, theta_ddot

def initialize_state():
    # Initialize the state
    theta = math.pi  # Angle (rad)
    theta_dot = 0.0  # Angular velocity (rad/s)
    x = 0.0       # Cart position (m)
    x_dot = 0.0      # Cart velocity (m/s)
    return [x, x_dot, theta, theta_dot]

def update_state(state, action):
    x, x_dot, theta, theta_dot = state
    x_ddot, theta_ddot = compute_accelerations(theta, theta_dot, action)

    # Euler integration
    x_dot += x_ddot * dt
    x += x_dot * dt
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

    return [x, x_dot, theta, theta_dot]

def control_law(theta, theta_dot):
    # Normalize theta to [-π, π]
    theta_norm = theta % (2 * math.pi)
    if theta_norm > math.pi:
        theta_norm -= 2 * math.pi

    # Control law with added bias to avoid always moving left
    if abs(theta_norm) > 0.2:  # Swing-up
        F = K_swing * theta_dot * math.cos(theta)
    else:  # Balancing
        F = -Kp * theta_norm - Kd * theta_dot

    # Optionally add a small bias to the force
    # Clip force to realistic limits
    F = np.clip(F, -10, 10)
    return F

def get_state(state):
    return np.array(state)

# Function to update the tracking line
def update_tracking_line():
    # Define a tracking path for the cart (in this case, a straight line)
    # You can modify this to create other types of paths (e.g., sine, parabola, etc.)
    x_values = np.linspace(-1.3, 1.3, 100)
    y_values = np.zeros_like(x_values)  # Path of the cart stays at y=0
    tracking_line.set_data(x_values, y_values)

def reward_function(cart_position, angle, cart_velocity=0, pole_velocity=0):

    # Normalize angle to [-π, π]
    angle_norm = angle % (2 * math.pi)
    if angle_norm > math.pi:
        angle_norm -= 2 * math.pi
        
    # Main reward component: how close to upright
    # Maps [-π, π] -> [0, 1] with 1 being upright
    upright_reward = (math.cos(angle_norm) + 1) / 2
    
    # Position penalty: how centered is the cart
    # Maps max track deviation to 0, centered to 1
    # pos_reward = max(0, 1 - abs(cart_position) / 2.0)
    
    # Combine rewards (upright position is most important)
    reward = 0.8 * upright_reward + 0.2 #* pos_reward

    # Early termination penalty
    # if abs(cart_position) > 1.5:  # If cart goes off track
        # reward = 0.0
        
    return reward

def step(current_state, action):
    x, _, theta, _ = current_state
    action = -10.0 if action == 0 else 10.0
    reward = reward_function(x, theta)
    new_state = update_state(current_state, action)
    return new_state, reward

def simulation():
    features = []
    rewards = []
    for _ in range(500):
        state = initialize_state()
        while True:
            x, x_dot, theta, theta_dot = state
            action = control_law(theta, theta_dot)
            reward = reward_function(x, theta)
            # Features array [state, action]
            feature = np.array([*state, action])
            # Update state
            updated_state = update_state(state, action)
            state = updated_state
            
            features.append(feature)
            rewards.append(reward)
            if reward > .99: break

    return np.array(features), np.array(rewards)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size))
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = ReplayBuffer(500000)
        
        # Neural Networks
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_agent():
    agent = DQLAgent(4, 2)
    for episode in range(500):
        score = 0
        state = initialize_state()
        while True:
            action = agent.act(state)
            next_state, reward = step(state, action)
            agent.memory.push(state, action, reward, next_state)

            loss = agent.train()
            state = next_state
            score += reward
            print(reward)
            if reward > 0.95:
                print('reward is above 80%!')
                break
        if episode % 10 == 0:
            agent.update_target_network()

    return agent

import time

def simulate_agent(agent):
    states = []
    actions = []
    for _ in range(1):
        state = initialize_state()
        done = False
        start_time = time.time()
        while not done:
            action = agent.act(state)
            states.append(state)
            actions.append(action)

            update_state, _ = step(state, action)
            state = update_state
            elapsed_time = time.time() - start_time
            if elapsed_time > 20:
                done = True 

    return np.array(states), np.array(actions)

agent = train_agent()
states, actions = simulate_agent(agent)

# print(len(states))
# print(rewards[0])

def update(i):
    x, x_dot, theta, theta_dot = states[i]
    # reward = reward_function(x, theta)
    # print(reward)
        # print(actions[frame])
        # print(abs(math.pi - theta))

        # Clamp the cart position to stay within the plot bounds
    x = np.clip(x, -1.2, 1.2)  # Prevent the cart from going too far left or right

    # Update cart position (a rectangle) and pole (a line)
    cart.set_data([x - 0.1, x + 0.1], [0, 0])  # Cart is a 1m wide rectangle
    pole.set_data([x, x + l * np.sin(theta)], [0, l * np.cos(theta)])

    time_text.set_text(f'Time: {i*dt:.2f}s')

    return cart, pole, time_text

update_tracking_line()  # Call once to initialize the tracking line
ani = FuncAnimation(fig, update, frames=len(states), interval=dt * 1000, blit=True)
plt.show()
