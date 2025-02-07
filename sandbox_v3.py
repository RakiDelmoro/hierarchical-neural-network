import math
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation


# System parameters
cart_mass = 1.0   # Cart mass (kg)
pole_mass = 0.1   # Pole mass (kg)
pole_length = 0.5   # Pole length (m)
gravity = 9.81  # Gravity (m/s²)
time_step = 0.02 # Time step (s)
total_time = 10.0 # Simulation duration (s)
# Time steps for the animation
time_steps = 250

# Controller gains
Kp = 150.0  # Proportional gain for balancing
Kd = 30.0   # Derivative gain for balancing
K_swing = 15.0  # Gain for swing-up

# Set up the figure and axis for the animation
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)  # Limit x-axis from -3m to 3m
ax.set_ylim(-1, 1)  # Limit y-axis from -1m to 1m
ax.set_aspect('equal')

# Create the cart (a simple rectangle) and pole (a line)
cart, = ax.plot([], [], 'k-', lw=6)  # Cart rectangle (black)
pole, = ax.plot([], [], 'r-', lw=3)  # Pole line (red)
tracking_line, = ax.plot([], [], 'b--', lw=1)  # Tracking line (blue dashed)
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
force_text = ax.text(0.05, 0.90, '', transform=ax.transAxes)
force_text.set_text('')

def compute_accelerations(theta, theta_dot, F):
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    denominator = (cart_mass + pole_mass) - (pole_mass * cos_theta**2) / 2
    numerator_x = F + (pole_mass/2)*gravity*sin_theta*cos_theta - pole_mass*pole_length*theta_dot**2*sin_theta
    x_ddot = numerator_x / denominator

    theta_ddot = (x_ddot * cos_theta + gravity * sin_theta) / (2 * pole_length)

    return x_ddot, theta_ddot

def initialize_state():
    # Initialize the state
    theta = math.pi  # Angle (rad)
    theta_dot = 0.0  # Angular velocity (rad/s)
    x = 0.0       # Cart position (m)
    x_dot = 0.0      # Cart velocity (m/s)
    return [x, x_dot, theta, theta_dot]

def update_state(state, force):
    x, x_dot, theta, theta_dot = state
    
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Compute accelerations using equations of motion
    temp = (force + pole_mass * pole_length * theta_dot**2 * sin_theta) / (cart_mass + pole_mass)
    theta_acc = (gravity * sin_theta - cos_theta * temp) / (pole_length * (4.0/3.0 - pole_mass * cos_theta**2 / (cart_mass + pole_mass)))
    x_acc = temp - pole_mass * pole_length * theta_acc * cos_theta / (cart_mass + pole_mass)

    # Euler integration
    x_dot += x_acc * time_step
    x += x_dot * time_step
    theta_dot += theta_acc * time_step
    theta += theta_dot * time_step

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
    x_values = np.linspace(-2.3, 2.3, 100)
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
    
    '''So far we don't penalize it when the cart is not centered (Simplicity)'''
    # Position penalty: how centered is the cart
    # Maps max track deviation to 0, centered to 1
    # pos_reward = max(0, 1 - abs(cart_position) / 2.0)
    
    # Combine rewards (upright position is most important)
    reward = 0.8 * upright_reward + 0.2 #* pos_reward

    # Early termination penalty
    # if abs(cart_position) > 1.5:  # If cart goes off track
        # reward = 0.0
        
    return reward

# def step(current_state, action):
#     x, _, theta, theta_dot = current_state
#     # action = -10.0 if action == 0 else 10.0
#     # reward = reward_function(x, theta)
#     angle_threshold = 0.5  # ~28.6 degrees for upright position
#     theta_limit = 12 * math.pi / 60  # Small angle limit for balance
#     position_limit = 1.5    # Cart position limits
    
#     # Normalize theta to [-pi, pi]
#     theta = ((theta + math.pi) % (2 * math.pi)) - math.pi

#     done = abs(x) > position_limit or abs(theta_dot) > 15.0
    
#     # Check if pole is in upright position
#     is_upright = abs(theta) < angle_threshold
#     if is_upright:
#         # When upright, reward for maintaining balance
#         # done = abs(theta) > theta_limit or x > 1.0 or x < -1.0
#         reward = 1.0
#     else:
#         # When not upright, reward for moving towards upright position
#         # Cosine reward peaks at theta = 0 (upright) and is minimum at theta = ±pi (downward)
#         reward = 0.5 * (math.cos(theta) + 1.0)
#         # reward = 0.0

#     new_state = update_state(current_state, action)
#     return new_state, reward, done

def step(current_state, action):
    x, x_dot, theta, theta_dot = current_state

    action = (action - 1) * 10.0  # Maps [0,1,2] to [-10,0,10]

    x = np.clip(x, -2.3, 2.3)
    # Normalize theta to [-pi, pi]
    theta = ((theta + math.pi) % (2 * math.pi)) - math.pi
    
    # Constants for different phases
    upright_threshold = 0.2  # ~11.5 degrees for upright position
    balance_threshold = 0.5  # ~28.6 degrees for near-upright position
    position_limit = 1.5    # Cart position limits

    # Energy-based reward components
    potential_energy = math.cos(theta)  # Highest at upright (theta = 0)
    kinetic_energy = 0.5 * theta_dot**2  # Reward for having some angular velocity

    truncated = abs(x) > position_limit or abs(theta_dot) > 15.0

    # Position penalty to keep cart centered
    position_penalty = -0.5 * (x**2)

    if abs(theta) < upright_threshold:
        # Phase 1: Maintaining balance when nearly upright
        reward = 2.0  # High reward for staying balanced
        reward += position_penalty  # Encourage staying centered
        # done = abs(x) > position_limit

    elif abs(theta) < balance_threshold:
        # Phase 2: Getting very close to upright
        reward = 1.0 + potential_energy
        reward += position_penalty
        # done = abs(x) > position_limit

    else:
        # Phase 3: Swinging up from bottom
        # Reward combination of potential energy (height) and kinetic energy (motion)
        swing_reward = -theta_dot * math.sin(theta)
        reward = 0.5 * (potential_energy + 0.5 * swing_reward)

    new_state = update_state(current_state, action)
    return new_state, reward, truncated

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size))
    
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
        self.batch_size = 256
        self.memory = ReplayBuffer(100000)
        
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
        # dones = torch.FloatTensor(dones)

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
    agent = DQLAgent(4, 3)
    scores = []
    episode = 0
    while True:
        score = 0
        state = initialize_state()
        done = False
        #TODO: DEBUG!!
        while not done:
            action = agent.act(state)
            next_state, reward, truncated = step(state, action)
            agent.memory.push(state, action, reward, next_state)
            # print(done)
            loss = agent.train()
            state = next_state
            score += reward
            done = truncated
        
        if episode % 10 == 0:
            agent.update_target_network()

        scores.append(score)
        avg_score = np.mean(scores[-750:])
        
        print(f'Episode: {episode+1}, Score: {score}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}')
        
        if avg_score >= 50.0:
            print(f'Environment solved in {episode+1} episodes!')
            break
        episode += 1
    return agent

import time

def simulate_agent(agent):
    states = []
    actions = []
    for _ in range(1):
        state = initialize_state()
        done = False
        # start_time = time.time()
        for i in range(500):
            action = agent.act(state)
            states.append(state)
            actions.append(action)
            print(state[2])
            update_state, _, done = step(state, action)
            state = update_state
            # elapsed_time = time.time() - start_time
            # if elapsed_time > 20:
            #     done = True 

    return np.array(states), np.array(actions)

# def simulate_agent():
#     states = []
#     actions = []
#     for _ in range(10):
#         state = initialize_state()
#         done = False
#         # start_time = time.time()
#         for i in range(750):
#             # action = agent.act(state)
#             action = random.choice((0, 1))
#             force = -10.0 if action == 0 else 10.0
#             states.append(state)
#             actions.append(force)
#             update_state, _, _ = step(state, force)
#             state = update_state
#             # elapsed_time = time.time() - start_time
#             # if elapsed_time > 20:
#             #     done = True 

#     return np.array(states), np.array(actions)

agent = train_agent()
states, actions = simulate_agent()

# Visualization
def update(i):
    x, x_dot, theta, theta_dot = states[i]
    # Clamp the cart position to stay within the plot bounds
    x = np.clip(x, -1.5, 1.5)  # Prevent the cart from going too far left or right
    # Update cart position (a rectangle) and pole (a line)
    cart.set_data([x - 0.1, x + 0.1], [0, 0])  # Cart is a 1m wide rectangle
    pole.set_data([x, x + pole_length * np.sin(theta)], [0, pole_length * np.cos(theta)])

    time_text.set_text(f'Time: {i*time_step:.2f}s')
    force_text.set_text(f'Force: {actions[i].item():.2f} N')
    return cart, pole, time_text, force_text

update_tracking_line()  # Call once to initialize the tracking line
ani = FuncAnimation(fig, update, frames=len(states), interval=time_step * 1000, blit=True)
plt.show()
