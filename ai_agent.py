import math
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
from PendulumModel.utils import parameters_init

# System parameters
CART_MASS = 1.0   # Cart mass (kg)
POLE_MASS = 0.1   # Pole mass (kg)
POLE_LENGTH = 0.5   # Pole length (m)
GRAVITY = 9.81  # Gravity (m/s²)
TIME_STEP = 0.02 # Time step (s)

# Set up the figure and axis for the animation
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)  # Limit x-axis from -3m to 3m
ax.set_ylim(-1, 1)  # Limit y-axis from -1m to 1m
ax.set_aspect('equal')

# Create the cart (a simple rectangle) and pole (a line)
cart, = ax.plot([], [], 'k-', lw=6)  # Cart rectangle (black)
pole, = ax.plot([], [], 'r-', lw=3)  # Pole line (red)
tracking_line, = ax.plot([], [], 'b--', lw=1)  # Tracking line (blue dashed)
time_text = ax.text(0.05, 0.90, '', transform=ax.transAxes)
force_text = ax.text(0.05, 0.80, '', transform=ax.transAxes)
force_text.set_text('')

def initialize_state():
    # Initialize the state
    pole_angle = math.pi  # Angle (rad) with small random perturbation
    pole_angular_velocity = 0.0  # Angular velocity (rad/s)
    cart_position = 0.0       # Cart position (m)
    cart_velocity = 0.0      # Cart velocity (m/s)
    return [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

def update_state(state, force):
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
    
    cos_theta = math.cos(pole_angle)
    sin_theta = math.sin(pole_angle)

    # Compute accelerations using equations of motion
    temp = (force + POLE_MASS * POLE_LENGTH * pole_angular_velocity**2 * sin_theta) / (CART_MASS + POLE_MASS)
    theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (POLE_LENGTH * (4.0/3.0 - POLE_MASS * cos_theta**2 / (CART_MASS + POLE_MASS)))
    x_acc = temp - POLE_MASS * POLE_LENGTH * theta_acc * cos_theta / (CART_MASS + POLE_MASS)

    # Euler integration
    cart_velocity += x_acc * TIME_STEP
    cart_position += cart_velocity * TIME_STEP
    pole_angular_velocity += theta_acc * TIME_STEP
    pole_angle += pole_angular_velocity * TIME_STEP

    cart_position = np.clip(cart_position, -1.5, 1.5)

    return [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

def get_state(state):
    return np.array(state)

# Function to update the tracking line
def update_tracking_line():
    # Define a tracking path for the cart (in this case, a straight line)
    # You can modify this to create other types of paths (e.g., sine, parabola, etc.)
    x_values = np.linspace(-2.3, 2.3, 100)
    y_values = np.zeros_like(x_values)  # Path of the cart stays at y=0
    tracking_line.set_data(x_values, y_values)

def step(current_state, action):
    force = (action - 1) * 10.0  # Maps [0,1,2] to [-10,0,10]
    reward, done = reward_function(current_state)
    new_state = update_state(current_state, force)
    return new_state, reward, done

def reward_function(current_state):
    cart_position, _, pole_angle, pole_angular_velocity = current_state
    angle_threshold = 0.2  # ~28.6 degrees for upright position
    position_limit = 1.5    # Cart position limits
    position_penalty = -0.5 * (cart_position**2)

    # Normalize theta to [-pi, pi]
    pole_angle = ((pole_angle + math.pi) % (2 * math.pi)) - math.pi
    done = abs(cart_position) > position_limit or abs(pole_angular_velocity) > 15.0
    reward = 0.0

    # Check if pole is in upright position
    is_upright = abs(pole_angle) < angle_threshold
    if is_upright:
        position_reward = 1.0 - (cart_position / position_limit) ** 2 
        reward = 1.0 + position_reward
    else:        
        # When not upright, reward for moving towards upright position
        # Cosine reward peaks at theta = 0 (upright) and is minimum at theta = ±pi (downward)
        reward = 0.5 * (math.cos(pole_angle) + 1.0)

    reward = reward + position_penalty
    return reward, done

class DQN(nn.Module):
 
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size))

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))   

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

        self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        # If true agent can explore random actions
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
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

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

        for _ in range(500):
            action = agent.act(state)
            next_state, reward, done = step(state, action)
            agent.memory.push(state, action, reward, next_state, done)

            loss = agent.train()
            state = next_state
            score += reward

        if episode % 10 == 0:
            print('Change parameters')
            agent.update_target_network()

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if avg_score > 500:
            print(f'Environment solved in {episode} episodes!')
            break

        print(f'Episode: {episode+1}, Score: {score}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}')
        episode += 1

    # Save model weights
    # torch.save(agent.policy_net.state_dict(), 'model.pth')
    
    return agent

def simulate_agent(agent):
    states = []
    actions = []
    for _ in range(10):
        state = initialize_state()
        for i in range(1000):
            action = agent.act(state)
            states.append(state)
            actions.append(action)
            print(state[2])
            update_state, _, _ = step(state, action)
            state = update_state

    return np.array(states), np.array(actions)

agent = train_agent()
# states, actions = simulate_agent(agent)

# # Visualization
# def update(i):
#     x, _, theta, _ = states[i]
#     # Clamp the cart position to stay within the plot bounds
#     x = np.clip(x, -1.5, 1.5)  # Prevent the cart from going too far left or right
#     # Update cart position (a rectangle) and pole (a line)
#     cart.set_data([x - 0.1, x + 0.1], [0, 0])  # Cart is a 1m wide rectangle
#     pole.set_data([x, x + POLE_LENGTH * np.sin(theta)], [0, POLE_LENGTH * np.cos(theta)])

#     time_text.set_text(f'Time: {i*TIME_STEP:.2f}s')
#     force_text.set_text(f'Force: {actions[i].item():.2f} N')
#     return cart, pole, time_text, force_text

# update_tracking_line()  # Call once to initialize the tracking line
# ani = FuncAnimation(fig, update, frames=len(states), interval=TIME_STEP * 1000, blit=True)
# plt.show()
