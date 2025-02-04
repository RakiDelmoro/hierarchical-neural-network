
# QUESTION TO SOLVE:
# How to make a neuron have it's own properties and learn the pattern of input and make a decision wether that neuron will fire or not

# The idea of having each neuron have it's own MLP (not shared) is we want each neuron to learn what's in the input for example:
# If our input is image of a DOG: We have a outer network of 784, 10, 2 (Only 2 class we want to classify wether an image is a dog or a cat)
# Middle neurons 10:
# each neuron has it's own MLP we want each neuron to activate if there's pattern in the input example dog have a longer face compare to cat
# So maybe one of the 10 neurons learn that the dog have longer face and if our input is dog that neuron will fire (like in other neurons we want to learn)
# What's in our data and fire if that is met (same as for cat)

# Neuron have it's own MLP and the readout is shared of all neurons
# The idea of each neurons have it's own MLP is that each neuron will learn differently about what is the pattern of the input
# Readout is shared for all neurons (Should we update the readout connections or not🤔)

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

class InvertedPendulum:
    def __init__(self):
        # System parameters
        self.gravity = 9.81  # m/s^2
        self.cart_mass = 1.0  # kg
        self.pole_mass = 0.1  # kg
        self.pole_length = 1.0   # m
        self.dt = 0.02  # seconds
        
        # State limits
        self.x_limit = 2.4  # m
        self.theta_limit = 12 * math.pi / 180  # radians
        
        # Action force
        self.force_mag = 10.0  # N
        
        # Reset the system
        self.reset()
    
    def reset(self):
        # Initialize state with small random values
        self.x = np.random.uniform(-0.05, 0.05)  # cart position
        self.x_dot = np.random.uniform(-0.05, 0.05)  # cart velocity
        self.theta = np.random.uniform(-0.05, 0.05)  # pole angle
        self.theta_dot = np.random.uniform(-0.05, 0.05)  # pole angular velocity
        return self.get_state()
    
    def get_state(self):
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])
    
    def step(self, action):
        # Convert action (0 or 1) to force
        force = -self.force_mag if action == 0 else self.force_mag
        
        # Extract state variables
        x, x_dot, theta, theta_dot = self.get_state()
        
        # Calculate derivatives using the equations of motion
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        # Compute accelerations using equations of motion
        temp = (force + self.pole_mass * self.pole_length * theta_dot**2 * sintheta) / (self.cart_mass + self.pole_mass)
        theta_acc = (self.gravity * sintheta - costheta * temp) / (self.pole_length * (4.0/3.0 - self.pole_mass * costheta**2 / (self.cart_mass + self.pole_mass)))
        x_acc = temp - self.pole_mass * self.pole_length * theta_acc * costheta / (self.cart_mass + self.pole_mass)
        
        # Update state using Euler integration
        self.x = x + self.dt * x_dot
        self.x_dot = x_dot + self.dt * x_acc
        self.theta = theta + self.dt * theta_dot
        self.theta_dot = theta_dot + self.dt * theta_acc
        
        # Check if state is terminal
        done = bool(
            self.x < -self.x_limit
            or self.x > self.x_limit
            or self.theta < -self.theta_limit
            or self.theta > self.theta_limit)

        # Calculate reward
        if not done: reward = 1.0
        else: reward = 0.0

        return self.get_state(), reward, done

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
        self.batch_size = 64
        self.memory = ReplayBuffer(50000)
        
        # Neural Networks
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
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
    env = InvertedPendulum()
    state_size = 4  # [x, x_dot, theta, theta_dot]
    action_size = 2  # [left, right]
    agent = DQLAgent(state_size, action_size)
    
    episodes = 500
    target_update_frequency = 10
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            
            loss = agent.train()
            state = next_state
            score += reward
        
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        
        print(f'Episode: {episode+1}, Score: {score}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}')
        
        if avg_score >= 50.0:
            print(f'Environment solved in {episode+1} episodes!')
            break

    return agent, scores

def simulate_agent(agent):
    env = InvertedPendulum()
    states = []
    actions = []
    for _ in range(1):
        state = env.reset()
        done = False
        start_time = time.time()
        while not done:
            action = agent.act(state)
            states.append(state)
            actions.append(action)

            update_state, _, done = env.step(action)
            state = update_state
            elapsed_time = time.time() - start_time
            if elapsed_time > 20:
                done = True 

    return np.array(states), np.array(actions)

def visualize_episode(states, actions, episode_length=1000):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-0.5, 1.5))
    ax.grid()

    # Cart elements
    cart_width, cart_height = 0.3, 0.1
    pole_length = 0.5 * 2  # Full length of the pole
    
    cart = plt.Rectangle((0 - cart_width/2, 0 - cart_height/2), cart_width, cart_height, fc='blue', zorder=2)
    pole, = ax.plot([], [], 'r-', lw=2, zorder=1)
    mass_point, = ax.plot([], [], 'ro', markersize=6)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    force_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    # Add a line to track the cart's movement
    tracking_line, = ax.plot([], [], 'g-', lw=1, zorder=3)  # Green tracking line
    
    def init():
        ax.add_patch(cart)
        pole.set_data([], [])
        mass_point.set_data([], [])
        time_text.set_text('')
        force_text.set_text('')
        tracking_line.set_data([], [])  # Initialize the tracking line
        return cart, pole, mass_point, time_text, force_text, tracking_line
    
    def animate(i):
        state = states[i]
        action = actions[i]
        x = state[0]
        theta = state[2]
        
        # Update cart position
        cart.set_xy([x - cart_width/2, 0 - cart_height/2])
        
        # Update pole position
        pole_x = [x, x + pole_length * np.sin(theta)]
        pole_y = [0, pole_length * np.cos(theta)]
        pole.set_data(pole_x, pole_y)
        
        # Update mass point
        mass_point.set_data(pole_x, pole_y)
        
        # Update tracking line (append the current x position of the cart to the line)
        x_vals, y_vals = tracking_line.get_data()
        x_vals = np.append(x_vals, x)
        y_vals = np.append(y_vals, 0)  # The y position of the cart is fixed at 0
        tracking_line.set_data(x_vals, y_vals)
        
        # Update text
        time_text.set_text(f'Time: {i*0.02:.2f}s')
        force_text.set_text(f'Force: {action.item():.2f} N')
        return cart, pole, mass_point, time_text, force_text, tracking_line

    ani = animation.FuncAnimation(fig, animate, frames=min(len(states), episode_length), interval=0.02*1000, blit=True, init_func=init)
    plt.show()


agent, scores = train_agent()
states, actions = simulate_agent(agent)
visualize_episode(states, actions)