import math
import torch
import random
import numpy as np
from collections import deque
from torch.nn.init import kaiming_uniform_

# Simulation parameters
CART_MASS = 1.0
POLE_MASS = 0.1
POLE_LENGTH = 0.5
GRAVITY = 9.81
TIME_STEP = 0.02

def relu(input_data, return_derivative=False):
    if return_derivative:
        return np.where(input_data > 0, 1, 0)
    else:
        return np.maximum(0, input_data)
    
def mse_loss(predicted, expected):
    avg_loss = np.mean((expected - predicted)**2)
    loss_for_backprop = 2 * (expected - predicted)

    return avg_loss, loss_for_backprop

def initialize_layer_connections(input_size, output_size):
    gen_w_matrix = torch.empty(size=(input_size, output_size))
    gen_b_matrix = torch.empty(size=(output_size,))
    weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = torch.nn.init.uniform_(gen_b_matrix, -bound, bound)

    return [np.array(weights), np.array(bias)]

def parameters_init(network_architecture):
    parameters = []
    for size in range(len(network_architecture)-1):
        input_size = network_architecture[size]
        output_size = network_architecture[size+1]
        connections = initialize_layer_connections(input_size, output_size)
        parameters.append(connections)

    return parameters

def initialize_state():
    pole_angle = math.pi
    pole_angular_velocity = 0.0
    cart_position = 0.0
    cart_velocity = 0.0

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

def calculate_reward(current_state):
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

def update_simulation(current_state, action):
    force = (action - 1) * 10.0
    reward, done = calculate_reward(current_state)
    new_state = update_state(current_state, force)
    
    return new_state, reward, done

def push_to_memory(memories_storage, states, actions, reward, next_state, done):
    memories_storage.append((states, actions, reward, next_state, done))

def fetch_from_memory(memories_storage, size):
    return random.sample(memories_storage, size)

def neural_network(parameters):

    def forward(input_state):
        activations = []

        activation = input_state
        for each in range(len(parameters)):
            weights = parameters[each][0]
            bias = parameters[each][1]

            pre_activation = np.matmul(activation, weights) + bias
            
            if each == len(parameters)-1:
                activation = pre_activation
            else:
                activation = relu(pre_activation)

            activations.append(activation)

        return activations

    return forward

def backpropagation(loss, activations, parameters):
    


def ai_agent(brain_architecture, memory_storage):
    # Hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    batch_size = 256

    state_size = 4
    action_size = 3

    parameters = parameters_init(brain_architecture)

    policy_net = neural_network(parameters)
    target_net = neural_network(parameters)

    def act(state):
        if random.random() < epsilon:
            return random.randrange(action_size)
        
        state = get_state(state)
        policy_net_output = policy_net(state)

        return policy_net_output.argmax().item()
    
    def train():
        if len(memory_storage) < batch_size:
            return

        batched_state = fetch_from_memory(memory_storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*batched_state)

        states = np.float32(states)
        actions = np.longlong(actions)
        rewards = np.float32(rewards)
        next_states = np.float32(next_states)
        dones = np.float32(dones)

        policy_net_activations = policy_net(states)
        target_net_activations = target_net(next_states)

        policy_net_action_prob = policy_net_activations[-1][np.arange(batch_size), actions]
        target_net_action_prob = target_net_activations[-1].max(axis=1)

        target_policy_net_output = rewards + (1 - dones) * gamma * target_net_action_prob

        avg_loss, loss_for_backprop = mse_loss(policy_net_action_prob, target_policy_net_output)

    return train, act

def agent_runner():
    agent_memories = deque(maxlen=100000)
    train, act = ai_agent(brain_architecture=[4, 128, 128, 3], memory_storage=agent_memories)
    
    scores = []
    episode = 0
    while True:
        score = 0
        state = initialize_state()
        done = False

        for _ in range(500):
            action = act(state)
            next_state, reward, done = update_simulation(state, action)
            push_to_memory(agent_memories, state, action, reward, next_state, done)

            train()


agent_runner()

