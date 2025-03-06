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
    
def mse_loss(predicted, expected, actions):
    avg_loss = np.mean((predicted - expected)**2)
    loss_per_action = 2 * (predicted - expected)

    loss_for_backprop = np.zeros(shape=(expected.shape[0], 3))
    loss_for_backprop[np.arange(expected.shape[0]), actions] = loss_per_action

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


def forward(input_state, parameters):
    activations = [input_state]

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

def calculate_activation_loss(activations, loss, parameters):
    activation_losses = [loss]
    for each in range(len(parameters)-1):
        weights = parameters[-(each+1)][0].T

        propagate_previous_loss = np.matmul(loss, weights)
        loss = relu(activations[-(each+2)], return_derivative=True) * propagate_previous_loss
        activation_losses.append(loss)

    return activation_losses

def backpropagation(activation_losses, activations, parameters):
    new_parameters = []
    # Left to Right
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]
        bias = parameters[-(each+1)][1]

        loss_for_update = activation_losses[each]
        activation = activations[-(each+2)]
        
        grad_weights = np.matmul(activation.T, loss_for_update) / loss_for_update.shape[0]
        grad_bias = np.sum(loss_for_update, axis=0) / loss_for_update.shape[0]

        new_weights = weights - (0.001 * grad_weights)
        new_bias = bias - (0.001 * grad_bias)

        new_parameters.insert(0, [new_weights, new_bias])

    return new_parameters

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))   

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def ai_agent(brain_architecture, memory_storage, policy_net_parameters, target_net_parameters):
    # Hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    batch_size = 256

    state_size = 4
    action_size = 3

    def act(state):
        if random.random() < epsilon:
            return random.randrange(action_size)
        
        state = np.float32(get_state(state))
        policy_net_output = forward(state, policy_net_parameters)

        return policy_net_output[-1].argmax().item()
    
    def train():
        nonlocal policy_net_parameters
        nonlocal epsilon

        if len(memory_storage) < batch_size:
            return policy_net_parameters, epsilon

        batched_state = fetch_from_memory(memory_storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*batched_state)

        states = np.float32(states)
        actions = np.longlong(actions)
        rewards = np.float32(rewards)
        next_states = np.float32(next_states)
        dones = np.float32(dones)

        policy_net_activations = forward(states, policy_net_parameters)
        target_net_activations = forward(next_states, target_net_parameters)

        policy_net_action_prob = policy_net_activations[-1][np.arange(batch_size), actions]
        target_net_action_prob = target_net_activations[-1].max(axis=1)

        target_policy_net_output = rewards + (1 - dones) * gamma * target_net_action_prob

        avg_loss, loss_for_backprop = mse_loss(policy_net_action_prob, target_policy_net_output, actions)
        activations_gradients = calculate_activation_loss(policy_net_activations, loss_for_backprop, policy_net_parameters)

        policy_net_parameters = backpropagation(activations_gradients, policy_net_activations, policy_net_parameters)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        return policy_net_parameters, epsilon

    return train, act

def agent_runner():
    policy_net_parameters = parameters_init([4, 128, 128, 3])
    target_net_parameters = policy_net_parameters

    agent_memories = deque(maxlen=100000)
    train, act = ai_agent([4, 128, 128, 3], agent_memories, policy_net_parameters, target_net_parameters)

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

            policy_net_parameters, epsilon = train()
            state = next_state
            score += reward

        if episode % 10 == 0:
            print('Change parameters')
            target_net_parameters = policy_net_parameters

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if avg_score > 500:
            print(f'Environment solved in {episode} episodes!')
            break

        print(f'Episode: {episode+1}, Score: {score}, Average Score: {avg_score:.2f} Epsilon: {epsilon}')
        episode += 1

agent_runner()

