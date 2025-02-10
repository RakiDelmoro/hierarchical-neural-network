import random
import numpy as np
from collections import deque
from PendulumModel.utils import parameters_init


def relu(input_data, return_derivative=False):
    if return_derivative:
        return np.where(input_data > 0, 1, 0)
    else:
        return np.maximum(0, input_data)

def memory(capacity):
    buffer = deque(maxlen=capacity)
    
    def put(state, action, reward, next_state, done):
        buffer.append((state, action, reward, next_state, done))

    def fetch(batch_size):
        return random.sample(buffer, batch_size)

    return buffer, put, fetch

def neural_network(parameters):
    
    def forward(input_data):
        activation = input_data
        activations = [activation]
        for each in range(len(parameters)):
            last_layer = len(parameters)-2
            weights = parameters[each][0]
            bias = parameters[each][1]
            pre_activation = np.matmul(activation, weights) + bias
            activation = pre_activation if last_layer else relu(pre_activation)
            activations.append(activation)
        return activations

    return forward

def agent(action_size, policy_parameters, target_parameters):
    # Agent hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    batch_size = 256

    # Neural Networks
    policy_net = neural_network(policy_parameters)
    target_net = neural_network(target_parameters)

    def act(state):
        # If true agent can explore random actions
        if random.random() < epsilon:
            return random.randrange(action_size)

        q_values = policy_net(state)[-1]
        return q_values.argmax().item()

    def train(batched_data):
        nonlocal epsilon
        states, actions, rewards, next_states, dones = zip(*batched_data)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.longlong)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Get the network probabilities for a given action
        policy_net_activations = policy_net(states)
        actions_probabilities = policy_net_activations[-1]

        current_q_values = actions_probabilities[np.arange(len(actions)), actions]

        next_q_values = target_net(next_states)[-1].max(1)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

        loss, loss_propagated = calculate_loss(current_q_values, target_q_values, actions)
        backpropagation(policy_net_activations, loss_propagated)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        return loss.item()

    def backpropagation(activations, loss_propagated):
        for each in range(len(activations)-1):
            # Parameters
            weights = policy_parameters[-(each+1)][0]
            bias = policy_parameters[-(each+1)][1]
            # Model activations and loss
            previous_activation = activations[-(each+2)]
            activation = activations[-(each+1)]
            # Loss to use for updating the parameters
            if each == 0:
                loss_propagated = loss_propagated
            else:
                loss_propagated = loss_propagated * relu(activation, return_derivative=True)
            # Propagate the loss to the next layer
            propagated_loss = np.matmul(loss_propagated, weights.transpose())
            # Update parameters
            weights -= 0.001 * np.matmul(previous_activation.transpose(), loss_propagated) / activation.shape[0]
            bias -= 0.001 * np.sum(loss_propagated, axis=0) / activation.shape[0]
            # Update new loss
            loss_propagated = propagated_loss

    def calculate_loss(policy_net_output, target_net_output, actions):
        loss_propagated = np.zeros((policy_net_output.shape[0], action_size))
        # Mean Squared Error
        mse_loss = np.mean((policy_net_output - target_net_output)**2)
        # Loss use for backpropagation
        action_loss = 2 * (policy_net_output - target_net_output)

        loss_propagated[np.arange(len(actions)), actions] = action_loss

        return mse_loss, loss_propagated
    
    return act, train

# Neural network
network_parameters = parameters_init(network_architecture=[4, 128, 128, 3])

def training_agent():
    # Parameters
    policy_parameters = network_parameters
    target_parameters = network_parameters

    # Neural network functions
    act_agent, train_agent = agent(action_size=3, policy_parameters=policy_parameters, target_parameters=target_parameters)
    previous_memories, push, get = memory(capacity=100000)

    scores = []
    for episode in range(1000):
        score = 0
        state = get_state(initialize_state())

        for _ in range(1000):
            action = act_agent(state)
            next_state, reward, done = step(state, action)
            push(state, action, reward, next_state, done)

            # Collect memories if the previous memories is less than batch size (For training in a batch)
            if len(previous_memories) < 256:continue

            # Fetch data in memory
            batched_data = get(batch_size=256)

            loss = train_agent(batched_data)
            state = get_state(next_state)
            score += reward

        scores.append(score)
        avg_score = np.mean(scores[-1000:])

        if episode % 10 == 0:
            target_parameters = policy_parameters

        print(f'Episode: {episode+1}, Score: {score}, Average Score: {avg_score:.2f}')

    return policy_parameters