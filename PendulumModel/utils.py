import math
import torch
import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.nn.init import kaiming_uniform_
from PendulumModel.constants import TIME_STEP, POLE_MASS, LENGTH, GRAVITY, CART_MASS, ACTION_MEAN, ACTION_STD

def batch_simulation(states, actions, batch_size, shuffle=True):
    num_samples = states.shape[0]
    samples_indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(samples_indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        normalize_states, normalize_actions = normalize_state(states[samples_indices[start_idx:end_idx]]), normalize_action(actions[samples_indices[start_idx:end_idx]])
        yield normalize_states, normalize_actions

def normalize_state(data):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    return 2 * (data - data_min) / (data_max - data_min) - 1

def normalize_action(actions):
    """Normalize actions to approximately [-1, 1] range"""
    return (actions - ACTION_MEAN) / ACTION_STD

def denormalize_actions(normalized_actions):
    """Convert normalized actions back to original scale"""
    return normalized_actions * ACTION_STD + ACTION_MEAN

def relu(input_data, return_derivative=False):
    if return_derivative:
        return np.where(input_data > 0, 1, 0)
    else:
        return np.maximum(0, input_data)

def tanh(input_data, return_derivative=False):
    if return_derivative:
        input_data = (np.exp(input_data) - np.exp(-input_data))/(np.exp(input_data) + np.exp(-input_data))
        return 1 - input_data * input_data
    else:
        return (np.exp(input_data) - np.exp(-input_data))/(np.exp(input_data) + np.exp(-input_data))
    
def train_neural_network(model_mode, parameters):
    def calculate_loss(prediction, expected):
        expected = expected.reshape(-1, 1)
        # Mean Squared Error
        mse_loss = np.mean((prediction - expected)**2)
        # Loss use for backpropagation
        neuron_loss = 2 * (prediction - expected)

        return mse_loss, neuron_loss

    def update_parameters(model_activations, loss):
        for each in range(len(model_activations)-1):
            # Parameters
            weights = parameters[-(each+1)][0]
            bias = parameters[-(each+1)][1]
            # Model activations and loss
            previous_activation = model_activations[-(each+2)]
            activation = model_activations[-(each+1)]
            # Loss to use for updating the parameters
            loss = loss * tanh(activation, return_derivative=True)
            # Propagate the loss to the next layer
            propagated_loss = np.matmul(loss, weights.transpose())
            # Update parameters
            weights -= 0.1 * np.matmul(previous_activation.transpose(), loss) / activation.shape[0]
            bias -= 0.1 * np.sum(loss, axis=0) / activation.shape[0]
            # Update new loss
            loss = propagated_loss

    def runner(dataloader):
        per_batch_loss = []
        for input_states, expected_actions in dataloader:   
            model_activations = model_mode(input_states)
            mse_loss, loss_for_backprop = calculate_loss(model_activations[-1], expected_actions)
            update_parameters(model_activations, loss_for_backprop)
            per_batch_loss.append(mse_loss)
        return np.mean(np.array(per_batch_loss))
    return runner

def test_neural_network(model_mode, states):
    predicted_states = []
    for each in range(len(states)):
        predicted_state = model_mode(states[each])
        predicted_states.append(predicted_state)
    return np.concatenate(predicted_states, axis=0)


def visualize_episode(states, actions, episode_length=1000):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-0.5, 1.5))
    ax.grid()
    
    # Cart elements
    cart_width, cart_height = 0.3, 0.1
    pole_length =  LENGTH * 2  # Full length of the pole
    
    cart = plt.Rectangle((0 - cart_width/2, 0 - cart_height/2), cart_width, cart_height,  fc='blue', zorder=2)
    pole, = ax.plot([], [], 'r-', lw=2, zorder=1)
    mass_point, = ax.plot([], [], 'ro', markersize=6)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    force_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    def init():
        ax.add_patch(cart)
        pole.set_data([], [])
        mass_point.set_data([], [])
        time_text.set_text('')
        force_text.set_text('')
        return cart, pole, mass_point, time_text, force_text
    
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
        # Update text
        time_text.set_text(f'Time: {i*TIME_STEP:.2f}s')
        force_text.set_text(f'Force: {action.item():.2f} N')
        return cart, pole, mass_point, time_text, force_text

    ani = animation.FuncAnimation(fig, animate, frames=min(len(states), episode_length), interval=TIME_STEP*1000, blit=True, init_func=init)
    plt.show()

def pid_controller(state, time_step, target_angle=0, Kp=1.0, Ki=0.1, Kd=1.0):
    theta = state[2]
    error = theta - target_angle
    derivative = state[3]  # theta_dot
    # integral += error * time_step
    F = - (Kp * error + Ki + Kd * derivative)
    return F

def cart_pole_dynamics(state, F, cart_mass, pole_mass, gravity, length):
    _, _, angle, angular_velocity = state
    sin_theta = np.sin(angle)
    cos_theta = np.cos(angle)
    denominator = cart_mass + pole_mass * sin_theta**2
    theta_ddot = (gravity * sin_theta - cos_theta * (F + pole_mass * length * angular_velocity**2 * sin_theta) / denominator) \
                 / (length * (4/3 - pole_mass * cos_theta**2 / denominator))
    x_ddot = (F + pole_mass * length * (angular_velocity**2 * sin_theta - theta_ddot * cos_theta)) / denominator
    return x_ddot, theta_ddot

def pendulum_simulation(episodes=1000, steps_per_episode=500):
    states = []
    actions = []
    for _ in range(episodes):
        # Random initial state near equilibrium
        position = np.random.uniform(-1, 1)
        velocity = np.random.uniform(-2, 2)
        angle = np.random.uniform(-0.2, 0.2)
        angular_velocity = np.random.uniform(-0.5, 0.5)
        # Initial state
        state = np.array([position, velocity, angle, angular_velocity])
        for _ in range(steps_per_episode):
            x, x_dot, theta, theta_dot = state
            F = np.random.uniform(-10, 10)
            # Record state (excluding integral) and action
            states.append([x, x_dot, theta, theta_dot])
            actions.append(F)
            # Update dynamics
            x_ddot, theta_ddot = cart_pole_dynamics(state[:4], F, CART_MASS, POLE_MASS, GRAVITY, LENGTH)
            x_dot += x_ddot * TIME_STEP
            x += x_dot * TIME_STEP
            theta_dot += theta_ddot * TIME_STEP
            theta += theta_dot * TIME_STEP
            # Update state
            state = np.array([x, x_dot, theta, theta_dot])
            # Terminate if pole falls
            if abs(theta) > np.pi/4 or abs(x) > 2.4: break

    return np.array(states), np.array(actions)

def neural_network_pendulum_simulation(model_mode):
    states = []
    actions = []
    for _ in range(10):
        position = np.random.uniform(-1, 1)
        velocity = np.random.uniform(-2, 2)
        angle = np.random.uniform(-0.2, 0.2)
        angular_velocity = np.random.uniform(-0.5, 0.5)
        # Initial state
        state = np.array([position, velocity, angle, angular_velocity])
        while True:
            x, x_dot, theta, theta_dot = state
            # Model prediction F
            F = denormalize_actions(model_mode(normalize_state(state)))
            # Record state (excluding integral) and action
            states.append([x, x_dot, theta, theta_dot])
            actions.append(F)
            # Update dynamics
            x_ddot, theta_ddot = cart_pole_dynamics(state[:4], F, CART_MASS, POLE_MASS, GRAVITY, LENGTH)
            x_dot += x_ddot * TIME_STEP
            x += x_dot * TIME_STEP
            theta_dot += theta_ddot * TIME_STEP
            theta += theta_dot * TIME_STEP
            # Update state
            state = np.array([x.item(), x_dot.item(), theta.item(), theta_dot.item()])
            # Terminate if pole falls
            if abs(theta) > np.pi/1 or abs(x) > 3: break

    return np.array(states), np.array(actions)

def initializer(input_size, output_size):
    gen_w_matrix = torch.empty(size=(input_size, output_size))
    gen_b_matrix = torch.empty(size=(output_size,))
    weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = init.uniform_(gen_b_matrix, -bound, bound)
    return [np.array(weights), np.array(bias)]

def parameters_init(network_architecture: list):
    parameters = []
    for size in range(len(network_architecture)-1):
        input_size = network_architecture[size]
        output_size = network_architecture[size+1]
        connections = initializer(input_size, output_size)
        parameters.append(connections)
    return parameters

