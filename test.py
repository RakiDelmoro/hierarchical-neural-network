import math
import torch
import numpy as np
from torch.nn import init
from torch.nn.init import kaiming_uniform_
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# System parameters
CART_MASS = 1.0
POLE_MASS = 0.1
LENGTH = 0.5
GRAVITY = 9.81
TIME_STEP = 0.02

def pid_controller(state, time_step, target_angle=0, Kp=10.0, Ki=0.1, Kd=1.0):
    theta = state[2]
    integral = state[4]  # Added integral term to the state
    error = theta - target_angle
    derivative = state[3]  # theta_dot
    integral += error * time_step
    F = - (Kp * error + Ki * integral + Kd * derivative)
    return F, integral

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
        integral = 0.0

        state = np.array([position, velocity, angle, angular_velocity, integral])
    
        for _ in range(steps_per_episode):
            x, x_dot, theta, theta_dot, integral = state
            F, new_integral = pid_controller(state, TIME_STEP)
            
            # Clip force to realistic values
            F = np.clip(F, -10, 10)
            
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
            state = np.array([x, x_dot, theta, theta_dot, new_integral])
            
            # Terminate if pole falls
            if abs(theta) > np.pi/1 or abs(x) > 3: break

    return np.array(states), np.array(actions)



    def forward(input_data):
        activation = input_data
        activations = [activation]
        for each in range(len(parameters)):
            weights = parameters[each][0]
            bias = parameters[each][1]
            pre_activation = np.matmul(activation, weights) + bias
            activation = np.tanh(pre_activation)
            activations.append(activation)
        return activations

    return forward

def visualize_episode(states, actions,  episode_length=1000):
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
        force_text.set_text(f'Force: {action:.2f} N')
        
        return cart, pole, mass_point, time_text, force_text
    
    ani = animation.FuncAnimation(
        fig, animate, frames=min(len(states), episode_length),
        interval=TIME_STEP*1000, blit=True, init_func=init
    )

    plt.show()

# Pendulum simulations
states, actions = pendulum_simulation()

# Create animation
visualize_episode(states, actions)

