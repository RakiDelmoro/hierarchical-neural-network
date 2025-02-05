import pygame
import math
import numpy as np
import torch.nn as nn
import torch
from scipy.integrate import odeint


# Pygame visualization
FPS = 60
WIDTH, HEIGHT = 750, 750
clock = pygame.time.Clock()
TRACK_LEFT, TRACK_RIGHT = 100, WIDTH - 100
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Inverted Pendulum RL Environment")
COLORS = {'WHITE': (255, 255, 255), 'BLACK': (0, 0, 0), 'RED': (255, 0, 0), 'BLUE': (0, 0, 255), 'GRAY': (128, 128, 128)}
# {'WHITE': (255, 255, 255), 'BLACK': (0, 0, 0), 'RED': (255, 0, 0), 'BLUE': (0, 0, 255), 'GRAY': (128, 128, 128)}

# Inverted pendulum properties
CART_WIDTH, CART_HEIGHT = 50, 10
GRAVITY =  9.81
POLE_MASS = 1.0
CART_MASS = 5.0
POLE_LENGTH = 1.0
CART_FORCE_MAGNITUDE = 250.0
TIME_STEP = 1.0 / FPS

def initialize_state():
    cart_position = WIDTH // 2
    cart_velocity = 0.0
    angle = math.pi
    angular_velocity = 0.0

    return [cart_position, cart_velocity, angle, angular_velocity]

def update_state(state, action):
    """Physics update remains unchanged (correct swing dynamics)"""
    cart_position, cart_velocity, angle, angular_velocity = state

    F = action * CART_FORCE_MAGNITUDE
    sin_theta = math.sin(angle)
    cos_theta = math.cos(angle)

    denominator = CART_MASS + POLE_MASS * sin_theta**2

    # Cart acceleration (x_ddot)
    x_ddot = (F + POLE_MASS * sin_theta * (POLE_MASS * angular_velocity**2 + GRAVITY * cos_theta)) / denominator

    # Angular acceleration (theta_ddot)
    theta_ddot = (-F * cos_theta - 
                POLE_MASS * POLE_MASS * angular_velocity**2 * sin_theta * cos_theta + 
                (CART_MASS + POLE_MASS) * GRAVITY * sin_theta) / (POLE_MASS * denominator)

    cart_velocity += x_ddot * TIME_STEP
    angular_velocity += theta_ddot * TIME_STEP
    # Update positions with track boundary enforcement
    cart_position += cart_velocity * TIME_STEP
    cart_position = np.clip(cart_position, TRACK_LEFT + CART_WIDTH/2,  TRACK_RIGHT - CART_WIDTH/2)
    angle += angular_velocity * TIME_STEP

    return [cart_position, cart_velocity, angle, angular_velocity]

def normalized_state(state):
    position, velocity, angle, angular_velocity = state
    position_normalize = ((position - TRACK_RIGHT) / (TRACK_RIGHT - TRACK_LEFT) * 2) - 1
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    return [position_normalize, direction_x, direction_y, angular_velocity * 0.1]

def reward(state):
        """
        Calculate reward for the inverted pendulum swing-up task.
        The goal is to swing up the pendulum and balance it upright while keeping the cart centered.
        """

        cart_position, cart_velocity, angle, angular_velocity = state

        # Parameters
        angle_weight = 1.0
        position_weight = 0.0
        velocity_weight = 0.3
        
        # Angle reward: maximum (1.0) when upright (angle = 0), minimum (0.0) when downward (angle = pi)
        # Using cosine gives a smooth reward that encourages continuous progress
        angle_reward = 0.5 * (1 + math.cos(angle))  # Normalized between 0 and 1
        
        # Position reward: maximum (1.0) at center, decreases with distance
        track_center = (TRACK_RIGHT + TRACK_LEFT) / 2
        position_deviation = abs(cart_position - track_center)
        max_deviation = (TRACK_RIGHT - TRACK_LEFT) / 2
        position_reward = 1.0 - (position_deviation / max_deviation)
        
        # Velocity penalties: discourage excessive velocities
        velocity_penalty = -(cart_velocity**2 + angular_velocity**2) * 0.01
        
        # Combine rewards
        total_reward = (angle_weight * angle_reward + position_weight * position_reward + velocity_weight * velocity_penalty)
        # total_reward = (angle_weight * angle_reward + velocity_weight * velocity_penalty)
        return total_reward

def visualize_current_state(state):
        cart_position, _, angle, _ = state

        SCREEN.fill(COLORS['WHITE'])
        l = POLE_LENGTH * 100  # Scale for visualization
        cart_y = HEIGHT // 2

        # Draw track
        pygame.draw.line(SCREEN, COLORS['GRAY'], (TRACK_LEFT, cart_y), (TRACK_RIGHT, cart_y), 5)

        # Draw cart
        pygame.draw.rect(SCREEN, COLORS['BLUE'], (cart_position - CART_WIDTH//2, cart_y - CART_HEIGHT//2,CART_WIDTH, CART_HEIGHT))

        # Draw pendulum
        pendulum_end = (cart_position + l * math.sin(angle), cart_y - l * math.cos(angle))
        pygame.draw.line(SCREEN, COLORS['BLACK'], (cart_position, cart_y), pendulum_end, 3)
        pygame.draw.circle(SCREEN, COLORS['RED'], (int(pendulum_end[0]), int(pendulum_end[1])), 10)
        pygame.display.flip()

def runner():
    running = True
    state = initialize_state()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Replace with agent's action selection
        action = np.random.uniform(-1, 1)  # Random policy example
        state = update_state(state, action)
        action_reward = reward(state)
        print(action_reward)
        visualize_current_state(state)
        clock.tick(FPS)

runner()
