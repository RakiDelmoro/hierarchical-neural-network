import pygame
import math
import numpy as np
import torch.nn as nn
import torch

class InvertedPendulum:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 750, 750
        self.FPS = 60
        self.COLORS = {'WHITE': (255, 255, 255), 'BLACK': (0, 0, 0), 'RED': (255, 0, 0), 'BLUE': (0, 0, 255), 'GRAY': (128, 128, 128)}

        self.CART_WIDTH, self.CART_HEIGHT = 50, 10
        self.TRACK_LEFT, self.TRACK_RIGHT = 100, self.WIDTH - 100
        self.gravity, self.pole_mass, self.cart_mass, self.pole_length = 9.81, 0.1, 1.0, 1.0
        self.dt = 1.0 / self.FPS
        self.cart_force = 10.0
        self.reset()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Inverted Pendulum RL Environment")
        self.clock = pygame.time.Clock()

    def reset(self):
        """Reset environment to initial state with pole pointing downward"""
        self.angle = math.pi # Start pointing straight down
        self.angular_velocity = 0.0  # No initial angular velocity
        self.cart_position = self.WIDTH // 2  # Start at track center
        self.cart_velocity = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Normalized state: [cos(theta), sin(theta), angular_velocity, cart_position]"""
        cart_position_normalized = ((self.cart_position - self.TRACK_LEFT) / (self.TRACK_RIGHT - self.TRACK_LEFT) * 2) - 1

        direction_x = math.cos(self.angle)
        direction_y = math.sin(self.angle)

        return [direction_x, direction_y, self.angular_velocity * 0.1, cart_position_normalized]          

    # def _calculate_reward(self):
        # return abs(math.pi - self.angle) / (2 * math.pi)

    def _calculate_reward(self):
        """
        Calculate reward for the inverted pendulum swing-up task.
        The goal is to swing up the pendulum and balance it upright while keeping the cart centered.
        
        State variables used:
        - self.angle: pendulum angle (pi is pointing down, 0 is pointing up)
        - self.angular_velocity: pendulum angular velocity
        - self.cart_position: cart position on track
        - self.cart_velocity: cart velocity
        
        Returns:
        float: Calculated reward
        """
        # Parameters
        angle_weight = 1.0
        position_weight = 0.5
        velocity_weight = 0.3
        
        # Angle reward: maximum (1.0) when upright (angle = 0), minimum (0.0) when downward (angle = pi)
        # Using cosine gives a smooth reward that encourages continuous progress
        angle_reward = 0.5 * (1 + math.cos(self.angle))  # Normalized between 0 and 1
        
        # Position reward: maximum (1.0) at center, decreases with distance
        track_center = (self.TRACK_RIGHT + self.TRACK_LEFT) / 2
        position_deviation = abs(self.cart_position - track_center)
        max_deviation = (self.TRACK_RIGHT - self.TRACK_LEFT) / 2
        position_reward = 1.0 - (position_deviation / max_deviation)
        
        # Velocity penalties: discourage excessive velocities
        velocity_penalty = -(self.cart_velocity**2 + self.angular_velocity**2) * 0.01
        
        # Combine rewards
        total_reward = (angle_weight * angle_reward +
                    position_weight * position_reward +
                    velocity_weight * velocity_penalty)
        
        return total_reward

    def update_physics(self, action):
        """Physics update remains unchanged (correct swing dynamics)"""
        F = action * self.cart_force
        sin_theta = math.sin(self.angle)
        cos_theta = math.cos(self.angle)

        denominator = self.cart_mass + self.pole_mass * sin_theta**2

        # Cart acceleration (x_ddot)
        x_ddot = (F + self.pole_mass * sin_theta * (self.pole_length * self.angular_velocity**2 + self.gravity * cos_theta)) / denominator

        # Angular acceleration (theta_ddot)
        theta_ddot = (-F * cos_theta - 
                     self.pole_mass * self.pole_length * self.angular_velocity**2 * sin_theta * cos_theta + 
                     (self.cart_mass + self.pole_mass) * self.gravity * sin_theta) / (self.pole_length * denominator)

        self.cart_velocity += x_ddot * self.dt
        self.angular_velocity += theta_ddot * self.dt

        # Update positions with track boundary enforcement
        self.cart_position += self.cart_velocity * self.dt
        self.cart_position = np.clip(self.cart_position, self.TRACK_LEFT + self.CART_WIDTH/2, 
                        self.TRACK_RIGHT - self.CART_WIDTH/2)
        self.angle += self.angular_velocity * self.dt

    def step(self, action):
        """Perform one timestep with given action"""
        self.update_physics(action)
        reward = self._calculate_reward()
        return self._get_state(), reward, {}

    def render(self):
        """Visualize current state"""
        self.screen.fill(self.COLORS['WHITE'])
        l = self.pole_length * 100  # Scale for visualization
        cart_y = self.HEIGHT // 2

        # Draw track
        pygame.draw.line(self.screen, self.COLORS['GRAY'],
                        (self.TRACK_LEFT, cart_y),
                        (self.TRACK_RIGHT, cart_y), 5)

        # Draw cart
        pygame.draw.rect(self.screen, self.COLORS['BLUE'],
                        (self.cart_position - self.CART_WIDTH//2, cart_y - self.CART_HEIGHT//2,
                         self.CART_WIDTH, self.CART_HEIGHT))

        # Draw pendulum
        pendulum_end = (self.cart_position + l * math.sin(self.angle),
                       cart_y - l * math.cos(self.angle))
        pygame.draw.line(self.screen, self.COLORS['BLACK'],
                        (self.cart_position, cart_y), pendulum_end, 3)
        pygame.draw.circle(self.screen, self.COLORS['RED'],
                          (int(pendulum_end[0]), int(pendulum_end[1])), 10)

        pygame.display.flip()

    def run(self, agent=None):
        """Main loop for visualization (optional)"""
        running = True
        state = self.reset()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Replace with agent's action selection
            action = np.random.uniform(-1, 1)  # Random policy example
            state, reward, _ = self.step(action)
            print(reward)
            self.render()
            self.clock.tick(self.FPS)

        pygame.quit()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

class SwingUpStabilizer(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.network(state)

def visualization_runner():
    env = InvertedPendulum()
    env.run()

visualization_runner()
