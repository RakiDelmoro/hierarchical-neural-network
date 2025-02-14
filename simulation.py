import pygame
import math
import numpy as np
import torch
import torch.nn as nn

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
CART_WIDTH = 80
CART_HEIGHT = 40
POLE_LENGTH = 200  # Pixels
SCALE = 400  # Conversion from meters to pixels

# Control circle constants
CONTROL_CIRCLE_RADIUS = 15
CONTROL_ZONE_RADIUS = 50  # Area where mouse movement is detected
MAX_DISTANCE = CONTROL_ZONE_RADIUS  # Maximum distance for force calculation

# Boundary settings
BOUNDARY_MARGIN = 100  # Pixels from screen edge
MAX_CART_TRAVEL = (SCREEN_WIDTH - 2 * BOUNDARY_MARGIN) / SCALE  # Convert to meters

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (150, 150, 150)
BOUNDARY_COLOR = (255, 150, 150)

# System parameters
cart_mass = 1.0
pole_mass = 0.1
pole_length_m = 0.5
gravity = 9.81
time_step = 0.02

# Force parameters
FORCE_MAGNITUDE = 10.0  # Maximum force applied

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size))

    def forward(self, x):
        return self.network(x)

class CartPoleSimulator:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.state = [0.0, 0.0, math.pi, 0.0]  # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        self.force = 0.0  # Current force being applied
        
        # Control circle center position
        self.control_center_x = SCREEN_WIDTH // 2
        self.control_center_y = SCREEN_HEIGHT - 50  # 50 pixels from bottom

        self.model = NeuralNetwork(4, 3)
        self.model.load_state_dict(torch.load('model.pth', weights_only=True))

    def calculate_mouse_force(self, mouse_pos):
        mouse_x, _ = mouse_pos

        # Calculate distance from control circle center to mouse
        dx = mouse_x - self.control_center_x
        distance = math.sqrt(dx * dx)

        if distance <= CONTROL_CIRCLE_RADIUS:
            return 0.0  # No force if mouse is inside control circle

        # Normalize distance to maximum force
        force = (dx / MAX_DISTANCE) * FORCE_MAGNITUDE
        # Clamp force to maximum magnitude
        return np.clip(force, -FORCE_MAGNITUDE, FORCE_MAGNITUDE)

    def handle_input(self):
        keys = pygame.key.get_pressed()

        state_tensor = torch.tensor(self.state, dtype=torch.float32)
        action_probabilities = self.model(state_tensor)
        action = torch.argmax(action_probabilities).item()
        self.force = (action - 1) * FORCE_MAGNITUDE

        if keys[pygame.K_LEFT]:
            self.force = -FORCE_MAGNITUDE
        elif keys[pygame.K_RIGHT]:
            self.force = FORCE_MAGNITUDE
        # else:
        #     # Get mouse position and calculate force
        #     mouse_pos = pygame.mouse.get_pos()
        #     self.force = self.calculate_mouse_force(mouse_pos)

        return self.force

    def update_physics(self):
        # Get force from input
        force = self.handle_input()

        # Current state
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = self.state

        # Physics calculations
        cos_theta = math.cos(pole_angle)
        sin_theta = math.sin(pole_angle)

        temp = (force + pole_mass * pole_length_m * pole_angular_velocity**2 * sin_theta) / (cart_mass + pole_mass)
        theta_acc = (gravity * sin_theta - cos_theta * temp) / (pole_length_m * (4.0/3.0 - pole_mass * cos_theta**2 / (cart_mass + pole_mass)))
        x_acc = temp - pole_mass * pole_length_m * theta_acc * cos_theta / (cart_mass + pole_mass)

        # Update state
        cart_velocity += x_acc * time_step
        cart_position += cart_velocity * time_step
        pole_angular_velocity += theta_acc * time_step
        pole_angle += pole_angular_velocity * time_step

        # Limit cart position
        cart_position = np.clip(cart_position, -MAX_CART_TRAVEL/2, MAX_CART_TRAVEL/2)
        if abs(cart_position) >= MAX_CART_TRAVEL/2:
            cart_velocity = 0

        self.state = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        return force

    def draw(self, force):
        self.screen.fill(WHITE)
  
        # Convert state to screen coordinates
        cart_x = SCREEN_WIDTH/2 + self.state[0] * SCALE
        cart_y = SCREEN_HEIGHT/2

        # Draw cart and pole
        pygame.draw.rect(self.screen, BLACK, 
                        [cart_x - CART_WIDTH/2, cart_y - CART_HEIGHT/2, 
                         CART_WIDTH, CART_HEIGHT])

        pole_end_x = cart_x + math.sin(self.state[2]) * POLE_LENGTH
        pole_end_y = cart_y - math.cos(self.state[2]) * POLE_LENGTH
        pygame.draw.line(self.screen, RED, (cart_x, cart_y), 
                        (pole_end_x, pole_end_y), 6)

        # Draw ground line
        pygame.draw.line(self.screen, BLACK, (0, SCREEN_HEIGHT/2 + CART_HEIGHT/2),
                        (SCREEN_WIDTH, SCREEN_HEIGHT/2 + CART_HEIGHT/2), 2)

        # Draw control circle and zone
        pygame.draw.circle(self.screen, GRAY, (self.control_center_x, self.control_center_y), 
                         CONTROL_ZONE_RADIUS, 1)  # Control zone
        pygame.draw.circle(self.screen, BLUE, (self.control_center_x, self.control_center_y), 
                         CONTROL_CIRCLE_RADIUS)  # Control circle

        # Draw mouse line when force is being applied
        # if abs(force) > 0:
        #     mouse_pos = pygame.mouse.get_pos()
        #     pygame.draw.line(self.screen, RED, 
        #                    (self.control_center_x, self.control_center_y), 
        #                    mouse_pos, 2)

        # Draw info text
        angle_text = self.font.render(f'Angle: {math.degrees(self.state[2]):.1f}°', True, BLUE)
        position_text = self.font.render(f'Position: {self.state[0]:.2f}m', True, BLUE)
        force_text = self.font.render(f'Force: {force:.1f}N', True, BLUE)

        self.screen.blit(angle_text, (10, 10))
        self.screen.blit(position_text, (10, 50))
        self.screen.blit(force_text, (10, 90))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            force = self.update_physics()
            self.draw(force)
            self.clock.tick(60)

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Mouse-Controlled Cart Pole")

    simulator = CartPoleSimulator(screen)
    simulator.run()

    pygame.quit()

if __name__ == "__main__":
    main()