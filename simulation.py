import pygame
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
CART_WIDTH = 80
CART_HEIGHT = 40
POLE_LENGTH = 200  # Pixels
SCALE = 400  # Conversion from meters to pixels

# Boundary settings
BOUNDARY_MARGIN = 100  # Pixels from screen edge
MAX_CART_TRAVEL = (SCREEN_WIDTH - 2 * BOUNDARY_MARGIN) / SCALE  # Convert to meters

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BOUNDARY_COLOR = (255, 150, 150)

# System parameters
cart_mass = 5.0
pole_mass = 0.01
pole_length_m = 0.5
gravity = 9.81
time_step = 0.02

# Keyboard control parameters
FORCE_MAGNITUDE = 10.0  # Force applied when key is pressed

class CartPoleSimulator:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.state = [0.0, 0.0, math.pi, 0.0]  # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        self.force = 0.0  # Current force being applied

    def handle_input(self):
        keys = pygame.key.get_pressed()
        self.force = 0.0
        if keys[pygame.K_LEFT]:
            self.force = -FORCE_MAGNITUDE
        elif keys[pygame.K_RIGHT]:
            self.force = FORCE_MAGNITUDE
        return self.force

    def update_physics(self):
        # Get force from keyboard input
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
        
        # # Draw boundaries
        # pygame.draw.rect(self.screen, BOUNDARY_COLOR,
        #                 [0, 0, BOUNDARY_MARGIN, SCREEN_HEIGHT])
        # pygame.draw.rect(self.screen, BOUNDARY_COLOR,
        #                 [SCREEN_WIDTH - BOUNDARY_MARGIN, 0, BOUNDARY_MARGIN, SCREEN_HEIGHT])
        
        # # Draw boundary lines
        # pygame.draw.line(self.screen, RED,
        #                 (BOUNDARY_MARGIN, 0),
        #                 (BOUNDARY_MARGIN, SCREEN_HEIGHT), 2)
        # pygame.draw.line(self.screen, RED,
        #                 (SCREEN_WIDTH - BOUNDARY_MARGIN, 0),
        #                 (SCREEN_WIDTH - BOUNDARY_MARGIN, SCREEN_HEIGHT), 2)
  
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
        
        # Draw info text
        angle_text = self.font.render(f'Angle: {math.degrees(self.state[2]):.1f}°', True, BLUE)
        position_text = self.font.render(f'Position: {self.state[0]:.2f}m', True, BLUE)
        force_text = self.font.render(f'Force: {force:.1f}N', True, BLUE)
        # controls_text = self.font.render('Use LEFT and RIGHT arrow keys to control', True, BLUE)
        
        self.screen.blit(angle_text, (10, 10))
        self.screen.blit(position_text, (10, 50))
        self.screen.blit(force_text, (10, 90))
        # self.screen.blit(controls_text, (SCREEN_WIDTH - 400, 10))
        
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
    pygame.display.set_caption("Keyboard-Controlled Cart Pole")

    simulator = CartPoleSimulator(screen)
    simulator.run()

    pygame.quit()

if __name__ == "__main__":
    main()