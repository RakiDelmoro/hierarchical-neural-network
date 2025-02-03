import pygame
import math
import numpy as np

class InvertedPendulum:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 500, 500
        self.FPS = 60
        self.COLORS = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'RED': (255, 0, 0),
            'BLUE': (0, 0, 255),
            'GRAY': (128, 128, 128)
        }
        self.CART_WIDTH, self.CART_HEIGHT = 50, 10
        self.TRACK_LEFT, self.TRACK_RIGHT = 50, self.WIDTH - 50
        self.g, self.m, self.M, self.l = 9.81, 1.0, .1, 1.0
        self.dt = 1.0 / self.FPS
        self.cart_force = 30.0
        self.reset()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Inverted Pendulum RL Environment")
        self.clock = pygame.time.Clock()

    def reset(self):
        """Reset environment to initial state with pole pointing downward"""
        self.theta = math.pi  # Start pointing straight down
        self.theta_dot = 0.0  # No initial angular velocity
        self.x = self.WIDTH // 2  # Start at track center
        self.x_dot = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Normalized state: [cos(theta), sin(theta), angular_velocity, cart_position]"""
        x_normalized = ((self.x - self.TRACK_LEFT) / (self.TRACK_RIGHT - self.TRACK_LEFT) * 2) - 1
        return [
            math.cos(self.theta),  # Pole direction (x component)
            math.sin(self.theta),  # Pole direction (y component)
            self.theta_dot * 0.1,  # Scaled angular velocity
            x_normalized           # Normalized cart position
        ]

    def _calculate_reward(self):
        """Reward function encouraging upward swing and stabilization"""
        reward = math.cos(self.theta)  # Max reward when pole is upright (θ=0)
        
        # Additional reward for upward swing above horizontal
        if abs(self.theta) < math.pi/2:
            reward += 1.0
        
        # Penalize cart position deviation from center
        track_center = (self.TRACK_LEFT + self.TRACK_RIGHT) / 2
        x_penalty = 0.1 * ((self.x - track_center) / (self.TRACK_RIGHT - track_center)) ** 2
        reward -= x_penalty
        
        return reward

    def _check_done(self):
        """Terminate only if cart goes out of track bounds"""
        return self.x <= self.TRACK_LEFT or self.x >= self.TRACK_RIGHT

    def update_physics(self, action):
        """Physics update remains unchanged (correct swing dynamics)"""
        F = action * self.cart_force
        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)
        
        denominator = self.M + self.m * sin_theta**2
        x_ddot = (F + self.m * sin_theta * (self.l * self.theta_dot**2 + self.g * cos_theta)) / denominator
        theta_ddot = (-F * cos_theta - self.m * self.l * self.theta_dot**2 * sin_theta * cos_theta 
                      + (self.M + self.m) * self.g * sin_theta) / (self.l * denominator)
        
        self.x_dot += x_ddot * self.dt
        self.theta_dot += theta_ddot * self.dt
        
        # Apply damping
        self.x_dot *= 0.999
        self.theta_dot *= 0.999
        
        # Update positions with track boundary enforcement
        self.x += self.x_dot * self.dt
        self.x = np.clip(self.x, self.TRACK_LEFT + self.CART_WIDTH/2, 
                        self.TRACK_RIGHT - self.CART_WIDTH/2)
        self.theta += self.theta_dot * self.dt

    def step(self, action):
        """Perform one timestep with given action"""
        self.update_physics(action)
        reward = self._calculate_reward()
        self.done = self._check_done()
        return self._get_state(), reward, self.done, {}


    def render(self):
        """Visualize current state"""
        self.screen.fill(self.COLORS['WHITE'])
        l = self.l * 100  # Scale for visualization
        cart_y = self.HEIGHT // 2
        
        # Draw track
        pygame.draw.line(self.screen, self.COLORS['GRAY'],
                        (self.TRACK_LEFT, cart_y),
                        (self.TRACK_RIGHT, cart_y), 5)
        
        # Draw cart
        pygame.draw.rect(self.screen, self.COLORS['BLUE'],
                        (self.x - self.CART_WIDTH//2, cart_y - self.CART_HEIGHT//2,
                         self.CART_WIDTH, self.CART_HEIGHT))
        
        # Draw pendulum
        pendulum_end = (self.x + l * math.sin(self.theta),
                       cart_y - l * math.cos(self.theta))
        pygame.draw.line(self.screen, self.COLORS['BLACK'],
                        (self.x, cart_y), pendulum_end, 3)
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
            action = np.random.choice([-1, 1])  # Random policy example
            state, reward, done, _ = self.step(action)
            # print(reward)
            print(done)
            self.render()
            self.clock.tick(self.FPS)
            
            if done:
                state = self.reset()
                
        pygame.quit()

# Example usage with random agent
if __name__ == "__main__":
    env = InvertedPendulum()
    env.run()