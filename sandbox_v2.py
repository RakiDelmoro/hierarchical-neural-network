import math
import pygame
import numpy as np

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

# Visualize pendulum simulation and I print reward for debugging purposes
def runner(states):
    running = True
    state = initialize_state()
    for each in range(len(states)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            episode_state = states[each]
            for i in range(len(episode_state)):
                visualize_current_state(episode_state[i])
                clock.tick(FPS)

class InvertedPendulum:
    def __init__(self, 
                 m_cart=1.0,     # mass of cart (kg)
                 m_pole=0.1,     # mass of pole (kg)
                 total_length=1.0,  # total length of pole (m)
                 gravity=9.81):  # gravitational acceleration (m/s^2)
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.total_length = total_length
        self.g = gravity
        
        # Derived parameters
        self.total_mass = m_cart + m_pole
        self.pole_mass_length = m_pole * total_length

    def dynamics(self, state, force):
        """
        Compute the dynamics of the inverted pendulum system
        
        State vector: [x, x_dot, theta, theta_dot]
        - x: cart position (m)
        - x_dot: cart velocity (m/s)
        - theta: pole angle from vertical (radians)
        - theta_dot: pole angular velocity (rad/s)
        
        Force: Applied force to the cart (N)
        
        Returns: State derivatives
        """
        x, x_dot, theta, theta_dot = state
        
        # Trigonometric helpers
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Denominator term
        denom = self.total_mass - self.m_pole * cos_theta * cos_theta
        
        # Acceleration of cart
        x_ddot = (force + self.pole_mass_length * theta_dot**2 * sin_theta - 
                  self.m_pole * self.g * cos_theta * sin_theta) / denom
        
        # Angular acceleration of pole
        theta_ddot = (self.g * sin_theta - x_ddot * cos_theta) / self.total_length
        
        return [x_dot, x_ddot, theta_dot, theta_ddot]

    def runge_kutta_integration(self, state, force, dt=0.01):
        """
        4th order Runge-Kutta numerical integration
        """
        k1 = np.array(self.dynamics(state, force))
        k2 = np.array(self.dynamics(state + dt/2 * k1, force))
        k3 = np.array(self.dynamics(state + dt/2 * k2, force))
        k4 = np.array(self.dynamics(state + dt * k3, force))
        
        next_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return next_state

    def generate_swing_up_trajectories(self, 
                                       num_trajectories=1000, 
                                       trajectory_length=200,
                                       exploration_noise=0.1):
        """
        Generate training trajectories for swing-up control
        
        Returns:
        - states: Array of state trajectories
        - actions: Array of applied force trajectories
        """
        states_collection = []
        actions_collection = []
        
        for _ in range(num_trajectories):
            # Initial random state
            state = np.array([
                WIDTH // 2,     # x
                np.random.uniform(-1, 1),     # x_dot
                np.pi,  # theta
                np.random.uniform(-2, 2)      # theta_dot
            ])
            
            trajectory_states = [state]
            trajectory_actions = []
            
            for _ in range(trajectory_length):
                # Simple swing-up policy with noise
                # Energy-based control with random exploration
                energy = self.potential_energy(state) + 0.5 * self.kinetic_energy(state)
                force = -np.sign(state[2]) * np.abs(np.sin(state[2])) * 10
                
                # Add exploration noise
                force += np.random.normal(0, exploration_noise)
                
                # Clip force to realistic bounds
                force = np.clip(force, -20, 20)
                
                trajectory_actions.append(force)
                
                # Simulate next state
                state = self.runge_kutta_integration(state, force)
                trajectory_states.append(state)
            
            states_collection.append(trajectory_states)
            actions_collection.append(trajectory_actions)
        
        return (np.array(states_collection), 
                np.array(actions_collection))

    def potential_energy(self, state):
        """Compute potential energy of the system"""
        x, _, theta, _ = state
        return self.pole_mass_length * self.g * (1 - np.cos(theta))

    def kinetic_energy(self, state):
        """Compute kinetic energy of the system"""
        x, x_dot, theta, theta_dot = state
        return 0.5 * self.total_mass * x_dot**2 + \
               0.5 * self.pole_mass_length * (x_dot * np.cos(theta) + 
                                              self.total_length * theta_dot)**2

import matplotlib.pyplot as plt

def visualize_trajectories(states):
    """
    Visualize generated trajectories
    
    Args:
    - states: numpy array of state trajectories
    """
    plt.figure(figsize=(15, 10))
    
    # Plot cart position
    plt.subplot(2, 2, 1)
    plt.title('Cart Position')
    for traj in states:
        plt.plot(traj[:, 0])
    plt.xlabel('Time Step')
    plt.ylabel('Position (m)')
    
    # Plot pole angle
    plt.subplot(2, 2, 2)
    plt.title('Pole Angle')
    for traj in states:
        plt.plot(traj[:, 2])
    plt.xlabel('Time Step')
    plt.ylabel('Angle (radians)')
    
    # Plot cart velocity
    plt.subplot(2, 2, 3)
    plt.title('Cart Velocity')
    for traj in states:
        plt.plot(traj[:, 1])
    plt.xlabel('Time Step')
    plt.ylabel('Velocity (m/s)')
    
    # Plot angular velocity
    plt.subplot(2, 2, 4)
    plt.title('Angular Velocity')
    for traj in states:
        plt.plot(traj[:, 3])
    plt.xlabel('Time Step')
    plt.ylabel('Angular Velocity (rad/s)')
    
    plt.tight_layout()
    plt.show()
    

def main():
    # Create inverted pendulum simulation
    pendulum = InvertedPendulum()
    
    # Generate training trajectories
    states, actions = pendulum.generate_swing_up_trajectories(
        num_trajectories=100, 
        trajectory_length=200
    )

    runner(states)

main()
