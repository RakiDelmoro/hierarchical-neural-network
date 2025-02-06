from PendulumModel.model import neural_network
from PendulumModel.utils import visualize_episode, pendulum_simulation, parameters_init, train_neural_network, batch_simulation, neural_network_pendulum_simulation, denormalize_actions

def run_simulation():
    # Generate pendulum simulation
    states, actions = pendulum_simulation()

    visualize_episode(states, actions)

    # Neural network
    network_parameters = parameters_init(network_architecture=[4, 32, 64, 1])
    training_mode, test_mode = neural_network(network_parameters)
    training_runner = train_neural_network(training_mode, network_parameters)

    # Don't train yet!
    for epoch in range(100):
        training_set = batch_simulation(states, actions, batch_size=2098)
        loss = training_runner(training_set)
        print(f'EPOCH: {epoch} Loss: {loss}')

    predicted_state, predicted_action = neural_network_pendulum_simulation(test_mode)
    # Create animation
    visualize_episode(predicted_state, predicted_action)

run_simulation()
