import random
import numpy as np
from features import GREEN, RED, RESET
from model.utilities import init_params, neuron, softmax, one_hot_encoded, init_model_parameters, init_weights_stress_transport

def network(network_architecture=[100, 10], neuron_architecture=[28, 100, 1], output_neurons_size=65):
    # Params init
    input_to_neurons_parameters = init_params(28, 100)
    memory_to_memory_parameters = init_params(100, 100)
    readout_parameters = [init_params(100*28, 1) for _ in range(output_neurons_size)]

    # Neurons
    neurons = [neuron(input_to_neurons_parameters, memory_to_memory_parameters, readout_parameters[each]) for each in range(output_neurons_size)]

    def forward(input_neurons):
        neurons_activation = []
        memories = []
        for each_neuron in range(output_neurons_size):
            neuron_activation, neurons_memories = neurons[each_neuron](input_neurons)
            neurons_activation.append(neuron_activation)
            memories.append(neurons_memories)
        output_neurons = softmax(np.concatenate(neurons_activation, axis=1, dtype=np.float32))
        return output_neurons, memories

    def neurons_stress(model_output, expected_output):
        avg_neurons_loss = -np.mean(np.sum(expected_output * np.log(model_output + 1e-15), axis=1))
        return avg_neurons_loss

    def update_each_neuron(neuron_memory, neuron_stress, parameters):
        # Backprop SUCKS Direct feedback error BETTER!
        stress = neuron_stress.reshape(-1, 1)

        weights = parameters[0]
        bias = parameters[1]
        # Update parameters
        weights -= 0.01 * np.matmul(neuron_memory.transpose(), stress) / neuron_stress.shape[0]
        bias -= 0.01 * np.sum(stress, axis=0) / stress.shape[0]

    def train_neurons(prediction, expected, neurons_memories):
        for neuron_idx in range(output_neurons_size):
            neuron_memory = neurons_memories[neuron_idx]
            neuron_activation = prediction[:, neuron_idx]
            expected_neuron_activation = expected[:, neuron_idx]
            neuron_parameters = readout_parameters[neuron_idx]
            # Mean squared error for a neuron
            neuron_stress = 2*(neuron_activation - expected_neuron_activation)
            update_each_neuron(neuron_memory, neuron_stress, neuron_parameters)

    def training_phase(dataloader):
        print("TRAINING....")
        batch_losses = []
        for batch_image, batch_expected in dataloader:
            input_batch_image = batch_image.reshape(-1, 28, 28)
            prediction, neurons_memories = forward(input_batch_image)
            avg_neurons_stress = neurons_stress(prediction, batch_expected)
            train_neurons(prediction, batch_expected, neurons_memories)
            print(avg_neurons_stress)
            batch_losses.append(avg_neurons_stress)

        return np.mean(np.array(batch_losses))

    def testing_phase(dataloader):
        print("TESTING....")
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            input_image = batched_image.reshape(-1, 28, 28)
            batched_label = batched_label.argmax(axis=-1)
            prediction, _ = forward(input_image)
            batch_accuracy = (prediction.argmax(axis=-1) == batched_label).mean()
            for each in range(len(batched_label)//10):
                model_prediction = prediction[each].argmax(-1)
                if model_prediction == batched_label[each]: correctness.append((model_prediction.item(), batched_label[each].item()))
                else: wrongness.append((model_prediction.item(), batched_label[each].item()))
            print(f'Number of samples: {i+1}\r', end='', flush=True)
            accuracy.append(np.mean(batch_accuracy))
        random.shuffle(correctness)
        random.shuffle(wrongness)
        print(f'{GREEN}Model Correct Predictions{RESET}')
        [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correctness) if i < 10]
        print(f'{RED}Model Wrong Predictions{RESET}')
        [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrongness) if i < 10]
        return np.mean(np.array(accuracy)).item()

    return training_phase, testing_phase
