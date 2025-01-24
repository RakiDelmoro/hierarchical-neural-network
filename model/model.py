import random
import numpy as np
from features import GREEN, RED, RESET
from neurons import linear_neurons
from model.utilities import init_params, neuron, softmax, one_hot_encoded, init_model_parameters, init_weights_stress_transport

def network(neuron_properties=[1000, 1], output_neurons_size=10):
    # Params init
    input_to_neurons_parameters = [init_params(784, 1000) for _ in range(output_neurons_size)]
    # Shared for all neurons
    shared_neurons_parameters = init_model_parameters(neuron_properties)

    outer_stress_weight_transport = [np.random.rand(1, 1000) for _ in range(output_neurons_size)]
    neuron_stress_weight_transport = init_weights_stress_transport(neuron_properties)

    # Neurons
    neurons = [neuron(shared_neurons_parameters) for _ in range(output_neurons_size)]

    def forward(input_neurons):
        neurons_activation = []
        neurons_memories = []
        for each_neuron in range(output_neurons_size):
            input_for_neuron = linear_neurons(input_neurons, input_to_neurons_parameters[each_neuron])
            neuron_activation, neuron_memories = neurons[each_neuron](input_for_neuron)
            neurons_activation.append(neuron_activation)
            neurons_memories.append(neuron_memories)
        output_neurons = softmax(np.concatenate(neurons_activation, axis=1, dtype=np.float32))
        return output_neurons, neurons_memories

    def neurons_stress(model_output, expected_output):
        avg_neurons_loss = -np.mean(np.sum(expected_output * np.log(model_output + 1e-15), axis=1))
        return avg_neurons_loss

    def update_each_neuron(neuron_memory, neuron_stress):
        # Backprop SUCKS Direct feedback error BETTER!
        stress = neuron_stress.reshape(-1, 1)

        for i in range(len(shared_neurons_parameters)):
            memory = neuron_memory[-(i+2)]
            neuron_stress = np.matmul(stress, neuron_stress_weight_transport[i])
            weights = shared_neurons_parameters[-(i+1)][0]
            bias = shared_neurons_parameters[-(i+1)][1]
            # Update parameters
            weights -= 0.01 * np.matmul(memory.transpose(), neuron_stress) / neuron_stress.shape[0]
            bias -= 0.01 * np.sum(neuron_stress, axis=0) / neuron_stress.shape[0]

    def update_outer_connections(memory_neurons, neuron_stress):
        stress = neuron_stress.reshape(-1, 1)
        for each in range(len(input_to_neurons_parameters)):
            neuron_stress = np.matmul(stress, outer_stress_weight_transport[each])
            weights = input_to_neurons_parameters[-(each+1)][0]
            bias = input_to_neurons_parameters[-(each+1)][1]
            # Update parameters
            weights -= 0.01 * np.matmul(memory_neurons.transpose(), neuron_stress) / memory_neurons.shape[0]
            bias -= 0.01 * np.sum(neuron_stress, axis=0) / memory_neurons.shape[0]

    def train_neurons(prediction, expected, input_neurons, neurons_memories):
        total_output_neurons = prediction.shape[-1]
        for neuron_idx in range(total_output_neurons):
            neuron_memory = neurons_memories[neuron_idx]
            neuron_activation = prediction[:, neuron_idx]
            expected_neuron_activation = expected[:, neuron_idx]
            # Mean squared error for a neuron
            neuron_stress = 2*(neuron_activation - expected_neuron_activation)
            update_each_neuron(neuron_memory, neuron_stress)
            update_outer_connections(input_neurons, neuron_stress)

    def training_phase(dataloader):
        batch_losses = []
        for batch_image, batch_expected in dataloader:
            input_batch_image = batch_image
            one_hot_encoded_expected = one_hot_encoded(batch_expected)
            prediction, neurons_memories = forward(input_batch_image)
            avg_neurons_stress = neurons_stress(prediction, one_hot_encoded_expected)
            train_neurons(prediction, one_hot_encoded_expected, input_batch_image, neurons_memories)
            print(avg_neurons_stress)
            batch_losses.append(avg_neurons_stress)

        return np.mean(np.array(batch_losses))

    def testing_phase(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            input_image = batched_image
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
