import random
from ANN.cell import Atom

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, size):
        self.axons = Atom.tensor([random.uniform(-1, 1) for _ in range(size)])
        self.dentrites = Atom.tensor(random.uniform(-1, 1))

    def __call__(self, input_x):
        ''' input_x should Atom.tensor so where just need to access the data'''
        batched_activation = []
        for each_batch in input_x.data:
            act = [sum((wi*xi for wi,xi in zip(self.axons.data, each_batch)), self.dentrites.data)]
            batched_activation.append(act)

        return Atom.tensor(batched_activation)

    def parameters(self):
        return self.axons + [self.dentrites]
    
    def __repr__(self):
        return f"Neuron_size: {len(self.axons)}"
