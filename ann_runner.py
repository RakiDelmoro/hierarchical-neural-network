import torch
import random
import numpy as np
from ANN.cell import Atom
from ANN.nn import Neuron

# Example 1: Simple operations
x = Atom.tensor(2.0)
w = Atom.tensor(3.0)
a = x * w + x

a.backward()
print(x)
print(w)
