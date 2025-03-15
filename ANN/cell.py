
class Atom:
    """Atom is like a Torch"""

    def __init__(self, neuron_feed, _children=()):
        self.data = neuron_feed
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    @classmethod
    def tensor(cls, data, _children=()):
        return cls(data, _children)

    def __add__(self, bias):
        # Make sure the bias have the same properties as the activation
        bias = bias if isinstance(bias, Atom) else Atom(bias)
        # Calculate the output neuron
        output_neuron = self.tensor(self.data + bias.data, (self, bias))

        # Backward pass within this calculation
        def _backward():
            # Accumulate gradients
            self.grad += output_neuron.grad
            bias.grad += output_neuron.grad
        output_neuron._backward = _backward

        return output_neuron

    def __mul__(self, weight):
        # Make sure the weight have the same properties as the activation
        weight = weight if isinstance(weight, Atom) else Atom(weight)
        # Calculate the output neuron
        output_neuron = self.tensor(self.data * weight.data, (self, weight))

        # Backward pass within this calculation
        def _backward():
            # Accumulate gradients
            self.grad += weight.data * output_neuron.grad
            weight.grad += self.data * output_neuron.grad
        output_neuron._backward = _backward

        return output_neuron

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output_neuron = self.tensor(self.data**other, (self,))

        # Backward pass within this calculation
        def _backward():
            # Accumulate gradients
            self.grad += (other * self.data**(other-1)) * output_neuron.grad
        output_neuron._backward = _backward

        return output_neuron

    def relu(self):
        # Make sure the other neuron have a same properties in our Neuron
        output_neuron = self.tensor(0 if self.data < 0 else self.data, (self,))

        # Backward pass within this calculation
        def _backward():
            # Accumulate gradients
            self.grad += (output_neuron.data > 0) * output_neuron.grad
        output_neuron._backward = _backward

        return output_neuron

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def backward(self):
        # topological order of all atoms calculation 
        topo = []
        visited = set()

        def build_topological_order(ice_tensor_calc):
            if ice_tensor_calc not in visited:
                visited.add(ice_tensor_calc)
                for child in ice_tensor_calc._prev:
                    build_topological_order(child)
                topo.append(ice_tensor_calc)
        build_topological_order(self)

        # go one calculation at a time and apply the chain rule to get its gradient
        self.grad = 1 # Initialize gradient
        for each_calc in reversed(topo):
            each_calc._backward()

    def __repr__(self):
        # return self.data
        return f'Atom.tensor({self.data} grad={self.grad})'
        # return f"IceTensor(data={self.data}, grad={self.grad})"
