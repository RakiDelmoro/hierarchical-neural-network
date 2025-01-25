
# QUESTION TO SOLVE:
# How to make a neuron have it's own properties and learn the pattern of input and make a decision wether that neuron will fire or not

# The idea of having each neuron have it's own MLP (not shared) is we want each neuron to learn what's in the input for example:
# If our input is image of a DOG: We have a outer network of 784, 10, 2 (Only 2 class we want to classify wether an image is a dog or a cat)
# Middle neurons 10:
# each neuron has it's own MLP we want each neuron to activate if there's pattern in the input example dog have a longer face compare to cat
# So maybe one of the 10 neurons learn that the dog have longer face and if our input is dog that neuron will fire (like in other neurons we want to learn)
# What's in our data and fire if that is met (same as for cat)

# Neuron have it's own MLP and the readout is shared of all neurons
# The idea of each neurons have it's own MLP is that each neuron will learn differently about what is the pattern of the input
# Readout is shared for all neurons (Should we update the readout connections or not🤔)
