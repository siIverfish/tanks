import unittest

from net import (
    NeuralNet,
)

from copy import deepcopy

# test the neural net thing
class TestNeuralNet(unittest.TestCase):
    def test_deep_copy(self):
        net = NeuralNet(
            num_inputs=3,
            num_layers=4,
            num_nodes_per_layer=5,
            num_outputs=2,
        )
        assert deepcopy(net) is not net
if __name__ == '__main__':
    unittest.main()