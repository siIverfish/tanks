# Values will be -1.0 to 1.0

import numpy as np
import random
import math

from typing import List, Optional

# todo input & output classes?

def restrict_between(min_: float, x: float, max_: float):
    return min(max_, max(min_, x))

class Node:
    """
    A single node of a layer of a neural net.

    Args:
        outputs (Optional[Layer[Node]]): The nodes that the nodes in this layer will output to -- the next layer in the network
    """
    
    PERCENT_WEIGHTS_CHANGED = 0.25
    WEIGHT_CHANGE_AMOUNT = 0.2
    MAX_WEIGHT = 2
    
    def __init__(self, outputs: Optional[List['Node']]):
        self.num_inputs = 0
        self.weights    = []
        self.values     = []
        if outputs is not None:
            for output in outputs:
                output.add_input()
    
    def add_input(self):
        """
        Prepares this node for encountering another input in use. Adds to `num_inputs` and `weights`.
        """
        self.num_inputs += 1
        self.weights.append(random.uniform(-self.MAX_WEIGHT, self.MAX_WEIGHT))
    
    def reset(self):
        """
        Resets this node's values for the next trigger.
        """
        assert len(self.values) == len(self.weights)
        assert len(self.values) == self.num_inputs
        self.values = []
    
    def input(self, value: float):
        """
        Adds a value to the node's `values` list..

        Args:
            value (float): A float between -2 and 2 to add to the `values` list.
        """
        self.values.append(value)
    
    def process(self) -> float:
        """
        Returns the result of this node's calculations with its values and weights to be distributed.

        Returns:
            float: The result of `np.dot(self.values, self.weights) / self.num_inputs`
        """
        return np.dot(self.values, self.weights) / self.num_inputs
    
    def pass_values(self):
        output_value = self.process()
        for output in self.outputs:
            output.input(output_value)
        self.reset()
    
    def make_random_change(self):
        # choose 1/4 of the weights to change
        change_indeces = random.choices(
            list(range(len(self.weights))), 
            k=math.ceil(len(self.weights) * self.PERCENT_WEIGHTS_CHANGED)
        )
        # change them
        for change_index in change_indeces:
            change_amount = random.uniform(-self.WEIGHT_CHANGE_AMOUNT, self.WEIGHT_CHANGE_AMOUNT)
            self.weights[change_index] += change_amount
            self.weights[change_index] = restrict_between(-self.MAX_WEIGHT, self.weights[change_index], self.MAX_WEIGHT)
    
    def __str__(self):
        print(f"N:{self.num_inputs}")


class Layer(list):
    """
    A class that represents a layer in the neural network and has methods for triggering and taking input.


    Args:
        num_nodes (int): The number of nodes in this layer. Should work with different values than the last layer.
        outputs (Optional[Layer[Node]]): The next nodes that this layer will output to.
    """
    
    PERCENT_NODES_CHANGED = 0.25
    
    def __init__(self, num_nodes: int, outputs: Optional[List[Node]]):
        super().__init__([ Node(outputs) for _ in range(num_nodes) ])
    
    def pass_values(self):
        """
        Calls the `pass_values` method on all of the nodes, passing their values to the next layer.
        """
        for node in self.nodes:
            node.pass_values()

    def input(self, node_inputs: List[float]):
        """
        This function should only be called on an input layer. 
        It manually triggers the input of all nodes with the given input values.

        Args:
            inputs (List[float]): A list of inputs (BETWEEN -2 AND 2) to be given to the nodes
        """
        for node, node_input in zip(self.nodes, node_inputs):
            node.input(node_input)
    
    def get_values(self) -> List[float]:
        """
        Returns a list of values from the nodes in this layer.
        Should only be called on an output layer.

        Returns:
            List[float]: The values in the layer, after processing
        """
        assert all(len(node.values) != 0 for node in self.nodes)
        return [node.process_values() for node in self.nodes]
    
    def make_random_change(self):
        # choose 1/4 of the nodes to change
        change_nodes = random.choices(
            list(range(len(self.nodes))), 
            k=math.ceil(len(self.nodes) * self.PERCENT_NODES_CHANGED)
        )
        # change them
        for change_node in change_nodes:
            change_node.make_random_change
    
    
    def pretty_print(self):
        print(f"Layer: {''.join(str(node) for node in self.nodes)}")

class NeuralNet:
    """
    A neural net. More here later.

    Args:
        num_inputs (int): The number of inputs (floats between -1 and 1) to this neural net.
        num_layers (int): The number of layers between the input and output layer.
        num_nodes_per_layer (int): The number of nodes in every middle layer. Input and output layers will have `num_inputs` and `num_outputs` nodes.
        num_outputs (int): The number of outputs from the neural net.
    """
    
    PERCENT_LAYERS_CHANGED = 0.2
    
    def __init__(self, *, 
                 num_inputs: int, 
                 num_layers: int, 
                 num_nodes_per_layer: int, 
                 num_outputs: int):
        self.layers = []
        
        # Add output layer -- reverse order because each node needs to know its outputs first
        self.output_layer = Layer(num_outputs, None)
        self.layers.append(self.output_layer)
        
        # Add all of the layers in-between
        for _ in range(num_layers):
            new_layer = Layer(num_nodes_per_layer, self.layers[0])
            self.layers.append(new_layer)
        
        # Add input layer last
        self.input_layer = Layer(num_inputs, self.layers[0])
        self.layers.append(self.input_layer)
    
    
    def get_prediction(self, node_inputs: List[float]) -> List[float]:
        self.input_layer.input(node_inputs)
        for layer in self.middle_layers:
            layer.pass_values()
        return self.output_layer.get_values()

    def pretty_print(self):
        print("\nNode Print:\n")
        for layer in self.layers:
            layer.pretty_print()
        print()
    
    def make_random_change(self):
        # choose 1/4 of the weights to change
        change_layers = random.choices(
            list(range(len(self.layers))), 
            k=math.ceil(len(self.layers) * self.PERCENT_LAYERS_CHANGED)
        )
        # change them
        for change_layer in change_layers:
            change_layer.make_random_change
        
    @property
    def middle_layers(self):
        return self.layers[1:-1]
        