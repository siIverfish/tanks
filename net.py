# Values will be -1.0 to 1.0

import numpy as np
import random
import math
import pickle
import os

from abc import ABC, abstractmethod
from typing import List, Optional
from constants import *

# todo input & output classes?


ENTROPY = 0.2


def restrict_between(min_: float, x: float, max_: float):
    return min(max_, max(min_, x))


class Node:
    """
    A single node of a layer of a neural net.

    Args:
        outputs (Optional[Layer[Node]]): The nodes that the nodes in this layer will output to -- the next layer in the network
    """

    PERCENT_WEIGHTS_CHANGED = ENTROPY
    WEIGHT_CHANGE_AMOUNT = 0.4
    MAX_WEIGHT = 3
    LETTER_PRINT_CODE = "N"
    
    @classmethod
    def random_node(cls, *args, **kwargs):
        node_type = random.choice([Node, Node, Node, MaxCutoffNode, MinCutoffNode]) # node is 3x more likely
        # 60% chance of being a node, 20% chance of being a max cutoff node, 20% chance of being a min cutoff node
        return node_type(*args, **kwargs)

    def __init__(self, outputs: Optional[List["Node"]], update_outputs: bool = True):
        self.num_inputs = 0
        self.weights = []
        self.values = []
        if outputs is not None and update_outputs:
            for output in outputs:
                # print(f"Adding input to {output}")
                output.add_input()
        self.outputs = outputs

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
        # print(f"Resetting node {self}")
        assert len(self.values) == len(self.weights)
        assert len(self.values) == self.num_inputs
        self.values = []

    def input(self, value: float):
        """
        Adds a value to the node's `values` list..

        Args:
            value (float): A float between -2 and 2 to add to the `values` list.
        """
        # print(f"Node {self} received {value}")
        self.values.append(value)

    def process(self) -> float:
        """
        Returns the result of this node's calculations with its values and weights to be distributed.

        Returns:
            float: The result of `np.dot(self.values, self.weights) / self.num_inputs`
        """
        # print(len(self.values), len(self.weights))
        return np.dot(self.values, self.weights) / self.num_inputs # NODE PROCESS

    def pass_values(self):
        # print(f"Passing values... values: {self.values} & weights: {self.weights} node {self}")
        output_value = self.process()
        for output in self.outputs:
            # print(f"                                                                                                                                              outputting to {output}")
            output.input(output_value)
        self.reset()

    def make_random_change(self):
        # choose 1/4 of the weights to change
        # print("NODE MAKE RANDOM CHANGE")
        change_indeces = random.choices(
            list(range(len(self.weights))),
            k=math.ceil(len(self.weights) * self.PERCENT_WEIGHTS_CHANGED),
        )
        # change them
        # print("change indeces", change_indeces)
        for change_index in change_indeces:
            change_amount = random.uniform(
                -self.WEIGHT_CHANGE_AMOUNT, self.WEIGHT_CHANGE_AMOUNT
            )
            self.weights[change_index] += change_amount
            self.weights[change_index] = restrict_between(
                -self.MAX_WEIGHT, self.weights[change_index], self.MAX_WEIGHT
            )

    def __str__(self):
        return f"{self.LETTER_PRINT_CODE}:{self.num_inputs}"# {id(self):2}"


class CutoffNode(Node, ABC):
    MAX_CUTOFF = 5
    CUTOFF_CHANGE_AMOUNT = 2

    @abstractmethod
    def cutoff_operator(self, value: float, cutoff: float) -> bool:
        pass

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.cutoff = np.random.uniform(-self.MAX_CUTOFF, self.MAX_CUTOFF)

    def make_random_change(self):
        super().make_random_change()
        self.cutoff += random.uniform(
            -self.CUTOFF_CHANGE_AMOUNT, self.CUTOFF_CHANGE_AMOUNT
        )
        self.cutoff = restrict_between(-self.MAX_CUTOFF, self.cutoff, self.MAX_CUTOFF)

    def process(self) -> float:
        value = np.dot(self.values, self.weights) / self.num_inputs # CUTOFFNODE PROCESS
        if self.cutoff_operator(value, self.cutoff):
            return value
        else:
            return 0.0


class MaxCutoffNode(CutoffNode):
    LETTER_PRINT_CODE = "MAX"
    
    def cutoff_operator(self, value: float, cutoff: float) -> bool:
        return value > cutoff


class MinCutoffNode(CutoffNode):
    LETTER_PRINT_CODE = "MIN"
    
    def cutoff_operator(self, value: float, cutoff: float) -> bool:
        return value < cutoff


class Layer(list):
    """
    A class that represents a layer in the neural network and has methods for triggering and taking input.


    Args:
        num_nodes (int): The number of nodes in this layer. Should work with different values than the last layer.
        outputs (Optional[Layer[Node]]): The next nodes that this layer will output to.
    """

    PERCENT_NODES_CHANGED = ENTROPY

    def __init__(
        self, num_nodes: int, outputs: Optional[List[Node]], is_input_layer=False
    ):
        super().__init__([Node.random_node(outputs) for _ in range(num_nodes)])
        if is_input_layer:
            for node in self:
                node.add_input()

    def pass_values(self):
        """
        Calls the `pass_values` method on all of the nodes, passing their values to the next layer.
        """
        for node in self:
            node.pass_values()

    def input(self, node_inputs: List[float]):
        """
        This function should only be called on an input layer.
        It manually triggers the input of all nodes with the given input values.

        Args:
            inputs (List[float]): A list of inputs (BETWEEN -2 AND 2) to be given to the nodes
        """
        for node, node_input in zip(self, node_inputs):
            # print(f"Inputting {node_input} to {node}")
            node.input(node_input)

    def get_values(self) -> List[float]:
        """
        Returns a list of values from the nodes in this layer.
        Should only be called on an output layer.

        Returns:
            List[float]: The values in the layer, after processing
        """
        assert all(len(node.values) != 0 for node in self)
        val = [node.process() for node in self]
        for node in self:
            node.reset()
        return val

    def make_random_change(self):
        # choose 1/4 of the nodes to change
        change_nodes_indeces = random.choices(
            list(range(len(self))), k=math.ceil(len(self) * self.PERCENT_NODES_CHANGED)
        )
        # change them
        for change_node_index in change_nodes_indeces:
            if np.random.uniform() < 0.5:
                self[change_node_index].make_random_change()
            else:
                # print("CHANGING NODE TO RANDOM NODE")
                new_node = Node.random_node(self[change_node_index].outputs, update_outputs=False)
                for _ in range(self[change_node_index].num_inputs):
                    new_node.add_input()
                self[change_node_index] = new_node
                # self.pretty_print()

    def pretty_print(self):
        print(f"Layer: {' '.join(str(node) for node in self)}")


class NeuralNet:
    """
    A neural net. More here later.

    Args:
        num_inputs (int): The number of inputs (floats between -1 and 1) to this neural net.
        num_layers (int): The number of layers between the input and output layer.
        num_nodes_per_layer (int): The number of nodes in every middle layer. Input and output layers will have `num_inputs` and `num_outputs` nodes.
        num_outputs (int): The number of outputs from the neural net.
    """

    PERCENT_LAYERS_CHANGED = ENTROPY

    def __init__(
        self,
        *,
        num_inputs: int,
        num_layers: int,
        num_nodes_per_layer: int,
        num_outputs: int,
    ):
        self.layers = []
        self.num_inputs = num_inputs

        # Add output layer -- reverse order because each node needs to know its outputs first
        self.output_layer = Layer(num_outputs, None)
        self.layers.append(self.output_layer)

        # Add all of the layers in-between
        for _ in range(num_layers):
            new_layer = Layer(num_nodes_per_layer, self.layers[0])
            self.layers.insert(0, new_layer)

        # Add input layer last
        self.input_layer = Layer(num_inputs, self.layers[0], is_input_layer=True)
        self.layers.insert(0, self.input_layer)

    def get_prediction(self, node_inputs: List[float]) -> List[float]:
        assert len(node_inputs) == self.num_inputs, "wrong input"

        self.input_layer.input(node_inputs)
        self.input_layer.pass_values()

        # print()
        # self.input_layer.pretty_print()
        # print(f"\n{self.middle_layers}\n")
        # self.output_layer.pretty_print()
        # print()

        for layer in self.middle_layers:
            # print("\nonto nex/t layer\n")
            layer.pass_values()
        # print("\noutput layer\n")
        # print([node.values for node in self.output_layer])
        return self.output_layer.get_values()

    def pretty_print(self):
        # print("\nNode Print:\n")
        for layer in self.layers:
            layer.pretty_print()
        # print()

    def make_random_change(self):
        # print("NET MAKE RANDOM CHANGE")
        # choose 1/4 of the weights to change
        change_layers = random.choices(
            list(range(len(self.layers))),
            k=math.ceil((len(self.layers) * self.PERCENT_LAYERS_CHANGED)),
        )
        # print(f"change layers: {change_layers}")
        # change them
        for change_layer in change_layers:
            # print("CALLING MAKE RANDOM CHANGE")
            self.layers[change_layer].make_random_change()

    def save(self):
        count = 1
        while os.path.isfile(FILE_FORMAT.format(count)):
            count += 1
        with open(FILE_FORMAT.format(count), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def middle_layers(self) -> List[Node]:
        return self.layers[1:-1]
