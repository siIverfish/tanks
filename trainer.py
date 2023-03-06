import random

from net import (
    NeuralNet,
)

from copy import deepcopy
from typing import List, Callable
from icecream import ic
from constants import *

import os

# todo: make get_winner better
# todo: store more than one net


class Trainer:
    """A trainer for neural nets that handles evolution logic.

    Args:
        neural_net (NeuralNet): A neural net to train. Can be freshly instantiated.
        fight_function (Callable): A function takes (model, other_model) and returns True if the first model wins.
    """

    BATCH_SIZE = 7

    def __init__(self, neural_net: NeuralNet, fight_function: Callable):
        self.neural_net = neural_net
        neural_net.pretty_print()
        self.fight_function = fight_function

    def get_random_old_winner(self) -> NeuralNet:
        count = 1
        while os.path.isfile(FILE_FORMAT.format(count)):
            count += 1
        random_index = random.randint(1, count - 1)
        return NeuralNet.load(FILE_FORMAT.format(random_index))

    def get_offspring(self, net: NeuralNet) -> NeuralNet:
        """
        Returns a slightly changed neural net.

        Args:
            net (NeuralNet): The net to base the clone off of.

        Returns:
            NeuralNet: The new, randomly modified net.
        """
        copy = deepcopy(net)
        copy.make_random_change()
        return copy

    def generate_batch(self, net: NeuralNet, batch_size: int) -> List[NeuralNet]:
        """
        Generates `batch_size` new neural networks based on the old one and also the old one itself.

        Args:
            net (NeuralNet): The old net it's based off of
            batch_size (int): Number of nets total, include old one -- batch size one will just return [net]
        """
        new_nets = [net]
        for _ in range(batch_size - 1):
            if random.uniform(0, 1) < 0.2:
                new_nets.append(self.get_random_old_winner())
            else:
                new_nets.append(self.get_offspring(net))
        return new_nets

    def get_winner(self, batch: List[NeuralNet]) -> NeuralNet:
        # this function is weird
        # get the scores of all the models fighting all the other models. Could maybe do less fighting?
        # TODO: store fighting for both
        scores = [0 for _ in batch]
        for index, neural_net in enumerate(batch):
            # for other_index, other_neural_net in enumerate(batch[index+1:]):
            for other_index, other_neural_net in enumerate(batch[index + 1 :]):
                # print(other_index)
                if neural_net is other_neural_net:
                    continue
                if self.fight_function(neural_net, other_neural_net) is neural_net:
                    scores[index] += 1
                else:
                    scores[other_index + index + 1] += 1
        # return the best one
        # print(max(scores), neural_net)
        best_score = max(scores)
        ic(best_score)
        if scores.count(best_score) > 1:
            # resolve ties
            best_neural_nets = []
            while best_score in scores:
                index = scores.index(best_score)
                scores[index] = -1  # so that it isnt picked again
                best_neural_nets.append(batch[index])
            return random.choice(best_neural_nets)
        else:  # dont need the else but its nice
            return batch[scores.index(best_score)]

    def train(self):
        batch = self.generate_batch(self.neural_net, self.BATCH_SIZE)
        winner = self.get_winner(batch)
        ic(winner)
        winner.save()
        # if winner != self.neural_net:
        #     print("WINNER CHANGED")
        #     assert self.fight_function(winner, self.neural_net)
        self.neural_net = winner
