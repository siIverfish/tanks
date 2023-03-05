import unittest

from net import (
    NeuralNet,
)

from trainer import Trainer

import math
import random
from copy import deepcopy



def parse_response(r):
    return r[0]

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
    
    TRAIN_LEN = 200
    
    # def test_train(self):
    #     net = NeuralNet(
    #         num_inputs=1,
    #         num_layers=4,
    #         num_nodes_per_layer=4,
    #         num_outputs=1
    #     )
    #     net.pretty_print()
        
    #     def training_func(n1, n2):
    #         x = random.choice([1, -1])
    #         output = n1.get_prediction([x])
    #         # print(output[0])
    #         output = round(output[0])
    #         return math.copysign(1, output) == -x
        
    #     trainer = Trainer(net, training_func)
    #     for _ in range(200):
    #         trainer.train()
            
    #     assert training_func(trainer.neural_net, None)
    
    def test_fight_adding(self):
        net = NeuralNet(
            num_inputs=2,
            num_layers=5,
            num_nodes_per_layer=4,
            num_outputs=1
        )

        def parse_response(r):
            return r[0]
        
        def training_func(n1, n2):
            a = random.randint(0, 10)
            b = random.randint(0, 10)
            result = a + b
            n1_result = parse_response(n1.get_prediction([a, b]))
            n2_result = parse_response(n2.get_prediction([a, b]))
            if abs(n1_result - result) < abs(n2_result - result):
                return True
            return False
        
        trainer = Trainer(net, training_func)
        
        for _ in range(self.TRAIN_LEN):
            trainer.train()
        
        # trainer.neural_net.save()
        
        assert round(parse_response(trainer.neural_net.get_prediction([8, 8]))) == 16
        assert round(parse_response(trainer.neural_net.get_prediction([1, 4]))) == 5
        assert round(parse_response(trainer.neural_net.get_prediction([3, 7]))) == 10
    
    def test_fight_subtracting(self):
        net = NeuralNet(
            num_inputs=2,
            num_layers=5,
            num_nodes_per_layer=4,
            num_outputs=1
        )
        
        def training_func(n1, n2):
            a = random.randint(0, 10)
            b = random.randint(0, 10)
            result = a - b
            n1_result = parse_response(n1.get_prediction([a, b]))
            n2_result = parse_response(n2.get_prediction([a, b]))
            if abs(n1_result - result) < abs(n2_result - result):
                return True
            return False
        
        trainer = Trainer(net, training_func)
        
        for _ in range(self.TRAIN_LEN):
            trainer.train()
        
        # print(round(parse_response(trainer.neural_net.get_prediction([8, 8]))))
        # print(round(parse_response(trainer.neural_net.get_prediction([1, 4]))))
        # print(round(parse_response(trainer.neural_net.get_prediction([3, 7]))))
        # print(round(parse_response(trainer.neural_net.get_prediction([9, 2]))))
        
        assert round(parse_response(trainer.neural_net.get_prediction([8, 8]))) == 0
        assert round(parse_response(trainer.neural_net.get_prediction([1, 4]))) == -3
        assert round(parse_response(trainer.neural_net.get_prediction([3, 7]))) == -4
        assert round(parse_response(trainer.neural_net.get_prediction([9, 2]))) == 7
        
    def test_load(self):
        net = NeuralNet.load("./resources/nets/n2015296599376.netsave")
        
        print(parse_response(net.get_prediction([8, 8])))
        print(parse_response(net.get_prediction([1, 4])))
        print(parse_response(net.get_prediction([3, 7])))
        
        assert round(parse_response(net.get_prediction([8, 8]))) == 16
        assert round(parse_response(net.get_prediction([1, 4]))) == 5
        assert round(parse_response(net.get_prediction([3, 7]))) == 10
        
        
        
if __name__ == '__main__':
    unittest.main()