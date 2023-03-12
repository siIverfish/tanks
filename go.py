
# script to go

from trainer import Trainer

from net import NeuralNet
from fighter import Fighter
from constants import FILE_FORMAT

import os

def load_latest_net() -> NeuralNet:
    i = 1
    while os.path.isfile(FILE_FORMAT.format(i)):
        i += 1
    print(f"LOADING NET {i}")
    return NeuralNet.load(FILE_FORMAT.format(i - 1))

def new_net() -> NeuralNet:
    neural_net = NeuralNet(
        num_inputs=44,
        num_layers=4,
        num_nodes_per_layer=50,
        num_outputs=5
    )
    neural_net.save()
    return neural_net
    

fighter = Fighter()
trainer = Trainer(load_latest_net(), fighter.fight_function)

while True:
    trainer.train()