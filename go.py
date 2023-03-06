
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

fighter = Fighter()
neural_net = NeuralNet(
    num_inputs=41,
    num_layers=3,
    num_nodes_per_layer=40,
    num_outputs=5
)
neural_net.save()
trainer = Trainer(neural_net, fighter.fight_function)

count = 0
while True:
    trainer.train()
    count += 1
    if count % 10 == 0:
        trainer.neural_net.save()