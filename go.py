
# script to go

from trainer import Trainer

from net import NeuralNet
from fighter import Fighter

def net_path(x: str) -> str:
    return "resources/nets/" + x

fighter = Fighter()
# neural_net = NeuralNet(
#     num_inputs=41,
#     num_layers=10,
#     num_nodes_per_layer=40,
#     num_outputs=5
# )
neural_net = NeuralNet.load(net_path("net_6.netsave"))
trainer = Trainer(neural_net, fighter.fight_function)

count = 0
while True:
    trainer.train()
    count += 1
    if count % 10 == 0:
        trainer.neural_net.save()