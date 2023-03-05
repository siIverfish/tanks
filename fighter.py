
from game import (
    Tank,
    NeuralNetControlledTank,
    PygameHandler,
    Bullet
)

from pygame import error as pygame_error_probably_exit
import random

class Fighter:
    MAX_FRAMES = 10_000
    
    def __init__(self):
        ...
    
    def fight_function(self, model1, model2):
        blue_tank = NeuralNetControlledTank(
            location=(500, 500),
            angle=20,
            color="blue",
            key_parser=None,
            neural_net=model1,
        )
        red_tank = NeuralNetControlledTank(
            location=(100, 100),
            angle=20,
            color="red",
            key_parser=None,
            neural_net=model2,
        )
        PygameHandler().add(blue_tank)
        PygameHandler().add(red_tank)
        frame_count = 0
        while sum(1 for ele in PygameHandler().elements if isinstance(ele, Tank)) == 2 and frame_count <= self.MAX_FRAMES:
            try:
                PygameHandler().loop()
            except pygame_error_probably_exit:
                break
        remaining_tanks = [e for e in PygameHandler().elements if isinstance(e, Tank)]
        if len(remaining_tanks) != 1: # something's weird and wrong
            return random.choice([blue_tank, red_tank]) 
        return remaining_tanks[0]