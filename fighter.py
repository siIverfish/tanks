
from game import (
    Tank,
    NeuralNetControlledTank,
    PygameHandler,
    Bullet
)

from pygame import error as pygame_error_probably_exit
import random
import time

class Fighter:
    MAX_FRAMES = 600
    
    def __init__(self):
        pass
        # PygameHandler(in_headless_mode=True)
    
    def cleanup(self):
        PygameHandler().elements = []
    
    def fight_function(self, model1, model2):
        blue_tank = NeuralNetControlledTank(
            location=(500, 500),
            angle=135,
            color="blue",
            key_parser=None,
            neural_net=model1,
        )
        red_tank = NeuralNetControlledTank(
            location=(100, 100),
            angle=-45,
            color="red",
            key_parser=None,
            neural_net=model2,
        )
        # PygameHandler(in_headless_mode=True)
        PygameHandler().add(blue_tank)
        PygameHandler().add(red_tank)
        frame_count = 0
        start_time = time.perf_counter()
        while sum(1 for ele in PygameHandler().elements if isinstance(ele, Tank)) == 2 and frame_count <= self.MAX_FRAMES:
            try:
                status = PygameHandler().loop()
                if status is not None:
                    if status == "quit":
                        return random.choice([blue_tank, red_tank]) 
                    elif status == "red":
                        return red_tank
                    elif status == "blue":
                        return blue_tank
            except pygame_error_probably_exit:
                print(f"Pygame error, probably an exit")
                raise pygame_error_probably_exit
                exit(0)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"still going, fps={100 / (time.perf_counter() - start_time)}")
                start_time = time.perf_counter()
        
        # determine winner if there was a timeout
        if frame_count >= self.MAX_FRAMES:
            red_tank_hp = sum(ele.hp for ele in PygameHandler().elements if isinstance(ele, Tank) and ele.color == "red")
            blue_tank_hp = sum(ele.hp for ele in PygameHandler().elements if isinstance(ele, Tank) and ele.color == "blue")
            self.cleanup()
            if red_tank_hp > blue_tank_hp:
                return red_tank
            elif blue_tank_hp > red_tank_hp:
                return blue_tank
        
        remaining_tanks = [e for e in PygameHandler().elements if isinstance(e, Tank)]
        self.cleanup()
        if len(remaining_tanks) != 1: # something's weird and wrong
            return random.choice([blue_tank, red_tank]) 
        return remaining_tanks[0]