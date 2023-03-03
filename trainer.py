import numpy as np

class Node():
    def __init__(self, num_inputs):
       self.num_inputs = num_inputs
       self.input_count = 0
       
       self.values = []
       self.outputs = []
       
       self.weights = np.random.rand(num_inputs)
    
    def input(self, value):
        self.values.append(value)
        self.input_count += 1
        if self.input_count == self.num_inputs:
            self.output()
    
    def process_values(self):
        return 
    
    def output(self):
        for output in self.outputs:
            output.input(self.values)
        self.values = []
        self.input_count = 0