import numpy as np

class Node():
    def __init__(self, num_inputs, outputs):
       self.num_inputs = num_inputs
       self.input_count = 0
       self.values = []
       self.outputs = outputs
       self.weights = np.random.rand(num_inputs)
    
    def input(self, value):
        self.values.append(value)
        self.input_count += 1
        if self.input_count == self.num_inputs:
            self.output()
    
    def process_values(self):
        return np.dot(self.values, self.weights) / self.num_inputs
    
    def add_input(self):
        self.weights.append(np.random.rand())
    
    def remove_input(self):
        del self.weights[-1]
    
    def random_change(self):
        index = np.random.randint(0, self.num_inputs)
        self.weights[index] = np.random.rand()
    
    def output(self):
        out = self.process_values()
        for output in self.outputs:
            output.input(out)
        self.values = []
        self.input_count = 0
    
    @property
    def num_inputs(self):
        return len(self.weights)

class Layer:
    MAX_PERCENT_CHANGE = 0.3
    
    def __init__(self, num_nodes, next_nodes):
        self.nodes = [Node(0, next_nodes) for _ in range(num_nodes)]
        for node in next_nodes:
            for _ in range(num_nodes):
                node.add_input()
    
    def make_random_change(self):
        nodes = np.random.choices(
            self.nodes, 
            k=np.random.randint(0, int(len(self.nodes) * self.MAX_PERCENT_CHANGE))
        )
        for node in nodes:
            node.random_change()

class OutputNode:
    def __init__(self):
        self.values = []
        self.num_inputs = 0
    
    def add_input(self):
        self.num_inputs += 1
    
    def input(self, value):
        self.values.append(value)        
    
    def process_values(self):
        return round(value / self.num_inputs)

    def output(self):
        return self.process_values()

class InputNode:
    def __init__(self, outputs):
        self.outputs = outputs
    
    def trigger(self, value):
        for output in self.outputs:
            output.input(value)







