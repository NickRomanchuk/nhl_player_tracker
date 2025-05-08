import numpy as np

class Queue:
    def __init__(self, object, n):
        self.queue = [object] * n
    
    def add(self, object):
        self.queue.pop()
        self.queue.append(object)
    
    def average(self):
        return np.mean(self.queue, axis=0)