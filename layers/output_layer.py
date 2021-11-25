from .layer import Layer
from abc import abstractmethod
from memory import Memory

class OutputLayer(Layer):
    def __init__(self, type, nodes, activation=None, parent=None):
        super().__init__(type, nodes, activation, parent)

        self.layer_losses = Memory()
        self.values_history = Memory()

    def add_history(self, val):
        self.values_history.add(val)

    def get_history(self):
        return self.values_history.data

    def clear_history(self):
        self.values_history.clear()

    def add_loss(self, loss):
        self.layer_losses.add(loss)

    def get_sum_loss(self):
        return self.layer_losses.sum()

    @abstractmethod
    def compute_loss(self):
        return

    @abstractmethod
    def compute_history_score(self):
        return