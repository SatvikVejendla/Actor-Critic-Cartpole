import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.math import log as tf_log
import numpy as np
from .output_layer import OutputLayer

class Actor(OutputLayer):

    def __init__(self, nodes, activation=None, parent=None):
        super().__init__("actor", nodes, activation, parent)

    def get_action(self, action_probs):
        return np.random.choice(self.nodes, p=np.squeeze(action_probs))

    def compute_loss(self, log_prob, advantage):
        return -log_prob * advantage
    
    def compute_history_score(self, action_probabilties, action):
        return tf_log(action_probabilties[0, action])
