import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import numpy as np

from tensorflow.keras.losses import Huber
from .output_layer import OutputLayer


huber_loss = Huber()
class Critic(OutputLayer):

    def __init__(self, nodes, activation=None, parent=None):
        super().__init__("critic", nodes, activation, parent)
    
    def compute_loss(self, value, target):
        return huber_loss(tf.expand_dims(value, 0), tf.expand_dims(target, 0))

    def compute_history_score(self, critic_values):
        return critic_values[0,0]
