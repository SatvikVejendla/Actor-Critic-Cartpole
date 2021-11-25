import tensorflow as tf
from tensorflow import keras

from layers.layer import Layer
from layers.actor import Actor
from layers.critic import Critic


class ACModel(tf.keras.Model):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):

        super().__init__()
        self.hidden = Layer("hidden", hidden_nodes, activation="relu")
        self.actor = Actor(output_nodes, activation="softmax", parent=self.hidden)
        self.critic = Critic(1, parent=self.hidden)

    def call(self, x):
        return self.actor(x), self.critic(x)
