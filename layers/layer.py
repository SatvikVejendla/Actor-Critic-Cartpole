import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

class Layer(tf.keras.Model):

    def __init__(self, type, nodes, activation=None, parent=None):
        super().__init__()

        self.type = type
        self.nodes = nodes
        self.activation = activation
        self.layer = Dense(nodes, activation=activation)
        self.parent = parent


    def call(self, inputs):
        inputs = inputs if self.parent == None else self.parent(inputs)
        return self.layer(inputs)

    def summary(self):
        print("Model Type: {}\tNodes: {}\tActivation: {}".format(self.type, self.nodes, self.activation))

