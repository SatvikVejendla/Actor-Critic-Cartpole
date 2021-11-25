import gym
from ACModel import ACModel
import tensorflow as tf
from tensorflow import keras
import numpy as np

max_ts = 200


input_nodes = 4
hidden_nodes = 128
output_nodes = 2

model = ACModel(input_nodes, hidden_nodes, output_nodes)
model.load_weights("results/model/data.ckpt")


env = gym.make("CartPole-v0")

while(True):
    state = env.reset()

    for i in range(1, max_ts):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)


        action_probs, critic_value = model(state)

        action = model.actor.get_action(action_probs)

        state, reward, done, _ = env.step(action)
        env.render()
