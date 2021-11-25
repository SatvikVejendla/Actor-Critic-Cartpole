import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.math import log as tf_log
from tensorflow import GradientTape
from abc import abstractmethod
from memory import Memory
from ACModel import ACModel

env = gym.make("CartPole-v0")

seed = 15
env.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


gamma = 0.95
eps = np.finfo(np.float32).eps.item()
max_ts = 200
learning_rate = 0.005

input_nodes = 4
hidden_nodes = 128
output_nodes = 2



optimizer = Adam(learning_rate=learning_rate)


running_reward = 0
episode_count = 0

model = ACModel(input_nodes, hidden_nodes, output_nodes)

reward_history = Memory()
rewards = Memory()

while True:
    state = env.reset()
    episode_reward = 0

    with GradientTape() as tape:
        for ts in range(1, max_ts):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)


            action_probs, critic_value = model(state)
            action = model.actor.get_action(action_probs)

            action_score = model.actor.compute_history_score(action_probs, action)
            critic_score = model.critic.compute_history_score(critic_value)

            model.actor.add_history(action_score)
            model.critic.add_history(critic_score)

            state, reward, done, _ = env.step(action)
            reward_history.add(reward)
            episode_reward += reward
            
            if done:
                break


        rewards.add(episode_reward)
        running_reward = 0.05 * episode_reward + (0.95) * running_reward
        

        returns = []
        discount_sum = 0


        for r in reward_history.data:
            discount_sum += (r + gamma)
            returns.append(discount_sum)


        returns = returns[::-1]
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)


        history = zip(model.actor.get_history(), model.critic.get_history(), returns)

        
        for log_prob, value, target in history:

            advantage = target - value

            actor_loss = model.actor.compute_loss(log_prob, advantage)
            critic_loss = model.critic.compute_loss(value, target)

            model.actor.add_loss(actor_loss)
            model.critic.add_loss(critic_loss)

        loss_value = model.actor.layer_losses.sum() + model.critic.layer_losses.sum()

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

       

        model.actor.clear_history()
        model.critic.clear_history()
        reward_history.clear()

    episode_count += 1
    if(episode_count % 10 == 0):
        print("running reward: {:.2f} at episode {}".format(running_reward, episode_count))
    
    if(running_reward >= 195):
        print("Solved at episode {}!".format(episode_count))
        model.save_weights("results/model/data.ckpt")

        rewards.save("results/progress.json")
        reward_history
        break