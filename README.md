# Actor-Critic-Cartpole
An A2C implementation for Cartpole




### Hyperparameters

Seed = 15
Gamme = 0.95
MAX_TIMESTEPS = 200
LEARNING_RATE = 0.005
Optimizer = Adam

input_nodes = 4
hidden_nodes = 128
output_nodes = 2


### Model Structure


I used subclassing from tf.keras.Model to create a model for the Actor and Critic with a shared hidden layer. The inputs will be fed into the shared hidden layer, then passed on the actor and critic models separately. Gradient Descent and Optimization is applied on this model to make it learn.


### Progress

The model converges in around 300 episodes, which is pretty good compared to the 1000 episodes in my [DQN model](https://github.com/SatvikVejendla/Cartpole-DQN).

The progress of the model during its training phase can be shown below.

![Progress](./results/progress.png?raw=true)
