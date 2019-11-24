import tensorflow as tf
import numpy as np
from spinup.exercises.problem_set_1 import exercise1_1

"""

Exercise 1.2: PPO Gaussian Policy

Implement an MLP diagonal Gaussian policy for PPO. 

Log-likelihoods will be computed using your answer to Exercise 1.1,
so make sure to complete that exercise before beginning this one.

"""

EPS = 1e-8

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """
    Builds a multi-layer perceptron in Tensorflow.

    Args:
        x: Input tensor.

        hidden_sizes: Tuple, list, or other iterable giving the number of units
            for each hidden layer of the MLP.

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer.

    Returns:
        A TF symbol for the output of an MLP that takes x as an input.

    """
    #######################
    #                     #
    #   YOUR CODE HERE    #
    #                     #
    #######################

    #New code (after referring to answers)
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

    #Old code
    '''
    #Number of columns = number of inputs
    input_layer_size = np.shape(x)[0]
    
    #Initialize weights and bias
    W1 = np.random.randn(hidden_sizes[0], input_layer_size) * 0.01
    b1 = np.zeros((hidden_sizes[0], 1))
    W2 = np.random.randn(np.shape(x)[1], hidden_sizes[0]) * 0.01
    b2 = np.zeros((np.shape(x)[1], 1))

    #Build graph
    layer_1 = tf.placeholder(tf.float32, shape=(hidden_sizes[0],))

    #Hidden layer
    layer_1 = tf.add(tf.matmul(W1, x), b1)
    layer_1 = activation(layer_1)

    #Output layer
    out_layer = tf.add(tf.matmul(W2, layer_1), b2)
    if output_activation == None:
        return out_layer
    else:
        out_layer = output_activation(out_layer)
        return out_layer
    '''

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    """
    Builds symbols to sample actions and compute log-probs of actions.

    Special instructions: Make log_std a tf variable with the same shape as
    the action vector, independent of x, initialized to [-0.5, -0.5, ..., -0.5].

    Args:
        x: Input tensor of states. Shape [batch, obs_dim].

        a: Input tensor of actions. Shape [batch, act_dim].

        hidden_sizes: Sizes of hidden layers for action network MLP.

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer (action layer).

        action_space: A gym.spaces object describing the action space of the
            environment this agent will interact with.

    Returns:
        pi: A symbol for sampling stochastic actions from a Gaussian 
            distribution.

        logp: A symbol for computing log-likelihoods of actions from a Gaussian 
            distribution.

        logp_pi: A symbol for computing log-likelihoods of actions in pi from a 
            Gaussian distribution.

    """
    #######################
    #                     #
    #   YOUR CODE HERE    #
    #                     #
    #######################
    #New code
    #get dimensions of actions
    act_dim = a.shape[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name="log_std", initializer=-0.5*np.ones(shape=act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std

    #Old code
    '''
    mu, var = tf.nn.moments(a, axes=[1, np.shape(a)[1]])
    log_std = tf.Variable(-0.5, shape=tf.shape(a))
    pi = exercise1_1.gaussian_likelihood(a, mu, log_std)
    '''

    logp = exercise1_1.gaussian_likelihood(a, mu, log_std)
    logp_pi = exercise1_1.gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """

    from spinup import ppo
    from spinup.exercises.common import print_result
    import gym
    import os
    import pandas as pd
    import psutil
    import time

    logdir = "/tmp/experiments/%i"%int(time.time())
    ppo(env_fn = lambda : gym.make('InvertedPendulum-v2'),
        ac_kwargs=dict(policy=mlp_gaussian_policy, hidden_sizes=(64,)),
        steps_per_epoch=4000, epochs=20, logger_kwargs=dict(output_dir=logdir))

    # Get scores from last five epochs to evaluate success.
    data = pd.read_table(os.path.join(logdir,'progress.txt'))
    last_scores = data['AverageEpRet'][-5:]

    # Your implementation is probably correct if the agent has a score >500,
    # or if it reaches the top possible score of 1000, in the last five epochs.
    correct = np.mean(last_scores) > 500 or np.max(last_scores)==1e3
    print_result(correct)