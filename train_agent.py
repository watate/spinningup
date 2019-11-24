from spinup import ppo
import tensorflow as tf
import gym

env_fn = lambda : gym.make('CartPole-v1')
ac_kwargs = dict(hidden_sizes = [64, 64], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='/data',exp_name = 'CartPole-v1')
ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=30, logger_kwargs=logger_kwargs)

#python -m spinup.run ppo --env CartPole-v1 --exp_name CartPole-v1

