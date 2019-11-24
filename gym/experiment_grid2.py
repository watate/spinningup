# Experiment Grid 2 compares different DRL algorithms
# Important notes:
# This experiment does not force hidden sizes for the algorithms. In general they will
#   default to their own values. (Some are (64, 64) some are  (300,400))

from spinup.utils.run_utils import ExperimentGrid
from spinup import vpg, trpo, ppo, ddpg, td3, sac
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # This is set top auto and type str so that it can adapt to whatever algorithm im using
    parser.add_argument('--cpu', type=str, default='auto')
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    algo_names = ['vpg', 'trpo', 'ppo', 'ddpg', 'td3', 'sac']
    algo = [vpg, trpo, ppo, ddpg, td3, sac]

    for i in range(len(algo)):
        eg = ExperimentGrid(name=algo_names[i])
        eg.add('env_name', 'MountainCarContinuous-v0', '', True)
        eg.add('seed', [10*i for i in range(args.num_runs)])
        eg.add('epochs', 10)
        eg.add('steps_per_epoch', 4000)

        # Use default hidden sizes in actor_critic function, comment below out
        #eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
        eg.add('ac_kwargs:activation', [tf.nn.relu], '')
        
        eg.run(algo[i])#, num_cpu=args.cpu)

