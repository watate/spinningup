from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo, ddpg, td3
import tensorflow as tf

# Change these values to affect algos below
env_name = 'MountainCarContinuous-v0'
ex_number = 'ex2-6'

'''
#Seed Test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    args = parser.parse_args()    

    eg = ExperimentGrid(name=ex_number)
    eg.add('env_name', env_name, '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 5000)
    eg.add('ac_kwargs:hidden_sizes', [(64, 64)], 'hid')
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.run(td3, num_cpu=args.cpu)
'''


'''
#Algo Test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # This is set top auto and type str so that it can adapt to whatever algorithm im using
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    algo_names = ['ppo', 'ddpg', 'td3']
    algo = [ppo, ddpg, td3]

    for i in range(len(algo)):
        eg = ExperimentGrid(name=ex_number+'_algotest_'+algo_names[i])
        eg.add('env_name', env_name, '', True)
        eg.add('seed', [10*i for i in range(args.num_runs)])
        eg.add('epochs', 10)
        eg.add('steps_per_epoch', 5000)

        # Use default hidden sizes in actor_critic function, comment below out
        eg.add('ac_kwargs:hidden_sizes', [(16,16)], 'hid')
        eg.add('ac_kwargs:activation', [tf.nn.relu], '')
        
        eg.run(algo[i], num_cpu=args.cpu)
'''



#Training
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()    

    eg = ExperimentGrid(name=ex_number)
    eg.add('env_name', env_name, '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 5)
    #eg.add('steps_per_epoch', 4000)
    #eg.add('max_ep_len', 1500)
    eg.add('saveCheckpointEpochStep', 1)
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.add('ac_kwargs:hidden_sizes', [(8,8)], 'hid')
    eg.run(ddpg, num_cpu=args.cpu)

 
