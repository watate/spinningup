from spinup.utils.run_utils import ExperimentGrid
from spinup import vpg, trpo, ppo
import tensorflow as tf

'''
#Seed Test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    args = parser.parse_args()    

    eg = ExperimentGrid(name='ex3_seed')
    eg.add('env_name', 'Pendulum-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 5000)
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.run(ddpg, num_cpu=args.cpu)
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

    algo_names = ['vpg', 'trpo', 'ppo']
    algo = [vpg, trpo, ppo]

    for i in range(len(algo)):
        eg = ExperimentGrid(name='ex4_algotest_'+algo_names[i])
        eg.add('env_name', 'Acrobot-v1', '', True)
        eg.add('seed', [10*i for i in range(args.num_runs)])
        eg.add('epochs', 10)
        eg.add('steps_per_epoch', 5000)

        # Use default hidden sizes in actor_critic function, comment below out
        eg.add('ac_kwargs:hidden_sizes', [(32,)], 'hid')
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

    eg = ExperimentGrid(name='ex4_trpo_30ep')
    eg.add('env_name', 'Acrobot-v1', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 30)
    #eg.add('steps_per_epoch', 4000)
    eg.add('max_ep_len', 1500)
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.add('ac_kwargs:hidden_sizes', [(16,),(16,16),(8,),(8,8),(4,),(4,4)], 'hid')
    eg.run(trpo, num_cpu=args.cpu)

