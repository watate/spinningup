from spinup.utils.run_utils import ExperimentGrid
from spinup import td3
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=str, default='auto')
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    x = 10
    layers = []
    layers_itr = []
    for i in range(x):
        layers.append(64)
        layers_itr.append(list(layers))
    layers = []           
    for i in range(x):
        layers.append(128)
        layers_itr.append(list(layers))
    layers = []           
    for i in range(x):
        layers.append(256)
        layers_itr.append(list(layers))
    layers = []           
    for i in range(x):
        layers.append(512)
        layers_itr.append(list(layers))          

    eg = ExperimentGrid(name='td3-bench')
    eg.add('env_name', 'MountainCarContinuous-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', layers_itr, 'hid')
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.run(td3, num_cpu=args.cpu)