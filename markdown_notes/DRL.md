# This document describes the key questions I have in DRL
# How to sample actions from a policy that I've trained?
## Tensorflow guide on how to save and load models: https://www.tensorflow.org/guide/saved_model
## How inputs and outputs are saved
```Python
# Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})
```

## Restore TF Function
```Python
def restore_tf_graph(sess, fpath):
    """
    Loads graphs saved by Logger.

    Will output a dictionary whose keys and values are from the 'inputs' 
    and 'outputs' dict you specified with logger.setup_tf_saver().

    Args:
        sess: A Tensorflow session.
        fpath: Filepath to save directory.

    Returns:
        A dictionary mapping from keys to tensors in the computation graph
        loaded from ``fpath``. 
    """
    tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                fpath
            )
    model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
    graph = tf.get_default_graph()
    model = dict()
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})
    return model
```

We use this function to restore a dictionary mapping keys to tensors in the computation graph
## Tensors in computation graph
### Example: PPO tensors
PPO Tensors can be found in the PPO definition. 
For example, these are some of the inputs to the PPO computation graph:
```Python
# Inputs to computation graph
x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)
```
Outputs from PPO computation graph:
```Python
# Main outputs from computation graph
pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)
```

## Save dictionary from restore_tf_graph
ppo_dict = spinup.utils.logx.restore_tf_graph(sess, modelpath)