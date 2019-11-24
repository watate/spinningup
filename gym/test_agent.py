import gym
import spinup
import tensorflow as tf
import spinup.utils.logx
from spinup.utils.test_policy import load_policy, run_policy


'''
# METHOD 1 (See Docs)
# BipedalWalker-v2
modelpath = '/home/watate/walter.spades@gmail.com/Python/spinningup/data/ex5_ddpg_100ep_bipedalwalker-v2/ex5_ddpg_100ep_bipedalwalker-v2_s0'

len = 0
episodes = 100
norender = False
#itr = -1
itr = -100

#Only for soft-actor critic
deterministic = False


# This part is unecessary because load_policy already restores tf_graph
#model = spinup.utils.logx.restore_tf_graph(sess, modelpath)


env, get_action = load_policy(modelpath, 
                                itr if itr >=0 else 'last',
                              deterministic)

run_policy(env, get_action, len, episodes, not(norender))
'''



# METHOD 2

# Cartpole-v0
#modelpath = "data/ppo-bench_cartpole-v0_hid64-64_relu/ppo-bench_cartpole-v0_hid64-64_relu_s20"

# Mountain Car Continuous
#modelpath = '/home/watate/walter.spades@gmail.com/Python/spinningup/data/ex2_ddpg_10ep_mountaincarcontinuous-v0/ex2_ddpg_10ep_mountaincarcontinuous-v0_s0' 

# Pendulum-v0
#modelpath = '/home/watate/walter.spades@gmail.com/Python/spinningup/data/ex3_ddpg_hid32_30ep_pendulum-v0/ex3_ddpg_hid32_30ep_pendulum-v0_s0'

# Acrobot-v1
#modelpath = '/home/watate/walter.spades@gmail.com/Python/spinningup/data/ex4_trpo_30ep_acrobot-v1_hid16-16/ex4_trpo_30ep_acrobot-v1_hid16-16_s0'

# BipedalWalker-v2
modelpath = '/home/watate/github/spinningup/data/ex5_ddpg_overnight_bipedalwalker-v2/ex5_ddpg_overnight_bipedalwalker-v2_s0'

episodes = 100
itr = -1

#Only for soft-actor critic
deterministic = False

env, get_action = load_policy(modelpath, 
                                itr if itr >=0 else 'last',
                              deterministic)


for i_episode in range(episodes):
    observation = env.reset()
    
    while(True): #for t in range(100):
        env.render()
        #print(observation)
        #action = env.action_space.sample()
        action = get_action(observation)

        #env.step returns these 4 variables
        observation, reward, done, info = env.step(action)
        
        #if done:
        #    print("Environment finished after {} timesteps".format(t+1))
        #    break
        
env.close()
