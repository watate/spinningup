import gym
import spinup
import tensorflow as tf
import spinup.utils.logx
from spinup.utils.test_policy import load_policy, run_policy


'''
# METHOD 1 (See Docs)

modelpath = "data/ppo-bench_cartpole-v0_hid64-64_relu/ppo-bench_cartpole-v0_hid64-64_relu_s20"
len = 0
episodes = 100
norender = False
#itr = -1
itr = 'last'

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


modelpath = '/home/watate/walter.spades@gmail.com/Python/spinningup/data/ex3_ddpg_hid32_30ep_pendulum-v0/ex3_ddpg_hid32_30ep_pendulum-v0_s0'

episodes = 100
itr = -1

#Only for soft-actor critic
deterministic = False

env, get_action = load_policy(modelpath, 
                                itr if itr >=0 else 'last',
                              deterministic)

done = False

for i_episode in range(episodes):
    observation = env.reset()
    
    while(True): #for t in range(100):
        env.render()
        #print(observation)

        if done:
            action = env.action_space.sample()
        else:
            action = get_action(observation)

        #env.step returns these 4 variables
        observation, reward, done, info = env.step(action)
        
        #if done:
            #print("Environment finished after {} timesteps".format(t+1))
            #break
        
env.close()
