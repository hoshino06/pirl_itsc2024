# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:05:29 2023

@author: hoshino
"""
import random
import numpy as np
from tqdm import tqdm # for showing progress bar

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from carla_env import CarEnv
from dqn_agent import DQNAgent

import numpy as np
import cv2 # openCV for camera
import queue

##############################################################################
# main algorithm
##############################################################################
def main_alg(env, agent):
    
    # Settings
    epsilon = 1
    EPSILON_DECAY = 0.9995 ## 0.9975 99975
    MIN_EPSILON = 0.001

    # For stats
    ep_rewards = []
    average_rewards = []
    q_values = []
    
    # Iterate over episodes
    EPISODES = 5000
    AGGREGATE_STATS_EVERY = 20 # episodes   
    
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        
        #env.vehicle = env.world.spawn_actor()
        # Restarting episode
        episode_reward  = 0
        outlook_horizon = 5.0 * np.random.rand()
        current_state  = np.array( list( env.reset()) + [outlook_horizon] )
        
        q_val = np.max(agent.get_qs(current_state))
        q_values.append(q_val)

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # get action id
            if np.random.random() > epsilon:
                # Greedy action from Q network
                action_idx = np.argmax(agent.get_qs(current_state))
            else:
                # Random action
                action_idx = np.random.randint(0, action_num)  
            
            # make a step
            new_veh_state, reward, done = env.step( action_idx )     
            horizon = current_state[-1] - env.time_step
            new_state = np.array( list(new_veh_state) + [horizon] )                    

            # get reward if             
            if horizon < 0:
                done   = True
                reward = 100
            
            # add reward
            episode_reward += reward

            # update replay memory and train main network
            agent.update_replay_memory((current_state, action_idx, reward, new_state, done))
            agent.train( is_terminal_state=done )

            # update current state
            current_state = new_state
        
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        # Log stats     
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
            average_rewards.append(average_reward)

    return agent, ep_rewards, average_rewards, q_values



###############################################################################
# main run
##############################################################################
if __name__ == '__main__': 

    #####################
    # Settings
    #####################
    carla_port = 3000
    time_step  = 0.05

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    #tf.random.set_seed(1) # for Tensorflow 2.x    
    
    #######################
    # Run rl algorithm
    #######################
    try:        
        # Create environment
        carla_env = CarEnv(port=carla_port, time_step=time_step)

        # Initialize agent
        obs_dim    = carla_env.state_dim + 1  # State augmentation
        action_num = carla_env.action_num   
        agent = DQNAgent(obs_dim, action_num)

        # Train
        _, ep_rewards, avg_rewards, qs = main_alg(env=carla_env, agent=agent)
        #print("\n\n\n", ep_rewards, avg_rewards)

        
    except KeyboardInterrupt:
        print('\nCancelled by user - carla-risk-prob.py.')

    finally:
        if 'carla_env' in locals():
            carla_env.destroy()
            
        fig, axs = plt.subplots(1, 2)
            
        '''if len(ep_rewards) > 0:
            X = np.linspace(start=0, stop=10, num=len(ep_rewards))
            axs[0].set_xlabel("iteration")
            axs[0].set_ylabel("ep_reward")
            
            #plt.subplot(1, 3, 1)
            axs[0].plot(X, ep_rewards)'''
            
        if len(avg_rewards) > 0:
            X = np.linspace(start=0, stop=500, num=len(avg_rewards))
            axs[0].set_xlabel("episode")
            axs[0].set_ylabel("avg_reward")
            
            #plt.subplot(1, 3, 2)
            axs[0].plot(X, avg_rewards)


        if len(qs) > 0:
            X2 = np.linspace(start=0, stop=500, num=len(qs))
            axs[1].set_xlabel("episode")
            axs[1].set_ylabel("q_value")
            
            #plt.subplot(1, 3, 3)
            axs[1].plot(X2, qs)

            plt.show()
    
