# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:58:23 2024
@author: hoshino
"""
# general packages
import numpy as np
import random

# keras
from keras.models import Sequential
from keras.layers import Dense, Activation

# PIRL agent
from rl_agent.DQN import RLagent, agentOptions
from rl_env.carla_env import CarEnv

# carla environment
class Env(CarEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def reset(self):
        carla_state = super().reset()
        horizon     = 5.0  # Fixed for all time
        self.state = np.array( list(carla_state) + [horizon] )        
        return self.state

    def step(self, action_idx):
        
        # make a step
        new_veh_state, reward, done = super().step( action_idx )
        horizon    = self.state[-1]
        new_state  = np.array( list(new_veh_state) + [horizon] )
        self.state = new_state

        return new_state, done        



def closed_loop_simulation(agent, env, T):

    current_state = env.reset()    
    state_trajectory = np.zeros([len(current_state), int(T/time_step)])
    for i in range(int(T/time_step)):
        
        action_idx = agent.get_epsilon_greedy_action(current_state)
        new_state, is_done = env.step(action_idx)
        
        current_state         = new_state   
        state_trajectory[:,i] = new_state
        print(new_state[0:3])


################################################################################################
# Main
if __name__ == '__main__':

    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &
    """    

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)

    ###########################
    # Environment
    carla_port = 3000
    time_step  = 0.05 
    map_for_testing = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/test.xodr"

    env    = Env(port=carla_port, time_step=time_step, 
                 custom_map_path = map_for_testing)
    actNum = env.action_num
    obsNum = len(env.reset())

    ############################
    # Load model

    data_dir = 'logs/test03122017'

    # Model (Use the same structure as during training)
    model = Sequential([
                Dense(32, input_shape=[obsNum, ]),
                Activation('tanh'), 
                Dense(32),  
                Activation('tanh'), 
                Dense(actNum),  
            ])
    
    agent  = RLagent(model, actNum, agentOptions())
    agent.load_weights(data_dir, ckpt_idx='latest')
        

    ######################################
    # Closed loop simulation
    T = 100
    try:  
        closed_loop_simulation(agent, env, T)
        
    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'env' in locals():
            env.destroy()
