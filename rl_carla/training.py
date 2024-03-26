# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:58:23 2024
@author: hoshino
"""
# general packages
import numpy as np
import random
from datetime import datetime

# keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

# PIRL agent
from rl_agent.DQN import RLagent, agentOptions, train, trainOptions
from rl_env.carla_env_accidental import CarEnv

# carla environment
class Env(CarEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def reset(self):
        carla_state = super().reset()
        horizon     = 5.0 * np.random.rand()
        self.state = np.array( list(carla_state) + [horizon] )        
        return self.state

    def step(self, action_idx):
        
        # make a step
        new_veh_state, reward, done = super().step( action_idx )
        horizon    = self.state[-1] - self.time_step
        new_state  = np.array( list(new_veh_state) + [horizon] )
        self.state = new_state

        # rewrite "reward" and "done" based on horizon
        if horizon <= 0:
            #print("horizon done")
            done   = True
            reward = 1
        
        return new_state, reward, done        


def choose_spawn_point(carla_env):
    sp_list = carla_env.get_all_spawn_points()    
    spawn_point = sp_list[0]
    return spawn_point

def random_spawn_point(carla_env):
    sp_list     = carla_env.get_all_spawn_points()       
    rand_1      = np.random.randint(0,len(sp_list))
    spawn_point = sp_list[rand_1]
    return spawn_point


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
    carla_port = 3706
    time_step  = 0.1
    map_for_training = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/train.xodr"
    map_for_testing  = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/test.xodr"

    env    = Env(port=carla_port, time_step=time_step,
                 custom_map_path = map_for_training,
                 spawn_method=random_spawn_point,
                 )
    actNum = env.action_num
    obsNum = len(env.reset())

    ############################
    # PIRL option    
    model = Sequential([
                Dense(32, input_shape=[obsNum, ]),
                Activation('tanh'), 
                Dense(32),  
                Activation('tanh'), 
                Dense(actNum),  
            ])
    
    agentOp = agentOptions(
        DISCOUNT   = 1, 
        OPTIMIZER  = Adam(learning_rate=1e-4), #0.0005
        REPLAY_MEMORY_SIZE = 5000, 
        REPLAY_MEMORY_MIN  = 100,
        MINIBATCH_SIZE     = 16,
        )    
   
    agent  = RLagent(model, actNum, agentOp)


    ######################################
    # Training option

    #LOG_DIR = None
    #LOG_DIR = 'logs/test'+datetime.now().strftime('%m%d%H%M')
    LOG_DIR = 'logs/test_25k_run1'
    
    trainOp = trainOptions(
        EPISODES = 25000, 
        SHOW_PROGRESS = True, 
        LOG_DIR     = LOG_DIR,
        SAVE_AGENTS = True, 
        SAVE_FREQ   = 500,
        SAVE_FREQ   = 10,
        )

    ######################################
    # Train 
    try:  
        train(agent, env, trainOp)
        
    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'env' in locals():
            env.destroy()

    