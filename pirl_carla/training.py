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
from pirl_agent.DQN import PIRLagent, agentOptions, train, trainOptions, pinnOptions
from rlenv.carla_env import CarEnv

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
            done   = True
            reward = 1
        
        return new_state, reward, done        

# Physics information
def convection_model(s_and_actIdx):

    s      = s_and_actIdx[:-1]
    actIdx = int(s_and_actIdx[-1]) 

    dxdt = np.zeros(15) 
    dsdt = np.concatenate([dxdt, np.array([-1])])
    
    return dsdt

def diffusion_model(x_and_actIdx):

    diagonals =  np.concatenate([0.2*np.ones(15), np.array([0])])
    sig  = np.diag(diagonals)
    diff = np.matmul( sig, sig.T )
 
    return diff

def sample_for_pinn():

    n_dim = 15 + 1
    T = 5
    x_vehicle_max = np.ones(15)
    x_vehicle_min = -np.ones(15)

    #######################
    # Interior points    
    nPDE  = 8
    x_max = np.array( list(x_vehicle_max) + [T] )
    x_min = np.array( list(x_vehicle_min) + [0] )
    X_PDE = x_min + (x_max - x_min)* np.random.rand(nPDE, n_dim)

    # Terminal boundary (at T=0 and safe)
    nBDini  = 8
    x_max = np.array( list(x_vehicle_max) + [0] )
    x_min = np.array( list(x_vehicle_min) + [0] )
    X_BD_TERM = x_min + (x_max - x_min) * np.random.rand(nBDini, n_dim)

    # Lateral boundary (unsafe set)        
    nBDsafe = 8
    x_max = np.array( list(x_vehicle_max) + [T] )
    x_min = np.array( list(x_vehicle_min) + [0] )
    X_BD_LAT = x_min + (x_max - x_min)* np.random.rand(nBDsafe, n_dim)
    X_BD_LAT[:,3] = np.random.choice([-2, 2], size=nBDsafe)    
    
    return X_PDE, X_BD_TERM, X_BD_LAT


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

    env    = Env(port=carla_port, time_step=time_step)
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
        OPTIMIZER  = Adam(learning_rate=0.01),
        REPLAY_MEMORY_SIZE = 5000, 
        REPLAY_MEMORY_MIN  = 100,
        MINIBATCH_SIZE     = 16,
        )
    
    pinnOp = pinnOptions(
        CONVECTION_MODEL = convection_model,
        DIFFUSION_MODEL  = diffusion_model,   
        SAMPLING_FUN     = sample_for_pinn,
        WEIGHT_PDE       = 0, 
        WEIGHT_BOUNDARY  = 1, 
        )
    
    agent  = PIRLagent(model, actNum, agentOp, pinnOp)


    ######################################
    # Training option

    #LOG_DIR = None
    LOG_DIR = 'logs/test'+datetime.now().strftime('%m%d%H%M')
    
    trainOp = trainOptions(
        EPISODES = 3000, 
        SHOW_PROGRESS = True, 
        LOG_DIR     = LOG_DIR,
        SAVE_AGENTS = False, 
        SAVE_FREQ   = 10,
        )

    ######################################
    # Train 
    train(agent, env, trainOp)

    