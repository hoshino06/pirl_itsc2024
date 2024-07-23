# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:58:23 2024
@author: hoshino
"""
# general packages
import numpy as np
import random
from datetime import datetime

from torch import nn
from torch.optim import Adam

# PIRL agent
from rl_agent.PIRL_torch import PIRLagent, agentOptions, train, trainOptions, pinnOptions
from rl_env.carla_env import CarEnv, spawn_train_map_c_north_east

# carla environment
class Env(CarEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def reset(self):
        carla_state = super().reset()
        horizon     = 5.0 * np.random.rand()
        self.state = np.array( list(carla_state) + [horizon] )        
        return self.state

    def step(self, action_idx=0):
        
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

###############################################################################
# Physics information
def convection_model(s_and_actIdx):
 
    # arg parse
    s      = s_and_actIdx[:-1]

    x     = s[:-1]
    vx    = x[0]            # m/s
    beta  = x[1]*(3.14/180) # deg -> rad
    vy    = vx*np.tan(beta)
    omega = x[2]*(3.14/180) # deg/s -> rad/s  
    lat_e = x[3]            # m
    psi   = x[4]*(3.14/180) # deg -> rad              

    actIdx = int(s_and_actIdx[-1]) 
    step_T_pool = np.linspace(0.6,1, num=5)    # throttle values (see carla_env)
    step_S_pool = np.linspace(-0.8,0.8, num=5) # steering angle values (see carla_env)
    throttleID = int(actIdx / len(step_S_pool))
    steerID    = int(actIdx % len(step_S_pool))
    throttle   = step_T_pool[throttleID]
    steer      = step_S_pool[steerID]
    steer      = steer * 70 * (3.14/180) # [-1,1] -> [-70deg,70deg] -> [-1.2rad, 1.2rad]
    
    # Parameters
    lf = 1.34
    lr = 1.3
    mass = 1265
    Iz = 2093
    Bf = 5.579
    Cf = 1.2
    Df = 16000
    Br = 7.579
    Cr = 1.2
    Dr = 16000

    Cm1 = 550*(3.45*0.919)/(0.34)
    Cm2 = 0 
    Cr0 = 50.
    Cr2 = 0.5

    # Model equation
    dxdt = np.zeros(15) 
    Frx  = (Cm1-Cm2*vx)*throttle - Cr0 - Cr2*(vx**2)
    alphaf = steer - np.arctan2((lf*omega + vy), abs(vx))
    alphar = np.arctan2((lr*omega - vy), abs(vx))
    Ffy    = Df * np.sin(Cf * np.arctan(Bf * alphaf))
    Fry    = Dr * np.sin(Cr * np.arctan(Br * alphar))

    dxdt[0] = 1/mass * (Frx - Ffy*np.sin(steer)) + vy*omega    #vx
    dxdt[1] = 1/(mass*vx) * (Fry + Ffy*np.cos(steer)) - omega  #beta
    dxdt[2] = 1/Iz *   (Ffy*lf*np.cos(steer) - Fry*lr)         #omega
    dxdt[3] = vy*np.cos(psi) + vx*np.sin(psi)                  #lat_error
    dxdt[4] = omega                                            #psi

    dsdt = np.concatenate([dxdt, np.array([-1])])
    
    return dsdt


def diffusion_model(x_and_actIdx):

    diagonals = np.concatenate([ 0.01*np.ones(5), 0*np.ones(10), np.array([0])])
    sig  = np.diag(diagonals)
    diff = np.matmul( sig, sig.T )
 
    return diff

def sample_for_pinn(replay_memory):

    n_dim = 15 + 1
    T    = 5
    Emax = 0.95
    x_vehicle_max = np.concatenate( [np.array([20, 10, 30]+[ Emax,  30]), np.ones(10)*10] )
    x_vehicle_min = np.concatenate( [np.array([ 5,-10,-30]+[-Emax, -30]),-np.ones(10)*10] )

    #######################
    # Interior points    
    nPDE  = 32
    x_max = np.array( list(x_vehicle_max) + [T] )
    x_min = np.array( list(x_vehicle_min) + [0] )
    X_PDE = x_min + (x_max - x_min)* np.random.rand(nPDE, n_dim)
    sample_state = [replay_memory[np.random.randint(0,len(replay_memory))][0]\
                    for i in range(nPDE)]
    X_PDE[:,5: -1] = np.asarray(sample_state)[:, 5:-1]
    assert X_PDE.shape == (nPDE, n_dim)

    # Terminal boundary (at T=0 and safe)
    nBDini  = 32
    x_max = np.array( list(x_vehicle_max) + [0] )
    x_min = np.array( list(x_vehicle_min) + [0] )
    X_BD_TERM = x_min + (x_max - x_min) * np.random.rand(nBDini, n_dim)
    sample_state = [replay_memory[np.random.randint(0,len(replay_memory))][0]\
                    for i in range(nBDini)]
    #sample_state[:, 15] = 0
    X_BD_TERM[:,5: 15] = np.asarray(sample_state)[:, 5:15]
    assert X_BD_TERM.shape == (nBDini, n_dim)

    # Lateral boundary (unsafe set)        
    nBDsafe = 32
    x_max = np.array( list(x_vehicle_max) + [T] )
    x_min = np.array( list(x_vehicle_min) + [0] )
    Emax = 1.0 
    X_BD_LAT = x_min + (x_max - x_min)* np.random.rand(nBDsafe, n_dim)
    X_BD_LAT[:,3] = np.random.choice([-Emax, Emax], size=nBDsafe)    
    sample_state = [replay_memory[np.random.randint(0,len(replay_memory))][0]\
                    for i in range(nBDsafe)]
    X_BD_LAT[:,5: 15] = np.asarray(sample_state)[:, 5:15]
    X_BD_LAT[:,3] = np.random.choice([-Emax, Emax], size=nBDsafe)
    assert X_BD_LAT.shape == (nBDsafe, n_dim)
      
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

    ###########################################################################
    # Environment
    carla_port = 4000
    time_step  = 0.05    
    map_train  = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/train.xodr"
    restart    = False

    # spawn method (initial vehicle location)
    def random_spawn_point(carla_env):
        sp_list     = carla_env.get_all_spawn_points()       
        rand_1      = np.random.randint(0,len(sp_list))
        spawn_point = sp_list[rand_1]
        return spawn_point

    # vehicle state initialization
    def vehicle_reset_method(): 
        
        # position and angle
        x_loc    = 0
        y_loc    = np.random.uniform(-0.5,0.5) #np.random.uniform(-0.8,0.8)
        psi_loc  = 0 #np.random.uniform(-20,20)
        # velocity and yaw rate
        vx = np.random.uniform(10,15)
        vy = 0 #0.5* float(vx * np.random.rand(1)) 
        yaw_rate = 0 #np.random.uniform(-360,360)       
        
        # It must return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]
        return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]

    # Spectator_coordinate
    spec_town2 = {'x':-7.39, 'y':312, 'z':10.2, 'pitch':-20, 'yaw':-45, 'roll':0}    
    spec_mapC_NorthEast = {'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0} 

    env    = Env(port=carla_port, time_step=time_step,
                 custom_map_path = None, #map_train,
                 actor_filter    = 'vehicle.audi.tt',  
                 spawn_method    = None, #spawn_train_map_c_north_east,
                 vehicle_reset   = vehicle_reset_method,                  
                 spectator_init  = spec_town2, #None, #None
                 spectator_reset = False, #True 
                 )
    actNum = env.action_num
    obsNum = len(env.reset())

    ###########################################################################
    # PIRL option   
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_stack = nn.Sequential(
                nn.Linear(obsNum, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, actNum),
                nn.Sigmoid()
                )
        def forward(self, x):
            output = self.linear_stack(x)
            return output    
    model = NeuralNetwork().to('cpu')
    
    agentOp = agentOptions(
        DISCOUNT   = 1, 
        OPTIMIZER  = Adam(model.parameters(), lr=5e-4),
        REPLAY_MEMORY_SIZE = 5000, 
        REPLAY_MEMORY_MIN  = 1000,
        MINIBATCH_SIZE     = 32,
        EPSILON_INIT        = 1, 
        EPSILON_DECAY       = 0.9998, 
        EPSILON_MIN         = 0.01, #0.01
        )
    
    pinnOp = pinnOptions(
        CONVECTION_MODEL = convection_model,
        DIFFUSION_MODEL  = diffusion_model,   
        SAMPLING_FUN     = sample_for_pinn,
        WEIGHT_PDE       = 1e-3, #1e-4 
        WEIGHT_BOUNDARY  = 1, 
        HESSIAN_CALC     = False,
        )
    
    agent  = PIRLagent(model, actNum, agentOp, pinnOp)
    #agent.load_weights('logs/Town2/03192036', ckpt_idx='latest')

    ######################################
    # Training option
    
    if restart == True:
        LOG_DIR = "logs/Town2/03201826/"
        ckp_path = agent.load_weights(LOG_DIR, ckpt_idx='latest')
        current_ep = int(ckp_path.split('-')[-1].split('.')[0])
        print(current_ep)
    else:
        LOG_DIR = 'logs/Town2/'+datetime.now().strftime('%m%d%H%M')
        current_ep = None
        
    #LOG_DIR = None
        
    """
    $ tensorboard --logdir logs/...
    """
    
    trainOp = trainOptions(
        EPISODES = 50_000, 
        SHOW_PROGRESS = True, 
        LOG_DIR     = LOG_DIR,
        SAVE_AGENTS = True, 
        SAVE_FREQ   = 1000,
        RESTART_EP  = current_ep
        )
    agentOp['RESTART_EP'] = current_ep

    ######################################
    # Train 
    try:  
        train(agent, env, trainOp)
        
    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'env' in locals():
            env.destroy()
