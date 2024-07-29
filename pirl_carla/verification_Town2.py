# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:58:23 2024
@author: hoshino
"""
####################################
# general packages
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn

#######################################
# PIRL agent and CarEnv
sys.path.append(os.pardir)
sys.path.append('.')

from rl_env.carla_env    import CarEnv
from rl_agent.PIRL_torch import PIRLagent, agentOptions, pinnOptions
from training_pirl_Town2 import convection_model, diffusion_model, sample_for_pinn

# carla environment
class Env(CarEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def reset(self):
        carla_state = super().reset()
        print(carla_state[0:3])
        horizon     = 6.0 # always reset with 5.0 (randomized in training)
        self.state = np.array( list(carla_state) + [horizon] )        
        return self.state

    def step(self, action_idx):
        
        # make a step
        new_veh_state, reward, done = super().step( action_idx )
        #horizon    = self.state[-1]
        horizon    = self.state[-1] - self.time_step
        new_state  = np.array( list(new_veh_state) + [horizon] )
        self.state = new_state

        return new_state, reward, done

###########################################
# Simulation function
def closed_loop_simulation(agent, env, T):
    
    # initialization
    current_state = env.reset()    
    state_trajectory   = np.zeros([len(current_state), int(T/env.time_step)])
    vehicle_trajectory = np.zeros([3, int(T/env.time_step)]) # (x,y,yaw)
    waypoints = []
    
    for i in range(int(T/time_step)):
        
        action_idx = agent.get_qs(current_state).argmax().numpy()
        new_state, _, is_done = env.step(action_idx)
        # get waypoint from new_state
        vehicle_locat = env.vehicle.get_transform().location
        way_point = env.world.get_map().get_waypoint(vehicle_locat, project_to_road=True)
        
        right_way_point = way_point.get_right_lane()
        left_way_point = way_point.get_left_lane()
        way_point = right_way_point if right_way_point.lane_width > left_way_point.lane_width else left_way_point

        _, wps = env.fetch_relative_states(way_point.transform, 0.5, 5) #0.5, 5
        
        waypoints.append(wps)
        
        current_state         = new_state   
        state_trajectory[:,i] = new_state
        #print(new_state[0:3])
        
        # state
        state_trajectory[:,i] = new_state # index 1 and 2
        #print(new_state[0:3])
 
        # position
        vehicle_trajectory[:,i] = env.get_vehicle_position()    
        
    return state_trajectory, vehicle_trajectory, waypoints


################################################################################
# Main
################################################################################
if __name__ == '__main__':

    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &
    """

    ####################################
    # Settings
    ####################################
    data_dir    = 'ITSC2024data/Town2/04291642'
    #data_dir    = 'logs/Town2/07231829'
    check_point = 'latest'

    carla_port = 3000
    time_step  = 0.05 
    video_save = None #'plot/Town2/simulation.mp4'
    spec_town2 = {'x':-7.39, 'y':312, 'z':10.2, 'pitch':-20, 'yaw':-45, 'roll':0}  #spectator coordinate  

    #################################
    # Environment
    #################################
    def choose_spawn_point(carla_env):
        sp_list = carla_env.get_all_spawn_points()    
        spawn_point = sp_list[1]
        return spawn_point
    
    def vehicle_reset_method(): 
        x_loc    = 0
        y_loc    = 0  #np.random.uniform(-0.5,0.5) 
        psi_loc  = 0  #np.random.uniform(-20,20)
        vx       = np.random.uniform(10,10)
        vy       = 0 
        yaw_rate = 0   
        return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]
    
    env    = Env(port=carla_port, time_step=time_step,
                 custom_map_path = None,
                 actor_filter    = 'vehicle.audi.tt',  
                 spawn_method    = choose_spawn_point,
                 vehicle_reset   = vehicle_reset_method,
                 waypoint_itvl   = 0.5,
                 spectator_init  = spec_town2,  
                 spectator_reset = False,
                 camera_save     = video_save,
                 )
    actNum = env.action_num
    obsNum = len(env.reset())    

    #################################################
    # Model (Use the same structure as during training)
    #################################################
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
    agentOp = agentOptions()
    pinnOp = pinnOptions(convection_model, diffusion_model, sample_for_pinn)    
    agent3  = PIRLagent(model, actNum, agentOp, pinnOp)
    agent3.load_weights(data_dir, ckpt_idx='latest') 

    ######################################
    # Closed loop simulation
    ######################################
    random.seed(1)
    np.random.seed(1)
    slip_angles_all = []
    yaw_rates_all = []
    for i in range(5):
        print(f"==================Vehicle {i+1}==================")
        T = 5
        states3, positions3, waypoints3 = closed_loop_simulation(agent3, env, T)
        #np.savez(log_dir+f'/data{i}', state=states3, position=positions3)
        slip_angle, yaw_rate = states3[1, :], states3[2, :]
        slip_angles_all.append(slip_angle)
        yaw_rates_all.append(yaw_rate)
        x3 = positions3[0,:]# * 3 - 25
        y3 = positions3[1,:]# + 1080
        plt.plot(y3, x3, color='blue', alpha=0.5, lw=0.5, label=f"vehicle {i+1}")
        plt.scatter(y3[0], x3[0], color='blue', marker='x')

    if 'env' in locals():
        env.destroy()

    #####################################
    # Plot slip angle and yaw rate
    #####################################
    
    time_steps = np.arange(len(slip_angles_all[0])) * env.time_step
    
    # Plot slip angle
    plt.figure(figsize=(10, 5))
    for i, s in enumerate(slip_angles_all):
        #smoothed_s = pd.Series(s).rolling(window=5).mean()
        #plt.plot(time_steps, smoothed_s, label=f'Slip Angle {i+1}')
        plt.plot(time_steps, s, label=f'Slip Angle {i+1}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Slip Angle (degrees)')
    plt.title('Slip Angle vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot yaw rate
    plt.figure(figsize=(10, 5))
    for i, y in enumerate(yaw_rates_all):
        #smoothed_y = pd.Series(y).rolling(window=5).mean()
        #plt.plot(time_steps, smoothed_y, label=f'Yaw Rate {i+1}')
        plt.plot(time_steps, y, label=f'Yaw Rate {i+1}')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Yaw Rate (degrees/second)')
    plt.title('Yaw Rate vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    