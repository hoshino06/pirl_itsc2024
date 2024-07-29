# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:58:23 2024
@author: hoshino
"""
####################################
# general packages
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn

#######################################
# PIRL agent and CarEnv
#######################################
from rl_agent.PIRL_torch import PIRLagent, agentOptions, pinnOptions
from rl_env.carla_env   import CarEnv, map_c_before_corner, road_info_map_c_north_east
from training_pirl_MapC import convection_model, diffusion_model, sample_for_pinn

# carla environment
class Env(CarEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def reset(self):
        carla_state = super().reset()
        print(carla_state[0:3])
        horizon     = 5.0 # always reset with 5.0 (randomized in training)
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
###########################################
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


###############################################################################
# Main
###############################################################################
if __name__ == '__main__':

    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=5000 &
    """    
    ####################################
    # Settings
    ####################################
    data_dir    = "logs/MapC/07270815"
    check_point = "latest"
    
    log_dir     = 'plot/MapC/data_trained'
    video_save  = None, #'plot/MapC/simulation.mp4'

    carla_port = 5000
    time_step  = 0.05 
    map_train  = "./maps/train.xodr"
    spec_mapC_NorthEast = {'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0} 
    spec_mapC_NorthEast = {'x':-960, 'y':200, 'z':18, 'pitch':-70, 'yaw':130, 'roll':0} 

    #################################
    # Environment
    #################################
    def vehicle_reset_method_():
        x_loc    = 0
        y_loc    = 0 
        psi_loc  = 0  
        vx       = 30
        vy       = -vx*np.random.uniform( np.tan(20/180*3.14), np.tan(25/180*3.14))
        yaw_rate = np.random.uniform(65, 65)
        
        return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]        
    
    env    = Env(port=carla_port, time_step=time_step,
                 custom_map_path = map_train,
                 actor_filter    = 'vehicle.audi.tt',  
                 spawn_method    = map_c_before_corner, 
                 vehicle_reset   = vehicle_reset_method_,
                 waypoint_itvl   = 3.0,
                 spectator_init  = spec_mapC_NorthEast, 
                 spectator_reset = False, 
                 camera_save     = video_save,
                 )
    actNum = env.action_num
    obsNum = len(env.reset())    

    # spawn_points in MapC
    sp, left_pt, right_pt = road_info_map_c_north_east(env, 100)
    np.savez(log_dir+'/../spawn_points', 
             center=np.array(sp), left=np.array(left_pt), right=np.array(right_pt))

    ############################
    # Load model (Use the same structure as during training)
    ############################
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_stack = nn.Sequential(
                nn.Linear(obsNum, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
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
    pinnOp = pinnOptions(convection_model,diffusion_model,sample_for_pinn)        
    agent  = PIRLagent(model, actNum, agentOp, pinnOp)
    agent.load_weights(data_dir, ckpt_idx=check_point) 


    ###################################
    # Simulation
    ###################################
    random.seed(1)
    np.random.seed(1)

    slip_angles_all = []
    yaw_rates_all = []
    
    for i in range(20):
        print(f"==================Vehicle {i+1}==================")    
    
        T = 5

        states, positions, waypoints = closed_loop_simulation(agent, env, T)
  
        #np.savez(log_dir+f'/data{i}', state=states, position=positions)        
        slip_angle, yaw_rate = states[1, :], states[2, :]
        slip_angles_all.append(slip_angle)
        yaw_rates_all.append(yaw_rate)

        # Plot vehicle trajectory
        x = positions[0,:]# * 3 - 25
        y = positions[1,:]# + 1080
        plt.plot(y, x, color='blue', alpha=0.5, lw=0.5, label=f"vehicle {i+1}")
        plt.scatter(y[0], x[0], color='blue', marker='x')

    if 'env' in locals():
        env.destroy()

    #####################################
    # Plots
    #####################################
    plt.show()
    
    time_steps = np.arange(len(slip_angles_all[0])) * env.time_step
    
    # Plot slip angle
    plt.figure(figsize=(10, 5))
    for i, s in enumerate(slip_angles_all):
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
        plt.plot(time_steps, y, label=f'Yaw Rate {i+1}')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Yaw Rate (degrees/second)')
    plt.title('Yaw Rate vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    