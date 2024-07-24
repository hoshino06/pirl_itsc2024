# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:58:23 2024
@author: hoshino
"""
# general packages
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from torch import nn

# PIRL agent
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


def calculate_angle(vec1, vec2):
    unit_vector_1 = vec1 / np.linalg.norm(vec1)
    unit_vector_2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def plot_lane_boundaries(waypoints, lane_width):
    flattened_waypoints = [waypoint for sublist in waypoints for waypoint in sublist]
    
    left_boundaries = []
    right_boundaries = []
    prev_direction = None
    prev_pos = np.array([flattened_waypoints[0].location.x, flattened_waypoints[0].location.y])

    for i in range(1, len(flattened_waypoints)):
        current_pos = np.array([flattened_waypoints[i].location.x, flattened_waypoints[i].location.y])
        direction = current_pos - prev_pos
        
        if np.linalg.norm(direction) == 0:  # Skip if no movement or too far away
            continue

        if prev_direction is not None:
            angle_change = calculate_angle(prev_direction, direction)
            if angle_change > np.pi / 2:  # Discontinuity in direction
                continue

        prev_direction = direction
        prev_pos = current_pos
        
        perp_direction = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        
        left_boundary = current_pos + perp_direction * (lane_width / 2)
        right_boundary = current_pos - perp_direction * (lane_width / 2)
        
        left_boundaries.append(left_boundary.tolist())
        right_boundaries.append(right_boundary.tolist())

    # Convert to numpy array for spline fitting
    left_boundaries = np.array(left_boundaries)
    right_boundaries = np.array(right_boundaries)
    
    k = min(3, len(left_boundaries) - 1)  # Example: max degree 3
    
    tck, u = splprep([left_boundaries[:,1], left_boundaries[:,0]], s=0.5, k=k)  # s is a smoothing factor
    left_y, left_x = splev(np.linspace(0, 1, 100), tck)
    
    tck, u = splprep([right_boundaries[:,1], right_boundaries[:,0]], s=0.5, k=k)
    right_y, right_x = splev(np.linspace(0, 1, 100), tck)

    plt.figure(figsize=(8, 6))
    plt.plot(left_y, left_x, 'r', label='Left Boundary')
    plt.plot(right_y, right_x, 'r', label='Right Boundary')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title('25K trained agent trajectories')
    #plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()


################################################################################################
# Main
if __name__ == '__main__':

    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=5000 &
    """    

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)

    ###########################
    # Environment
    carla_port = 5000
    time_step  = 0.05 
    map_train  = "./maps/train.xodr"

    # vehicle state initialization
    def vehicle_reset_method_():
        # position and angle
        x_loc    = 0
        y_loc    = 0 
        psi_loc  = 0 #np.random.uniform(-20,20)
        # velocity and yaw rate
        vx       = 30 #np.random.uniform(15,25)
        rand_num = -0.8   #np.random.uniform(-0.75, -0.85)
        vy       = 0.5*vx*rand_num 
        yaw_rate = -80*rand_num 
        
        # It must return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]
        return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]        

    # Spectator_coordinate
    spec_town2 = {'x':-7.39, 'y':312, 'z':10.2, 'pitch':-20, 'yaw':-45, 'roll':0}    
    spec_mapC_NorthEast = {'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0} 

    #draw_lane_boundaries()
    untrained_slip_angles_all = []
    untrained_yaw_rates_all = []
    slip_angles_all = []
    yaw_rates_all = []
    
    env    = Env(port=carla_port, time_step=time_step,
                 custom_map_path = map_train,
                 actor_filter    = 'vehicle.audi.tt',  
                 spawn_method    = map_c_before_corner, #spawn_train_map_c_north_east,
                 vehicle_reset   = vehicle_reset_method_,
                 waypoint_itvl   = 3.0,
                 spectator_init  = spec_mapC_NorthEast, #None, 
                 spectator_reset = False, #True                  
                 )
    actNum = env.action_num
    obsNum = len(env.reset())    
    
    #######################################################################
    log_dir = 'plot/MapC/data_trained'

    # spawn_points in MapC
    sp, left_pt, right_pt = road_info_map_c_north_east(env, 100)
    np.savez(log_dir+'/../spawn_points', 
             center=np.array(sp), left=np.array(left_pt), right=np.array(right_pt))
       
    for i in range(20):
        print(f"==================Vehicle {i+1}==================")
    
        ############################
        # Load model
    
        data_dirs = [
            'logs/MapC/04251704'           
            #'./ITSC2024data/MapC/hoshino/04071140-19k'
            #'/home/ubuntu/extreme_driving/arnav/pirl_carla/ITSC2024data/MapC/hoshino/03250427'
            #'/home/ubuntu/extreme_driving/arnav/pirl_carla/ITSC2024data/MapC/hoshino/04030520'
            #'/home/arnav/extreme_driving/arnav/rl_carla/logs/test_25k_run1/', # use to plot 1k & 12.5k
            #'/home/arnav/extreme_driving/arnav/rl_carla/logs/test_25k_run_half_2/' # use to plot 25k
        ]

        # Model (Use the same structure as during training)
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
            REPLAY_MEMORY_SIZE = 5000, 
            REPLAY_MEMORY_MIN  = 1000,
            MINIBATCH_SIZE     = 32,
            EPSILON_INIT        = 1, #0.9998**20_000, 
            EPSILON_DECAY       = 0.9998, 
            EPSILON_MIN         = 0.01,
            )
        
        pinnOp = pinnOptions(
            CONVECTION_MODEL = convection_model,
            DIFFUSION_MODEL  = diffusion_model,   
            SAMPLING_FUN     = sample_for_pinn,
            WEIGHT_PDE       = 1e-4, 
            WEIGHT_BOUNDARY  = 1, 
            HESSIAN_CALC     = False,
            )    
        
        agent3  = PIRLagent(model, actNum, agentOp, pinnOp)
        agent3.load_weights(data_dirs[0], ckpt_idx='latest') 
    
        ######################################
        # Closed loop simulation
        T = 5

        '''states1, positions1 = closed_loop_simulation(agent1, env, T)
        states2, positions2 = closed_loop_simulation(agent2, env, T)'''
        states3, positions3, waypoints3 = closed_loop_simulation(agent3, env, T)
  
        np.savez(log_dir+f'/data{i}', state=states3, position=positions3)
   
        
        slip_angle, yaw_rate = states3[1, :], states3[2, :]
        slip_angles_all.append(slip_angle)
        yaw_rates_all.append(yaw_rate)

        # Plot vehicle trajectory
        x3 = positions3[0,:]# * 3 - 25
        y3 = positions3[1,:]# + 1080
        plt.plot(y3, x3, color='blue', alpha=0.5, lw=0.5, label=f"vehicle {i+1}")
        plt.scatter(y3[0], x3[0], color='blue', marker='x')

            

    if 'env' in locals():
        env.destroy()

    plt.show()
    
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
    
    