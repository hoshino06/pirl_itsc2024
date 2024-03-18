import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Carla Env
sys.path.append(os.pardir)
from rl_env.carla_env import CarEnv


###############################################################################
# simulation
def carla_simulation(T):

    # carla_env
    rl_env = CarEnv(port=carla_port, time_step=time_step, 
                    custom_map_path = map_for_testing,
                    spawn_method=choose_spawn_point,
                    autopilot=True)

        
    # initialization
    current_state = rl_env.reset()    
    state_trajectory   = np.zeros([len(current_state), int(T/rl_env.time_step)])
    vehicle_trajectory = np.zeros([3, int(T/rl_env.time_step)]) # (x,y,yaw)

    # iteration
    for i in range(int(T/rl_env.time_step)):
 
        # step
        new_state, reward, is_done = rl_env.step()
        
        # state
        state_trajectory[:,i] = new_state
        print(new_state[0:3])
 
        # position
        vehicle_trajectory[:,i] = rl_env.get_vehicle_position()    
    
    return rl_env, state_trajectory, vehicle_trajectory

 
def choose_spawn_point(carla_env):
    sp_list = carla_env.get_all_spawn_points()    
    spawn_point = sp_list[0]
    return spawn_point

def vehicle_reset_method(): 
    # position and angle
    x_loc    = 0
    y_loc    = 0 #np.random.uniform(-5,5)
    psi_loc  = 0
    # velocity and yaw rate
    vx = 10
    vy = 0
    yaw_rate = 0                
    return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]


###############################################################################
if __name__ == '__main__':
    
    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &
    """    

    SAMPLING_TIME = 0.05
    T = 10

    ###########################
    # Environment
    carla_port = 3000
    time_step  = 0.05 
    map_for_testing = "/home/ubuntu/carla/carla_drift_0_9_5/CarlaUE4/Content/Carla/Maps/OpenDrive/test_refined.xodr"

    try:  

        # carla simuation
        rl_env, states, positions = carla_simulation(T)

        # Plot vehicle trajectory
        x = positions[0,:]
        y = positions[1,:]
        plt.plot(x, y, 'k', alpha=0.5, lw=0.5)
        plt.scatter(x[0], y[0], color='red', marker='x')
        
    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'env' in locals():
            rl_env.destroy()
