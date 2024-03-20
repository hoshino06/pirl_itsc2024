import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Vehicle model
from params.carla_params import CarlaParams
from models.dynamic import Dynamic

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

        
def model_simulation(T, x0):

    # load vehicle parameters    
    params = CarlaParams(control='pwm')
    model = Dynamic(**params)
    
    model._diffequation(0,x0,[0.5,0])
    
    # 
    u = (np.array([[0.5], [0.1]])*np.ones([2,10]) )
    
    model.sim_continuous(x0, u, np.linspace(0,10,11))
    
    # initialization
    current_state = x0
    state_trajectory   = np.zeros([len(current_state), int(T/time_step)])
    vehicle_trajectory = np.zeros([3, int(T/time_step)]) # (x,y,yaw)

    # iteration
    for i in range(int(T/time_step)):
 
        # step
        new_state = x0
        
        # state
        state_trajectory[:,i] = new_state
        print(new_state[0:3])
 
        # position
        vehicle_trajectory[:,i] = [0,0,0]  
    
    return model, state_trajectory, vehicle_trajectory
    
    

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

    try:  

        # carla simuation
        rl_env, states, positions = carla_simulation(T)

        # model simulation
        x_pos = [-966.46466064, 521.85345459,  91.26826477]
        x_dyn = [10, 0, 0]
        x0    = x_pos + x_dyn

        #x0 = positions[:,0]
        model, states, positions = model_simulation(T, x0)

        # Plot vehicle trajectory
        #x = positions[0,:]
        #y = positions[1,:]
        #plt.plot(x, y, 'k', alpha=0.5, lw=0.5)
        #plt.scatter(x[0], y[0], color='red', marker='x')
        
    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()

