import glob
import os
import sys
import numpy as np

# Vehicle model
from params.carla_params import CarlaParams
from models.dynamic import Dynamic

# Carla Env
sys.path.append(os.pardir)
from rl_env.carla_env import CarEnv
 

SAMPLING_TIME = 0.05

#####################################################################
# load vehicle parameters

params = CarlaParams(control='pwm')
model = Dynamic(**params)



###############################################################################
# simulation
def main(env, T):
        
    # initialization
    current_state = env.reset()    
    state_trajectory   = np.zeros([len(current_state), int(T/env.time_step)])
    vehicle_trajectory = np.zeros([3, int(T/env.time_step)]) # (x,y,yaw)

    # iteration
    for i in range(int(T/env.time_step)):
 
        # step
        new_state, reward, is_done = env.step()
        
        # state
        state_trajectory[:,i] = new_state
        print(new_state[0:3])
 
        # position
        vehicle_trajectory[:,i] = env.get_vehicle_position()    
    
    return state_trajectory, vehicle_trajectory

        

###############################################################################
if __name__ == '__main__':
    
    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &
    """    

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

    env    = CarEnv(port=carla_port, time_step=time_step, 
                    custom_map_path = map_for_testing,
                    autopilot=True)

    try:  
        states, positions = main(env, T)

        


        
    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'env' in locals():
            env.destroy()

