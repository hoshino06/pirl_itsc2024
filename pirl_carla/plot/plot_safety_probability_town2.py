"""
Plot safe probability vs map
"""
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


# PIRL agent and CarEnv
sys.path.append(os.pardir)
from rl_agent.PIRL_DQN import PIRLagent, agentOptions, pinnOptions
from training_pirl_Town2 import Env, convection_model, diffusion_model, sample_for_pinn

###########################################################################
# Load PIRL agent and carla environment     
def load_agent(env):
 
    actNum = env.action_num
    obsNum = len(env.reset())

    model = Sequential([
                Dense(32, input_shape=[obsNum, ]),
                Activation('tanh'), 
                Dense(32),  
                Activation('tanh'), 
                Dense(actNum),  
            ])
    
    agentOp = agentOptions(
        DISCOUNT   = 1, 
        OPTIMIZER  = Adam(learning_rate=1e-4),
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
        WEIGHT_PDE       = 1e-5, 
        WEIGHT_BOUNDARY  = 1, 
        HESSIAN_CALC     = False,
        )    
    
    agent  = PIRLagent(model, actNum, agentOp, pinnOp)
    agent.load_weights('../logs/Town2/03201529', ckpt_idx='latest')
    
    return agent


###############################################################################
if __name__ == '__main__':
    
    ###########################
    # Get nominal trajectory
    carla_port = 5000
    time_step  = 0.05    
    spec_town2 = {'x':-7.39, 'y':312, 'z':10.2, 'pitch':-20, 'yaw':-45, 'roll':0}    

    try:  

        # Get reference state
        def choose_spawn_point(carla_env):
            sp_list = carla_env.get_all_spawn_points()    
            spawn_point = sp_list[1]
            return spawn_point
        
        # carla_env
        rl_env = Env(port=carla_port, time_step=time_step, 
                        custom_map_path = None,
                        spawn_method    = choose_spawn_point,
                        spectator_init  = spec_town2, 
                        spectator_reset = False, 
                        autopilot       = True)

        rl_env.reset()
        rl_env.step()
        x_vehicle = rl_env.getVehicleState()
        x_road    = rl_env.getRoadState() 

        horizon   = 3
        x_vehicle = np.array( rl_env.getVehicleState() )
        x_road    = np.array( rl_env.getRoadState() )
        s         = np.concatenate([ x_vehicle, x_road, np.array([horizon]) ])

        print(f'x_vehicle: {x_vehicle}')
        print(f'x_vehicle: {x_road}')

        ##############################
        
        agent = load_agent(rl_env)
                
        e_list = np.linspace(-1.1, 1.1, 20)
        v_list = np.zeros(len(e_list))
        for i, e in enumerate(e_list):
            
            x_vehicle = np.array( [10, 0, 0] )
            x_road    = np.array( rl_env.getRoadState() )
            x_road[0] = e
            x_road[1] = 0
            s         = np.concatenate([ x_vehicle, x_road, np.array([horizon]) ])                        
            
            v_list[i] = np.max(agent.get_qs(s))    

        plt.plot(e_list, v_list, 'k', alpha=0.5, lw=0.5)
        plt.xlabel('Lateral error [m]')
        plt.ylabel('Safety probability')

    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()
        
    



