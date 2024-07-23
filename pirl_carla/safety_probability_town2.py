"""
Plot safe probability vs map
"""
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# PIRL agent and CarEnv
sys.path.append(os.pardir)
sys.path.append('.')

from rl_agent.PIRL_torch import PIRLagent, agentOptions, pinnOptions
from training_pirl_Town2 import Env, convection_model, diffusion_model, sample_for_pinn


###########################################################################
# Settings

carla_port = 4000
time_step  = 0.05    
spec_town2 = {'x':-7.39, 'y':312, 'z':10.2, 'pitch':-20, 'yaw':-45, 'roll':0}    


###########################################################################
# Load PIRL agent and carla environment     
def load_agent(env):
 
    actNum = env.action_num
    obsNum = len(env.reset())

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
        #OPTIMIZER  = Adam(learning_rate=1e-4),
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
    
    agent  = PIRLagent(model, actNum, agentOp, pinnOp)
    agent.load_weights('ITSC2024data/Town2/04291642', ckpt_idx=40_000) 
    
    return agent


def contour_plot(x, y, z, key = ["e", "psi"], filename=None):
    #(x, y) = (e, psi)/ (e, vx)
    x, y = np.meshgrid(x, y)
    x = x.transpose()
    y = y.transpose()
    
    # Creating the contour plot
    plt.figure(figsize=(6, 4))
    contour = plt.contourf(x, y, z, cmap='viridis')
    plt.colorbar(contour)  # Adding color bar to show z values
    
    # Adding labels and title
    if key[0] == "e" and key[1] == 'psi':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel(r'Heading error $\psi$ [rad]')
    if key[0] == "e" and key[1] == 'v':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel('Longitudinal velocity $v_x$ [m/s]')
    
    if filename:
       
       plt.savefig(filename, bbox_inches="tight")

    # Displaying the plot
    plt.show()

    
###############################################################################
if __name__ == '__main__':
    
    ###########################
    # Get nominal trajectory

    try:  

        # Get reference state
        def choose_spawn_point(carla_env):
            sp_list = carla_env.get_all_spawn_points()    
            spawn_point = sp_list[1]
            return spawn_point
        
        # carla_env
        rl_env = Env(port=carla_port, time_step=time_step, 
                        custom_map_path = None,
                        spawn_method    = None,# choose_spawn_point,
                        spectator_init  = spec_town2, 
                        spectator_reset = False, 
                        autopilot       = False)
        rl_env.reset()
        #rl_env.step()
        x_vehicle = rl_env.getVehicleState()
        x_road    = rl_env.getRoadState()

        horizon   = 5
        x_vehicle = np.array( rl_env.getVehicleState() )
        x_road    = np.array( rl_env.getRoadState() )
        s         = np.concatenate([ x_vehicle, x_road, np.array([horizon]) ])

        print(f'x_vehicle: {x_vehicle}')
        print(f'x_vehicle: {x_road}')

        ##############################
        agent = load_agent(rl_env)
           
        rslu = 50
        psi_scale = 0.4
        e_scale = 1.0
        e_list = np.linspace(-e_scale, e_scale, rslu)
        v_list = np.linspace(6, 20, rslu)
        psi_list = np.linspace(-psi_scale, psi_scale, rslu)
        
        interval = 0.5
        next_num = 0
        
        key =  ["e", "psi"] # v or psi 
        if key[1]=="psi":
            eps = 0.1 # initial distance
        else:
            eps = 0.5 # initial distance
        multi_test = False
        n = 10 #number of test points if multi_test == True
        
        if multi_test == True:
            refer_list = []
            vector_list = []
            waypoints_list = []
            for i in range(n):
                refer_list.append(rl_env.test_random_spawn_point(eps = eps + 0.01*i))
                vector, waypoints = rl_env.fetch_relative_states(wp_transform = refer_list[i], interval= interval, next_num= next_num)
                vector_list.append(vector)
                waypoints_list.append(waypoints)
        else:
            refer_point = rl_env.test_random_spawn_point(eps = eps)#eps = 0.1/0.5
            vector, waypoints = rl_env.fetch_relative_states(wp_transform = refer_point, interval= interval, next_num= next_num)        
        
        x_vehicle = np.array( [10, 0, 0] )
        x_road    = np.array( rl_env.getRoadState() )
        #print(vector_list)
        #print(type(vector_list))
        
        safe_p = np.zeros((len(e_list), len(psi_list)))
        if key[1] == "v":
            y_list = v_list
        else:
            y_list = psi_list
        
        x_road = [0, 0]
        for i in range(len(e_list)):
            x_road[0] = e_list[i]
            for j in range(len(y_list)):
                if key[1] == "v":
                    x_vehicle[0] = y_list[j]
                else:
                    x_road[1] = y_list[j]
                
                if multi_test == False:
                    new_x_road = x_road + vector
                    s         = np.concatenate([ x_vehicle, new_x_road, np.array([horizon]) ])       
                    safe_p[i][j]   = agent.get_qs(s).max()
                else:
                    for vector in vector_list:
                        new_x_road = x_road + vector
                        s         = np.concatenate([ x_vehicle, new_x_road, np.array([horizon]) ])       
                        safe_p[i][j]   += np.max(agent.get_qs(s))
                    safe_p[j][i] /= n
        
        np.savez(f"plot/Town2/safe_prob_{key[0]}_{key[1]}.npz", x=e_list, y=y_list, z=safe_p)
        
        contour_plot(x=e_list, y=y_list, z=safe_p, key=key,
                     filename=f'plot/Town2/safe_prob_{key[0]}_{key[1]}.png')

    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()
        
   
