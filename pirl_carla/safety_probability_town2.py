"""
Plot safe probability vs map
"""
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

carla_port = 3000
time_step  = 0.05    
spec_town2 = {'x':-7.39, 'y':312, 'z':10.2, 'pitch':-20, 'yaw':-45, 'roll':0}    
key        =  ["e", "psi"] # v or psi 

#log_dir = 'ITSC2024data/Town2/04291642'
log_dir     = 'logs/Town2/0725/1640'
check_point = 'latest' 

###########################################################################
# Load PIRL agent and carla environment     
def load_agent(env, log_dir):
 
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
    agent  = PIRLagent(model, actNum, agentOp, pinnOp)
    
    #agent.load_weights('ITSC2024data/Town2/04291642', ckpt_idx=40_000) 
    agent.load_weights(log_dir, ckpt_idx=check_point) 
    
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

    try:  
        ###################################
        # Load env and agent
        ##################################
        rl_env = Env(port=carla_port, time_step=time_step, 
                        custom_map_path = None,
                        spawn_method    = None, 
                        spectator_init  = spec_town2, 
                        spectator_reset = False, 
                        autopilot       = False)
        agent = load_agent(rl_env, log_dir)

        ##############################
           
        rslu = 50
        psi_scale = 0.4
        e_scale = 1.0
        e_list = np.linspace(-e_scale, e_scale, rslu)
        v_list = np.linspace(5, 25, rslu)
        psi_list = np.linspace(-psi_scale, psi_scale, rslu)
        
        #############################
        # get waypoints
        #############################
        interval = 0.5
        next_num = 0
        
        if key[1]=="psi":
            eps = 0.1 # initial distance
        else:
            eps = 0.5 # initial distance
        
        refer_point = rl_env.test_random_spawn_point(eps = eps) #eps = 0.1/0.5
        vector, waypoints = rl_env.fetch_relative_states(wp_transform = refer_point, interval= interval, next_num= next_num)
        print(vector)
        # plt.scatter(vector[0:5], vector[5:10])
        # plt.xlim([-2,2])
        # plt.ylim([0,3])        
        
        ##############################
        # safety probability
        #############################
        x_vehicle = np.array( [10, 0, 0] )
        horizon   = 5

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
                
                new_x_road = x_road + vector
                s          = np.concatenate([ x_vehicle, new_x_road, np.array([horizon]) ])       
                safe_p[i][j] = agent.get_qs(s).max()
        
        #################################       
        # Save and plot results
        ##################################
        #np.savez(f"plot/Town2/safe_prob_{key[0]}_{key[1]}.npz", x=e_list, y=y_list, z=safe_p)
        contour_plot(x=e_list, y=y_list, z=safe_p, key=key,
                     filename=f'plot/Town2/safe_prob_{key[0]}_{key[1]}.png')

    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()
        
   