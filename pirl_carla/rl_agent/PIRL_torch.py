""" DQN based PIRL implementation with pytorch
        1. agentOptions
        2. pinnOptions
        3. PIRLagent
        4. trainOptions
        5. train
"""

__author__ = 'Hikaru Hoshino'
__email__ = 'hoshino@eng.u-hyogo.ac.jp'

import os
import numpy as np
import random
import copy
from collections import deque # double-ended que
from tqdm import tqdm  # progress bar
import torch
from   torch.utils.tensorboard import SummaryWriter

# Agent Options
def agentOptions(
        DISCOUNT            = 0.99, 
        OPTIMIZER           = None,
        REPLAY_MEMORY_SIZE  = 5_000,
        REPLAY_MEMORY_MIN   = 100,
        MINIBATCH_SIZE      = 16, 
        UPDATE_TARGET_EVERY = 5, 
        EPSILON_INIT        = 1,
        EPSILON_DECAY       = 0.998, 
        EPSILON_MIN         = 0.01,
        RESTART_EP          = None, 
        ):
    
    agentOp = {
        'DISCOUNT'          : DISCOUNT,
        'OPTIMIZER'         : OPTIMIZER,
        'REPLAY_MEMORY_SIZE': REPLAY_MEMORY_SIZE,
        'REPLAY_MEMORY_MIN' : REPLAY_MEMORY_MIN,
        'MINIBATCH_SIZE'    : MINIBATCH_SIZE, 
        'UPDATE_TARGET_EVERY':UPDATE_TARGET_EVERY, 
        'EPSILON_INIT'      : EPSILON_INIT,
        'EPSILON_DECAY'     : EPSILON_DECAY, 
        'EPSILON_MIN'       : EPSILON_MIN,
        'RESTART_EP'        : RESTART_EP
        }
    
    return agentOp

# PINN Options
def pinnOptions(
        CONVECTION_MODEL,
        DIFFUSION_MODEL,
        SAMPLING_FUN, 
        WEIGHT_PDE      = 1e-3, 
        WEIGHT_BOUNDARY = 1, 
        HESSIAN_CALC    = True,
        ):

    pinnOp = {
        'CONVECTION_MODEL': CONVECTION_MODEL,
        'DIFFUSION_MODEL' : DIFFUSION_MODEL, 
        'SAMPLING_FUN'    : SAMPLING_FUN,
        'WEIGHT_PDE'      : WEIGHT_PDE,
        'WEIGHT_BOUNDARY' : WEIGHT_BOUNDARY,
        'HESSIAN_CALC'      : HESSIAN_CALC,
        }

    return pinnOp

# Deep Q-Network Agent class
class PIRLagent:
    def __init__(self, model, actNum, agentOp, pinnOp): 

        # Agent Options
        self.actNum  = actNum
        self.agentOp = agentOp
        self.pinnOp  = pinnOp
        
        # Q-networks
        self.model     = model
        self.optimizer = agentOp['OPTIMIZER']

        # Target Q-network 
        self.target_model = copy.deepcopy(self.model)

        # Replay Memory
        self.replay_memory = deque(maxlen=agentOp['REPLAY_MEMORY_SIZE'])

        # Initialization of variables
        self.epsilon = agentOp['EPSILON_INIT'] if agentOp['RESTART_EP'] == None else max( self.agentOp['EPSILON_MIN'], agentOp['EPSILON_INIT']*np.power(agentOp['EPSILON_DECAY'], agentOp['RESTART_EP']))
        self.target_update_counter = 0
                
        self.terminate            = False
        self.last_logged_episode  = 0
        self.training_initialized = False

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def get_qs(self, state):
        state = torch.tensor(state, dtype=torch.float32)        
        return self.model(state)
    
    def get_epsilon_greedy_action(self, state):
        
        if np.random.random() > self.epsilon:
            # Greedy action from Q network
            action_idx = int( torch.argmax(self.get_qs(state)) )
        else:
            # Random action
            action_idx = np.random.randint(0, self.actNum)  
        return action_idx

    def train_step(self, experience, is_episode_done):

        ########################
        # Update replay memory
        self.update_replay_memory(experience)

        if len(self.replay_memory) < self.agentOp['REPLAY_MEMORY_MIN']:
            return

        #print('--------------')
        #start_time = datetime.datetime.now()        

        ########################
        # Sample minibatch from experience memory
        minibatch = random.sample(self.replay_memory, self.agentOp['MINIBATCH_SIZE'])

        #######################
        # Calculate traget y
        current_states = np.array([transition[0] for transition in minibatch], dtype=np.float32)        
        current_qs_list = self.model(torch.from_numpy(current_states)).detach().numpy()

        new_current_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        future_qs_list = self.target_model(torch.from_numpy(new_current_states)).detach().numpy()
        
        X = [] # feature set
        y = [] # label   set (target y)

        for index, (current_state, action, reward, new_state, is_terminal) in enumerate(minibatch):
            if not is_terminal:
                max_future_q = future_qs_list[index].max()
                new_q = reward + self.agentOp['DISCOUNT'] * max_future_q
            else:
                new_q = reward

            current_qs = np.array(current_qs_list[index]) 
            current_qs[action] = new_q           # update for target

            X.append(current_state)
            y.append(current_qs)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        #end_time = datetime.datetime.now()
        #elapsed_time = end_time - start_time
        #print("sample_DQN:", elapsed_time)
        #start_time = datetime.datetime.now()        

        ##########################
        # Samples for PDE
        X_PDE, X_BDini, X_BDlat = self.pinnOp['SAMPLING_FUN'](self.replay_memory)
        #X_PDE = tf.Variable(X_PDE)        

        # Convection and diffusion coefficient
        with torch.no_grad():
            Qsa = self.model(torch.tensor(X_PDE, dtype=torch.float))
            Uidx_PDE   = Qsa.argmax(1).numpy().reshape(-1, 1)               
        f =  np.apply_along_axis(self.pinnOp['CONVECTION_MODEL'], 1, 
                                 np.concatenate([X_PDE, Uidx_PDE], axis=1) )
        A =  np.apply_along_axis(self.pinnOp['DIFFUSION_MODEL'], 1, 
                                 np.concatenate([X_PDE, Uidx_PDE], axis=1))

        #end_time = datetime.datetime.now()
        #elapsed_time = end_time - start_time
        #print("sample_PDE:", elapsed_time)
        #start_time = datetime.datetime.now()        

        ####################
        # DQN Loss (lossD)
        ####################
        y_pred = self.model(torch.from_numpy(X))
        y_trgt = torch.from_numpy(y)
        lossD  = torch.nn.functional.mse_loss( y_pred, y_trgt )
    
        ####################
        # PDE loss (lossP)
        ####################
        #start_time_hess = datetime.datetime.now()        

        if self.pinnOp['HESSIAN_CALC']: 

            #from functorch import hessian
            
            X_PDE = torch.tensor(X_PDE, dtype=torch.float, requires_grad=True)
            Qsa   = self.model(X_PDE)
            V     = Qsa.max(1) 
            dV_dx = torch.autograd.grad(V.values.sum(), X_PDE, create_graph=True)[0]
            
            # jacobian_rows = [torch.autograd.grad(dV_dx, X_PDE, vec)[0]
            #          for vec in torch.eye(len(X_PDE[0]))]
            # return torch.stack(jacobian_rows)
                
        else: 
            
            X_PDE = torch.tensor(X_PDE, dtype=torch.float, requires_grad=True)
            Qsa   = self.model(X_PDE)
            V     = Qsa.max(1) 
            dV_dx = torch.autograd.grad(V.values.sum(), X_PDE, create_graph=True)[0]
                        
        #end_time_hess = datetime.datetime.now()
        #elapsed_time = end_time_hess - start_time_hess

        #print("calc_Hess:", elapsed_time)

        '''
        # check gradient implementation (for debug)
        print('\n V=', V)
        ##
        V_dx = tf.reduce_max( self.model( X_PDE + [0.01, 0, 0]), axis=1)
        dV_dx_man = ( V_dx - V ) / 0.01
        print('dV_dx[:,0]=',  dV_dx[:,0])
        print('dV_dx[:,0] ~ ', dV_dx_man)
        ##
        V_dx2 = tf.reduce_max( self.model( X_PDE - [0.01, 0, 0]), axis=1)
        HessV0_man = ( V_dx - 2.0*V + V_dx2 ) / ( (0.01)**2 )
        print(Hess[:,0])
        print(HessV0_man)
        '''                  

        ## Convection term
        conv_term = ( dV_dx * torch.tensor(f, dtype=torch.float32) ).sum(1)

        if self.pinnOp['HESSIAN_CALC']:
            # Diffusion term            
            #diff_term = (1/2) * tf.linalg.trace( tf.matmul(A, HessV) )
            #diff_term = tf.cast(diff_term, dtype=tf.float32)
                          
            # lossP
            # lossP = tf.metrics.mean_squared_error(conv_term + diff_term, 
            #                                       np.zeros_like(conv_term) )
            lossP = torch.nn.functional.mse_loss( conv_term, 
                                                  torch.zeros_like(conv_term) )             

        else:
            # lossP
            lossP = torch.nn.functional.mse_loss( conv_term, 
                                                  torch.zeros_like(conv_term) )             
        
        ########################
        # Boundary loss (lossB)
        ########################
        # termanal boundary (\tau = 0)
        y_bd_ini = self.model(torch.tensor(X_BDini, dtype=torch.float32)).max(1).values
        lossBini = torch.nn.functional.mse_loss( y_bd_ini, torch.ones_like(y_bd_ini) )
        
        # lateral boundary
        y_bd_lat = self.model(torch.tensor(X_BDlat, dtype=torch.float32)).max(1).values
        lossBlat = torch.nn.functional.mse_loss( y_bd_lat, torch.zeros_like(y_bd_lat) )
        
        lossB = lossBini + lossBlat

        #####################
        # Total Loss function
        #####################
        Lambda = self.pinnOp['WEIGHT_PDE']
        Mu     = self.pinnOp['WEIGHT_BOUNDARY']
        loss = lossD + Lambda*lossP + Mu*lossB      

        #end_time = datetime.datetime.now()
        #elapsed_time = end_time - start_time
        #print("loss:", elapsed_time)
        #start_time = datetime.datetime.now()

        ############################
        # Update trainable variables
        ############################
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        #end_time = datetime.datetime.now()
        #elapsed_time = end_time - start_time
        #print("grad:", elapsed_time)

        if is_episode_done:
            #############################
            # Update target Q-function and decay epsilon            
            self.target_update_counter += 1

            if self.target_update_counter > self.agentOp['UPDATE_TARGET_EVERY']:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_update_counter = 0

            ##############################
            # Decay epsilon
            if self.epsilon > self.agentOp['EPSILON_MIN']:
                self.epsilon *= self.agentOp['EPSILON_DECAY']
                self.epsilon = max( self.agentOp['EPSILON_MIN'], self.epsilon)

    def load_weights(self, ckpt_dir, ckpt_idx=None):

        if not os.path.isdir(ckpt_dir):         
            raise FileNotFoundError("Directory '{}' does not exist.".format(ckpt_dir))

        if not ckpt_idx or ckpt_idx == 'latest': 
            check_points = [item for item in os.listdir(ckpt_dir) if 'agent' in item]
            check_nums   = np.array([int(file_name.split('-')[1]) for file_name in check_points])
            latest_ckpt  = f'/agent-{check_nums.max()}'  
            ckpt_path    = ckpt_dir + latest_ckpt
        else:
            ckpt_path = ckpt_dir + f'/agent-{ckpt_idx}'
            if not os.path.isfile(ckpt_path):   
                raise FileNotFoundError("Check point 'agent-{}' does not exist.".format(ckpt_idx))

        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['weights'])
        self.target_model.load_state_dict(checkpoint['target-weights'])        
        self.replay_memory = checkpoint['replay_memory']
        
        print(f'Agent loaded weights stored in {ckpt_path}')
        
        return ckpt_path    


###################################################################################
# Learning Algorithm

def trainOptions(
        EPISODES      = 50, 
        LOG_DIR       = None,
        SHOW_PROGRESS = True,
        SAVE_AGENTS   = True,
        SAVE_FREQ     = 1,
        RESTART_EP    = None
        ):
    
    trainOp = {
        'EPISODES' : EPISODES, 
        'LOG_DIR'  : LOG_DIR,
        'SHOW_PROGRESS': SHOW_PROGRESS,
        'SAVE_AGENTS'  : SAVE_AGENTS,
        'SAVE_FREQ'    : SAVE_FREQ,
        'RESTART_EP'   : RESTART_EP
        }
        
    return trainOp


def each_episode(agent, env, trainOp): 
    
    #############################
    # Reset esisodic reward and state
    episode_reward = 0
    current_state = env.reset()

    episode_q0 = agent.get_qs(current_state).max()

    ###############################
    # Iterate until episode ends
    is_done = False
    while not is_done:

        # get action
        action_idx = agent.get_epsilon_greedy_action(current_state)
        
        # make a step
        new_state, reward, is_done = env.step(action_idx)
        episode_reward += reward

        # train Q network
        experience = (current_state, action_idx, reward, new_state, is_done)
        agent.train_step( experience, is_done )

        # update current state
        current_state = new_state

    return episode_reward, episode_q0
    

def train(agent, env, trainOp):
    
    # Log file
    if trainOp['LOG_DIR']: 
        
        # For training stats
        summary_writer = SummaryWriter(log_dir=trainOp['LOG_DIR'])        

    start = 1 if trainOp['RESTART_EP'] == None else trainOp['RESTART_EP']
    # Iterate episodes
    if trainOp['SHOW_PROGRESS']:     
        iterator = tqdm(range(start+1, trainOp['EPISODES'] + 1), ascii=True, unit='episodes')
    else:
        iterator = range(start+1, trainOp['EPISODES'] + 1)

    for episode in iterator:

        ep_reward, ep_q0 = each_episode(agent, env, trainOp)

        if trainOp['LOG_DIR']: 
            summary_writer.add_scalar("Episode Reward", ep_reward, episode)
            summary_writer.add_scalar("Episode Q0",     ep_q0,     episode)
            summary_writer.flush()

            if trainOp['SAVE_AGENTS'] and episode % trainOp['SAVE_FREQ'] == 0:
                
                ckpt_path = trainOp['LOG_DIR'] + f'/agent-{episode}'
                torch.save({'weights':        agent.model.state_dict(),
                            'target-weights': agent.target_model.state_dict(),
                            'replay_memory':  agent.replay_memory}, 
                           ckpt_path)
                
    return 