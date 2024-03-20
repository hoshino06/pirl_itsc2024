#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:58:23 2024
@author: hoshino
"""
# general packages
import numpy as np
import random
from collections import deque # double-ended que
from tqdm import tqdm  # progress bar
import datetime

# keras
import tensorflow as tf
from keras.models     import clone_model
from keras.optimizers import Adam

# Agent Options
def agentOptions(
        DISCOUNT   = 0.99, 
        OPTIMIZER  = Adam(learning_rate=0.01),
        REPLAY_MEMORY_SIZE = 5_000,
        REPLAY_MEMORY_MIN  = 100,
        MINIBATCH_SIZE     = 16, 
        UPDATE_TARGET_EVERY = 5, 
        EPSILON_INIT        = 1,
        EPSILON_DECAY       = 0.998, 
        EPSILON_MIN         = 0.01,
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
        'HESSIAN_CALC'    : HESSIAN_CALC,
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
        self.target_model = clone_model(self.model) #close with freshly initialized weights
        self.target_model.set_weights(self.model.get_weights())

        # Replay Memory
        self.replay_memory = deque(maxlen=agentOp['REPLAY_MEMORY_SIZE'])

        # Initialization of variables
        self.epsilon = agentOp['EPSILON_INIT']
        self.target_update_counter = 0
                
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False


    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]

    def get_epsilon_greedy_action(self, state):
        
        if np.random.random() > self.epsilon:
            # Greedy action from Q network
            action_idx = np.argmax(self.get_qs(state))
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

        print('--------------')
        start_time = datetime.datetime.now()        

        ########################
        # Sample minibatch from experience memory
        minibatch = random.sample(self.replay_memory, self.agentOp['MINIBATCH_SIZE'])

        #######################
        # Calculate traget y
        current_states = np.array([transition[0] for transition in minibatch])        
        current_qs_list = self.model(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model(new_current_states)
        
        X = [] # feature set
        y = [] # label   set (target y)

        for index, (current_state, action, reward, new_state, is_terminal) in enumerate(minibatch):
            if not is_terminal:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.agentOp['DISCOUNT'] * max_future_q
            else:
                new_q = reward

            current_qs = np.array(current_qs_list[index])  
            current_qs[action] = new_q           # update for target

            X.append(current_state)
            y.append(current_qs)

        X = np.array(X)
        y = np.array(y)

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print("sample_DQN:", elapsed_time)
        start_time = datetime.datetime.now()        

        ##########################
        # Samples for PDE
        X_PDE, X_BDini, X_BDsafe = self.pinnOp['SAMPLING_FUN']()
        X_PDE = tf.Variable(X_PDE)        

        # Convection and diffusion coefficient
        X_PDE = tf.Variable(X_PDE)
        Qsa = self.model(X_PDE)
        Uidx_PDE   = np.argmax(Qsa, axis=1).reshape(-1, 1)
        f          = np.apply_along_axis(self.pinnOp['CONVECTION_MODEL'], 1, 
                                         np.concatenate([X_PDE, Uidx_PDE], axis=1) )
        A =  np.apply_along_axis(self.pinnOp['DIFFUSION_MODEL'], 1, np.concatenate([X_PDE, Uidx_PDE], axis=1))


        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print("sample_PDE:", elapsed_time)
        start_time = datetime.datetime.now()        

        #########################
        # Calculate loss function
        with tf.GradientTape(watch_accessed_variables=False) as tape_for_loss:
  
            # Watch gradient wrt NN weights
            tape_for_loss.watch(self.model.trainable_variables)

            ####################
            # DQN Loss (lossD)
            ####################
            y_pred = self.model(X)         
            lossD = tf.metrics.mean_squared_error( y, y_pred )
        
            ####################
            # PDE loss (lossP)
            ####################
            start_time_hess = datetime.datetime.now()        

            if self.pinnOp['HESSIAN_CALC']: 
                with tf.GradientTape(watch_accessed_variables=False) as tape_dx2:
                    tape_dx2.watch( X_PDE )
                    with tf.GradientTape(watch_accessed_variables=False) as tape_dx:
                        tape_dx.watch( X_PDE )
                        Qsa    = self.model(X_PDE)
                        V      = tf.reduce_max(Qsa, axis=1)
                    dV_dx = tape_dx.gradient( V, X_PDE)
                    dV_dx = tf.cast(dV_dx, dtype=tf.float32)
                HessV = tape_dx2.batch_jacobian( dV_dx, X_PDE )
            else: 
                with tf.GradientTape(watch_accessed_variables=False) as tape_dx:
                    tape_dx.watch( X_PDE )
                    Qsa    = self.model(X_PDE)
                    V      = tf.reduce_max(Qsa, axis=1)
                dV_dx = tape_dx.gradient( V, X_PDE)
                dV_dx = tf.cast(dV_dx, dtype=tf.float32)

            end_time_hess = datetime.datetime.now()
            elapsed_time = end_time_hess - start_time_hess

            print("calc_Hess:", elapsed_time)

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
            conv_term =  tf.reduce_sum( dV_dx * f, axis=1 )

            if self.pinnOp['HESSIAN_CALC']:
                # Diffusion term            
                diff_term = (1/2) * tf.linalg.trace( tf.matmul(A, HessV) )
                diff_term = tf.cast(diff_term, dtype=tf.float32)
                              
                # lossP
                lossP = tf.metrics.mean_squared_error(conv_term + diff_term, 
                                                      np.zeros_like(conv_term) )
            else:
                # lossP
                lossP = tf.metrics.mean_squared_error(conv_term, 
                                                      np.zeros_like(conv_term) )             
            
            ########################
            # Boundary loss (lossB)
            ########################
            # termanal boundary (\tau = 0)
            y_bd_ini = tf.reduce_max(self.model(X_BDini), axis=1)
            lossBini = tf.metrics.mean_squared_error( y_bd_ini, np.ones_like(y_bd_ini) )
            
            # lateral boundary
            y_bd_safe = tf.reduce_max(self.model(X_BDsafe), axis=1)
            lossBsafe = tf.metrics.mean_squared_error( y_bd_safe, np.zeros_like(y_bd_safe) )
            
            lossB = lossBini + lossBsafe

            #####################
            # Total Loss function
            #####################
            Lambda = self.pinnOp['WEIGHT_PDE']
            Mu     = self.pinnOp['WEIGHT_BOUNDARY']
            loss = lossD + Lambda*lossP + Mu*lossB      

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print("loss:", elapsed_time)
        start_time = datetime.datetime.now()

        ############################
        # Update trainable variables
        ############################
        gradients = tape_for_loss.gradient(loss, self.model.trainable_variables)
        #print(gradients[0])

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print("grad:", elapsed_time)
        

        if is_episode_done:
            #############################
            # Update target Q-function and decay epsilon            
            self.target_update_counter += 1

            if self.target_update_counter > self.agentOp['UPDATE_TARGET_EVERY']:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

            ##############################
            # Decay epsilon
            if self.epsilon > self.agentOp['EPSILON_MIN']:
                self.epsilon *= self.agentOp['EPSILON_DECAY']
                self.epsilon = max( self.agentOp['EPSILON_MIN'], self.epsilon)

    def load_weights(self, ckpt_dir, ckpt_idx=None):

        checkpoint = tf.train.Checkpoint(model = self.model)
        manager    = tf.train.CheckpointManager(checkpoint, 
                                                directory=ckpt_dir, 
                                                max_to_keep=1000)
        if not ckpt_idx or ckpt_idx == 'latest': 
            ckpt_path = manager.latest_checkpoint
        else:
            ckpt_path = manager.checkpoints[ckpt_idx]
   
        checkpoint.restore(ckpt_path)
        
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
        ):
    
    trainOp = {
        'EPISODES' : EPISODES, 
        'LOG_DIR'  : LOG_DIR,
        'SHOW_PROGRESS': SHOW_PROGRESS,
        'SAVE_AGENTS'  : SAVE_AGENTS,
        'SAVE_FREQ'    : SAVE_FREQ,
        }
        
    return trainOp


def each_episode(agent, env, trainOp): 
    
    #############################
    # Reset esisodic reward and state
    episode_reward = 0
    current_state = env.reset()

    episode_q0 = np.max(agent.get_qs(current_state))

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
        summary_writer = tf.summary.create_file_writer(trainOp['LOG_DIR'])

        # Check point (for recording weights)
        ckpt    = tf.train.Checkpoint(model=agent.model)
        manager = tf.train.CheckpointManager(ckpt, trainOp['LOG_DIR'], trainOp['EPISODES'],
                                             checkpoint_name='weights')

    # Iterate episodes
    if trainOp['SHOW_PROGRESS']:     
        iterator = tqdm(range(1, trainOp['EPISODES'] + 1), ascii=True, unit='episodes')
    else:
        iterator = range(1, trainOp['EPISODES'] + 1)

    for episode in iterator:

        ep_reward, ep_q0 = each_episode(agent, env, trainOp)

        if trainOp['LOG_DIR']: 
            with summary_writer.as_default():
                tf.summary.scalar('Episode Reward', ep_reward, step=episode)                    
                tf.summary.scalar('Episode Q0',     ep_q0,     step=episode)                    
            if trainOp['SAVE_AGENTS'] and episode % trainOp['SAVE_FREQ'] == 0:
                manager.save(checkpoint_number=episode) 

    return 