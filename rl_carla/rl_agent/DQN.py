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
        EPSILON_DECAY       = 0.95, 
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


# Deep Q-Network Agent class
class RLagent:
    def __init__(self, model, actNum, agentOp): 

        # Agent Options
        self.actNum = actNum
        self.agentOp = agentOp
        
        # Q-networks
        self.model = model
        self.model.compile(loss = "mean_squared_error", 
                           optimizer= agentOp['OPTIMIZER'])

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
        y = [] # label   set

        for index, (current_state, action, reward, new_state, is_terminal) in enumerate(minibatch):
            if not is_terminal:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.agentOp['DISCOUNT'] * max_future_q
            else:
                new_q = reward

            current_qs = np.array(current_qs_list[index])
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        #############################
        # Update of weights
        self.model.fit(np.array(X), np.array(y), 
                       batch_size=self.agentOp['MINIBATCH_SIZE'], 
                       verbose=0, shuffle=False)

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
            if episode % trainOp['SAVE_FREQ'] == 0:
                manager.save(checkpoint_number=episode) 

    return 