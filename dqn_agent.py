# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:37:14 2023
@author: hoshino
"""
# general packages
import time
import numpy as np
import random
from collections import deque # double-ended que
from tqdm import tqdm         # progress bar

# deep learning packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

# DQN hyper parameters
REPLAY_MEMORY_SIZE     = 5_000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
UPDATE_TARGET_EVERY = 5
DISCOUNT = 1 #0.99 # Should be 1 for risk quantification

MEMORY_FRACTION = 0.8 # GPU

# Logging
MODEL_NAME = "DQN_test"


# Modified Tensorboard class
# from https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        #self.writer = tf.summary.FileWriter(self.log_dir)        # for Tensorflow 1.x 
        self.writer = tf.summary.create_file_writer(self.log_dir) # for Tensorflow 2.x

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Deep Q-Network Agent class
class DQNAgent:
    def __init__(self, obs_dim, action_num): 
        
        # Q-networks
        self.model = self.create_model(obs_dim, action_num)
        self.target_model = self.create_model(obs_dim, action_num)
        self.target_model.set_weights(self.model.get_weights())

        # Replay Memory
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Data Log
        #self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        # Internal variables
        self.target_update_counter = 0
                
        #self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self, obs_dim, action_num):
        
        OBSERVATION_SPACE_DIM = [obs_dim,] # (X1, X2, T)
        ACTION_NUM = action_num          # u in {-1, 0, 1}
        
        model = Sequential([
                    Dense(32, input_shape= OBSERVATION_SPACE_DIM ),
                    Activation('tanh'), 
                    Dense(32),  
                    Activation('tanh'), 
                    Dense(ACTION_NUM),  
                ])
        
        model.compile(loss=self.my_loss_fn, optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"])
        return model  

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]


    def train(self, is_terminal_state):

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Samples from experience memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])        
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE, verbose=0)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE, verbose=0)
        
        X = [] # feature set
        y = [] # label   set

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        # Update of weights
        # self.model.fit(np.array(X), np.array(y), 
        #                batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, 
        #                callbacks=[self.tensorboard] if is_terminal_state else None)
        self.model.fit(np.array(X), np.array(y), 
                       batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        if is_terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    ###################################################################################
    # Custom loss function
    def my_loss_fn(self, y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


###################################################################################
# Environment (exmaple in ACC2024 paper)
class PlanerEnv:

    dt = 0.1;
    ACTIONS = [-1, 0, 1]
    ACTION_SPACE_SIZE = len(ACTIONS)    
    
    def reset(self):

        s  = np.sign( np.random.randn(2) )
        r  = np.random.rand(2)
        X1  = s[0]*r[0]*1.5
        X2  = s[1]*r[1]*1.5  #sign(r(2))*0.8 + sign(r(3))*0.35*rand;
        T   = np.random.randint(15)*self.dt
        self.state = np.array([X1, X2, T])
        
        return self.state

    def step(self, action):       
           
        # New state
        X1 = self.state[0]
        X2 = self.state[1]
        T  = self.state[2]
        U  = action
        new_state = np.array([
                          X1 + self.dt*( -X1**3- X2),
                          X2 + self.dt*( X1   + X2  +U),
                           T - self.dt 
                     ])
        
        # Check terminal conditios 
        isTimeOver = (T <= self.dt)
        isUnsafe   = abs( X2 ) > 1
        done       = isTimeOver or isUnsafe

        # Reward
        if done and (not isUnsafe):
            reward = 1
        else:
            reward = 0

        self.state = new_state
        
        return new_state, reward, done



##############################################################################
### Main algorithm ###########################################################
if __name__ == '__main__':

    #########################
    # Training option
    EPISODES = 5000    
    AGGREGATE_STATS_EVERY = 20  # episodes  

    #########################
    # Hyper parameters    
    EPSILON_DECAY = 0.95 ## 0.9975 99975
    MIN_EPSILON = 0.001


    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    #tf.set_random_seed(1) # for Tensorflow 1.x
    tf.random.set_seed(1) # for Tensorflow 2.x
    
    # Create agent and environment
    env   = PlanerEnv()
    agent = DQNAgent(obs_dim=3, action_num=3)

    ##############################
    # Iterate over episodes
    epsilon = 1
    ep_rewards = []
    average_rewards = []
    
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    #for episode in range(1, EPISODES + 1):
        
        # Update tensorboard step every episode
        #agent.tensorboard.step = episode

        # Restarting episode
        episode_reward = 0
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # get action
            if np.random.random() > epsilon:
                # Greedy action from Q network
                action_idx = np.argmax(agent.get_qs(current_state))
            else:
                # Random action
                action_idx = np.random.randint(0, env.ACTION_SPACE_SIZE)  
            
            # make a step
            action = env.ACTIONS[action_idx]
            new_state, reward, done = env.step(action)
            episode_reward += reward

            # update replay memory and train main network
            agent.update_replay_memory((current_state, action_idx, reward, new_state, done))
            agent.train( is_terminal_state=done )

            # update current state
            current_state = new_state
        
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        print(EPSILON_DECAY)

        # Log stats
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
            average_rewards.append(average_reward)

    state = np.array([[0,0,1], [1,2,3]])
    qs = agent.model.predict( state, verbose=0 )

    # Save model 
    #if min_reward >= MIN_REWARD:
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


                        