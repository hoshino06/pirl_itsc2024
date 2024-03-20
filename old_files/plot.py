import random
from tqdm import tqdm # for showing progress bar
from datetime import datetime

import numpy as np
import pickle

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf
from keras.models import load_model

# Custom loss function
def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


def plot_result(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            
        new_model = load_model('model/my_model.keras', custom_objects={'my_loss_fn': my_loss_fn})

        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            
        fig, axs = plt.subplots(1, 2)
        
        if 'averaged_rewards' in data and len(data['averaged_rewards']) > 0:
            X = np.linspace(start=0, stop=50_000, num=len(data['averaged_rewards']))
            axs[0].set_xlabel("episode")
            axs[0].set_ylabel("avg_reward")
        
            #plt.subplot(1, 3, 2)
            axs[0].plot(X, data['averaged_rewards'])


        if 'episodic_q0' in data and  len(data['episodic_q0']) > 0:
            X2 = np.linspace(start=0, stop=50_000, num=len(data['episodic_q0']))
            axs[1].set_xlabel("episode")
            axs[1].set_ylabel("q_value")
        
            #plt.subplot(1, 3, 3)
            axs[1].plot(X2, data['episodic_q0'])

            plt.show()

    except FileNotFoundError:
        print("No such file")
        
    return

def apply_exponential_smoothing(data_frame, column_name, decay_factor):
    # Apply exponential smoothing on the specified column using the provided decay factor (alpha)
    return data_frame[column_name].ewm(alpha=decay_factor, adjust=False).mean()

def plot_25k_dqn_results(ep_reward_filepaths, q0_value_filepaths):
    dfs = []
    for fp in ep_reward_filepaths:
        df = pd.read_csv(fp)
        # Adjust step numbers for the second file
        if len(dfs) > 0:
            df['Step'] += dfs[-1]['Step'].iloc[-1]
            df['Value'] += dfs[-1]['Value'].iloc[-1]
        dfs.append(df)    
    ep_reward_data = pd.concat(dfs, ignore_index=True)
        
    dfs = []
    for fp in q0_value_filepaths:
        df = pd.read_csv(fp)
        # Adjust step numbers for the second file
        if len(dfs) > 0:
            df['Step'] += dfs[-1]['Step'].iloc[-1]
            #df['Value'] += dfs[-1]['Value'].iloc[-1]
        dfs.append(df)
    q0_value_data = pd.concat(dfs, ignore_index=True)
    
    
    ep_reward_data['Smoothed_Value'] = apply_exponential_smoothing(ep_reward_data, 'Value', 0.01)
    q0_value_data['Smoothed_Value'] = apply_exponential_smoothing(q0_value_data, 'Value', 0.01)
    
    # Plot ep_reward data
    plt.figure(figsize=(10, 5))
    plt.plot(ep_reward_data['Step'], ep_reward_data['Smoothed_Value'], label='ep_reward')
    plt.xlabel('Step')
    plt.ylabel('Episode Reward')
    plt.title('Smoothed Episode Reward over Steps')
    plt.grid(True)
    plt.show()
    
    # Plot q0 data
    plt.figure(figsize=(10, 5))
    plt.plot(q0_value_data['Step'], q0_value_data['Smoothed_Value'], label='q0')
    plt.xlabel('Step')
    plt.ylabel('Q0')
    plt.title('Smoothed Q0 over Steps')
    plt.grid(True)
    plt.show()

    

if __name__ == '__main__':
    #data_file_path = "data/data_20240219_082529.pickle"
    #plot_result(data_file_path)
    
    ep_reward_files = ['/home/arnav/Downloads/19k_ep_reward.csv', '/home/arnav/Downloads/6k_ep_reward.csv']
    q0_files = ['/home/arnav/Downloads/19k_q0.csv', '/home/arnav/Downloads/6k_q0.csv']
    
    plot_25k_dqn_results(ep_reward_files, q0_files)
    
    

