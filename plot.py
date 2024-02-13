import random
import numpy as np
from tqdm import tqdm # for showing progress bar
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from carla_env import CarEnv
from dqn_agent import DQNAgent







def plot_result(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            
        print(data)
        print("agent" in data)
        print('averaged_rewards' in data)
        print('episodic_q0' in data)
        fig, axs = plt.subplots(1, 2)
        
        if 'averaged_rewards' in data and len(data['averaged_rewards']) > 0:
            X = np.linspace(start=0, stop=data['max_episodes'], num=len(data['averaged_rewards']))
            axs[0].set_xlabel("episode")
            axs[0].set_ylabel("avg_reward")
        
            #plt.subplot(1, 3, 2)
            axs[0].plot(X, data['averaged_rewards'])


        if 'episodic_q0' in data and  len(data['episodic_q0']) > 0:
            X2 = np.linspace(start=0, stop=data['max_episodes'], num=len(data['episodic_q0']))
            axs[1].set_xlabel("episode")
            axs[1].set_ylabel("q_value")
        
            #plt.subplot(1, 3, 3)
            axs[1].plot(X2, data['episodic_q0'])

            plt.show()

    except FileNotFoundError:
        print("No such file")
        
    return

if __name__ == '__main__':
    file_path = "data/data_20240213_162049.pickle"
    plot_result(file_path)
