import numpy as np
import pickle

import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import load_model

# Custom loss function
def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


def plot_result(file_path):
    try:
        
        new_model = load_model('model/my_model.keras', custom_objects={'my_loss_fn': my_loss_fn})

        # Show the model architecture
        new_model.summary()
        
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            
        print(len(data))
        print('averaged_rewards' in data)
        print('episodic_q0' in data)
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

if __name__ == '__main__':
    data_file_path = "data/data_20240219_082529.pickle"
    plot_result(data_file_path)

