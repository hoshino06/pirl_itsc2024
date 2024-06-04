import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorboard as tb

# figure
#data_ep_rw  = pd.read_csv('MapC_04071140-19k_ep_reward.csv')
data_ep_rw  = pd.read_csv('Town2_04291642_ep_rewards.csv')
data_avr_rw = data_ep_rw.rolling(400).mean() 

#data_ep_q0  = pd.read_csv('MapC_04071140-19k_ep_q0.csv')
data_ep_q0  = pd.read_csv('Town2_04291642_ep_q0.csv')
data_avr_q0 = data_ep_q0.rolling(400).mean() 

plt.plot(data_avr_q0['Step'], data_avr_q0['Value'], lw=1, label='Average Q value')
plt.plot(data_avr_rw['Step'], data_avr_rw['Value'], lw=1, label='Averge Episode Reward')

plt.xlim([0, 30_000])
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.legend()

plt.savefig('Town2_learning_curve.png', bbox_inches="tight")

