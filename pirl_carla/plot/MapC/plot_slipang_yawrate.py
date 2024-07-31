import os
import numpy as np
import matplotlib.pyplot as plt

# data 
log_dir = 'data_trained/'
files   = os.listdir(log_dir)
files.sort()

# figure
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

for f in files:

    data = np.load(log_dir+f)    
    
    x    = data['state'][1,:] # * 3 - 25
    y    = data['state'][2,:] # + 1080
    time = np.linspace(0, 5, len(x))

    ax1.plot(time, x, lw=1, zorder=10)
    ax2.plot(time, y, lw=1, zorder=10)

# Axis 1
ax1.set_xlim([0, 5])
ax1.set_ylabel('Slip angle [deg]')

# Axis 2
ax2.set_xlabel('Time [s]')
ax2.set_xlim([0, 5])
ax2.set_ylabel('Yaw rate [deg/s]')

plt.savefig('time_series_mapC.png', bbox_inches="tight")