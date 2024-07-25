import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.plugin_event_accumulator import EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE


###############################################################################
# main
###############################################################################
if __name__ == '__main__':

    #################
    # Data
    #################
    data_dir   = '../../ITSC2024data/Town2/04291642'
    #data_dir   = '../../logs/Town2/07251044'

    # Read event file
    event_file = glob.glob(os.path.join(data_dir, "events*"))[0]
    print(f'loading {event_file}')
    acc = EventAccumulator(path=event_file, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    acc.Reload()

    # Extract learning data
    ep_q0 = [t.tensor_proto.float_val[0] for t in acc.Tensors('Episode Q0')]
    ep_rw = [t.tensor_proto.float_val[0] for t in acc.Tensors('Episode Reward')]

    ep_q0 = pd.Series(ep_q0)
    ep_rw = pd.Series(ep_rw)


    # Averaging
    data_avr_q0 = ep_q0.rolling(400).mean() 
    data_avr_rw = ep_rw.rolling(400).mean() 

    # Plot data
    plt.plot(data_avr_q0, lw=1, label='Average Q value')
    plt.plot(data_avr_rw, lw=1, label='Averge Episode Reward')

    plt.xlim([0, 30_000])
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    
    #plt.savefig('Town2_learning_curve.png', bbox_inches="tight")
