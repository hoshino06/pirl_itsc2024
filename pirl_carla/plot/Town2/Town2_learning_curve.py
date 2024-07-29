import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.plugin_event_accumulator import EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE

###############################################################################
# main
###############################################################################
if __name__ == '__main__':

    #################
    # Data files
    #################
    data_root_dir   = '../../logs/Town2/0725'
    data_dirs = [os.path.join(data_root_dir, file) for file in os.listdir(data_root_dir)]
    event_files = [ glob.glob(os.path.join(data_dir, "events*"))[0] for data_dir in data_dirs ]     

    #########################
    # Extract learning data
    #########################
    all_ep_q0 = pd.DataFrame()
    all_ep_rw = pd.DataFrame()
    
    for i, event_file in enumerate(event_files):        

        print(f'loading {event_files[i]}')    
        acc = EventAccumulator(path=event_files[i], size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
        acc.Reload()

        ep_q0 = [t.tensor_proto.float_val[0] for t in acc.Tensors('Episode Q0')]
        ep_rw = [t.tensor_proto.float_val[0] for t in acc.Tensors('Episode Reward')]

        all_ep_q0[i] = pd.Series(ep_q0)
        all_ep_rw[i] = pd.Series(ep_rw)

    ############################
    # Averaging
    ############################
    data_avr_q0 = all_ep_q0.rolling(100).mean() 
    data_avr_rw = all_ep_rw.rolling(100).mean() 
    mean_q0     = data_avr_q0.mean(axis=1)
    std_q0      = data_avr_q0.std(axis=1)
    mean_rw     = data_avr_rw.mean(axis=1)
    std_rw      = data_avr_rw.std(axis=1)

    #############################
    # Plot data
    #############################
    plt.plot(mean_q0, lw=1, label='Average Q value')
    plt.plot(mean_rw, lw=1, label='Averge Episode Reward')

    plt.fill_between(range(len(mean_q0)), mean_q0 - std_q0, mean_q0 + std_q0, alpha=0.3, label='')
    plt.fill_between(range(len(mean_rw)), mean_rw - std_rw, mean_rw + std_rw, alpha=0.3, label='')

    plt.xlim([0, 50_000])
    plt.ylim([0,1])
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend(loc='lower right')
    
    plt.savefig('Town2_learning_curve.png', bbox_inches="tight")
