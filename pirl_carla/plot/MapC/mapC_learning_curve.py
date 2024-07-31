import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.plugin_event_accumulator import EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE


def load_data(data_root_dir):

    #################
    # Data files
    #################
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

    return all_ep_q0, all_ep_rw

def averaging(data, wnum): 
    data_avr  = data.rolling(wnum, min_periods=100).mean() 
    data_mean = data_avr.mean(axis=1)
    data_std  = data_avr.std(axis=1)
    return data_mean, data_std

###############################################################################
# main
###############################################################################
if __name__ == '__main__':

    #############################
    # Load and average data
    #############################
    
    data_root_dir   = '../../logs/MapC/2e-4'
    _, ep_rw_2e_m4  = load_data(data_root_dir)
    
    data_root_dir   = '../../logs/MapC/1e-4'
    _, ep_rw_1e_m4  = load_data(data_root_dir)
    
    data_root_dir   = '../../logs/MapC/5e-5'
    _, ep_rw_5e_m5  = load_data(data_root_dir)
    
    data_root_dir   = '../../logs/MapC/2e-5'
    _, ep_rw_2e_m5  = load_data(data_root_dir)
    
    wnum = 5000
    mean_2e_m4, std_2e_m4 = averaging(ep_rw_2e_m4, wnum)
    mean_1e_m4, std_1e_m4 = averaging(ep_rw_1e_m4, wnum)
    mean_5e_m5, std_5e_m5 = averaging(ep_rw_5e_m5, wnum)
    mean_2e_m5, std_2e_m5 = averaging(ep_rw_2e_m5, wnum)

    #############################
    # Plot data
    #############################
    plt.plot(mean_2e_m4, lw=1, label='2e-4')
    plt.plot(mean_1e_m4, lw=1, label='1e-4')
    plt.plot(mean_5e_m5, lw=1, label='5e-5')
    plt.plot(mean_2e_m5, lw=1, label='2e-5')

    plt.fill_between(range(len(mean_2e_m4)), 
                     mean_2e_m4 - std_2e_m4, mean_2e_m4 + std_2e_m4, 
                     alpha=0.2, label='')
    plt.fill_between(range(len(mean_1e_m4)), 
                     mean_1e_m4 - std_1e_m4, mean_1e_m4 + std_1e_m4, 
                     alpha=0.2, label='')
    plt.fill_between(range(len(mean_5e_m5)), 
                     mean_5e_m5 - std_5e_m5, mean_5e_m5 + std_5e_m5, 
                     alpha=0.15, label='')
    plt.fill_between(range(len(mean_2e_m5)), 
                     mean_2e_m5 - std_2e_m5, mean_2e_m5 + std_2e_m5, 
                     alpha=0.15, label='')

    plt.xlim([0, 100_000])
    plt.ylim([0,1])
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    
    plt.savefig('mapC_learning_curve.png', bbox_inches="tight")
