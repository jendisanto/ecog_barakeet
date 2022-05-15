import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit

# paths
base_dir = '/Users/jenndisanto/Documents/2022/changlab-rotation/data/logfiles'
# '/Users/lauragwilliams/Documents/projects/barakeet/phoneme_seequence/sequence/data'

# get a list of the behavioural log files
files = glob.glob('%s/EC2*.csv' % (base_dir))

# loop through each file and put into one dataframe
dfs = list()
for f in files:
    df = pd.read_csv(f)
    df = df[np.array([str(s) != 'nan' for s in df['stim_number']])]
    if df['block_type'].unique() == 2:

        d = df['slider.response'].values
        d = np.abs(d - 11)
        df['slider.response'] = d

    dfs.append(df)
df = pd.concat(dfs)
df = df.reset_index()

# get mouse x and y
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for si, stim in enumerate(np.unique(df['stim_number'].values)):
    this_df = df.query("stim_number == @stim and block_type == 1.0" % (stim))
    
    # loop through time
    for tt in range(len(this_df)):

        # params
        c = plt.cm.RdBu_r(df['slider.response'].values[tt] / 10.)
        x = np.array(eval(np.array(this_df['mouse.x'])[tt]))
        y = np.array(eval(np.array(this_df['mouse.y'])[tt]))
    
        # plot this timepoint
        axs[si].plot(x, y, color=c, alpha=0.5, lw=0.5)
    axs[si].set_title(stim)
plt.show()
