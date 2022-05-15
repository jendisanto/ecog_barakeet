# modules
import pandas as pd
import sys
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import ndx_ecog  # Even though we won't use this directly, we need to import it when reading our NWB files since we use the ECoG extension
from pynwb import NWBHDF5IO
from changlab_to_nwb.TDT_to_NWB import TDTtoNWB
from mne.io import RawArray
from mne import create_info, find_events, Epochs, concatenate_epochs




def rolling_window(array, window_size, freq):
    shape = (array.shape[0] - window_size + 1, window_size)
#     print(shape)
    strides = (array.strides[0],) + array.strides
#     print(strides)
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], freq)]



subject = 'EC243'
block = '48'
'''
subject:    str, e.g. "EC228"
blocks:     list, e.g. ['35', '36']
data_type:  str, options: "LFP", "high_gamma"
time_lock:  str, options: "word_onset", "POD", "word_offset"
tmin:       float, seconds
tmax:       float, seconds
'''

# paths
root_dir = '/Users/jenndisanto/Documents/2022/changlab-rotation'
data_dir = '%s/data-ecog' % (root_dir)
logfile_dir = '%s/data/logfiles' % (root_dir)

# loop through blocks to get epochs
epoch_list = list()

print('Analysing data for %s B%s..' % (subject, block))

# load log file
log = pd.read_csv('%s/%s_sequence_B%s.csv' % (logfile_dir, subject, block))

# clean log file to just get auditory trials
log = log.query("stim_number > 0")

# read nwb file
nwb_path = '%s/%s_B%s.nwb' % (data_dir, subject, block)
nwb_file_io = NWBHDF5IO(nwb_path, 'r')
nwb_file = nwb_file_io.read()

speaker_data = nwb_file.stimulus['speaker1'].data[:] #[0:int(13745118/10)]

#params
speaker_rate = nwb_file.stimulus['speaker1'].rate
silence_thresh = 0.08
silence_len = int(0.5 * speaker_rate)

#find silent windows
data_bool = speaker_data > silence_thresh
windows = rolling_window(data_bool, silence_len, 1)
silent_windows = ~np.array(list(map(np.any, windows)))

#find silent->sound indices
switches = silent_windows[:-1] > silent_windows[1:]
idx = np.where(switches == True)
timestamps = idx / speaker_rate

#save timestamps
np.save('%s/timestamps_idx' % (data_dir), idx)
np.save('%s/timestamps_sec' % (data_dir), timestamps)
print(len(idx))