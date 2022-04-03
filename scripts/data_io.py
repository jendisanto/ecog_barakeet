
# modules
import pandas as pd
import sys
import numpy as np
from scipy.stats import zscore

import ndx_ecog  # Even though we won't use this directly, we need to import it when reading our NWB files since we use the ECoG extension
from pynwb import NWBHDF5IO
from changlab_to_nwb.TDT_to_NWB import TDTtoNWB
from mne.io import RawArray
from mne import create_info, find_events, Epochs, concatenate_epochs

# paths
root_dir = '/Users/lauragwilliams/Dropbox (UCSF Department of Neurological Surgery)/lg/barakeet'
plot_dir = '%s/vis' % (root_dir)
textgrid_dir = '%s/external/TextGrids' % (root_dir)

# helper script to get timing from the textgrids
def event_timing(word, amb_level, tg_dir, time_lock):
    from glob import glob

    # build fname
    fname = glob('%s/*_%s_*%s*.TextGrid' % (textgrid_dir, word, int(amb_level)))[0]

    # load file
    tg = pd.read_csv(fname)

    # lock to moment
    if time_lock == 'POD':
        for ii in range(len(tg)):
            if 'POD' in str(tg.values[ii]):
                pod_timing = float(str(tg.values[ii+5]).split('= ')[1].split(' ')[0])

    elif time_lock == 'word_offset':
        for ii in range(len(tg)):
            if 'word' in str(tg.values[ii]):
                pod_timing = float(str(tg.values[ii+2]).split('= ')[1].split(' ')[0])

    return pod_timing



def load_epochs(subject, blocks, data_type, time_lock, tmin, tmax):
    '''
    subject:    str, e.g. "EC228"
    blocks:     list, e.g. ['35', '36']
    data_type:  str, options: "LFP", "high_gamma"
    time_lock:  str, options: "word_onset", "POD", "word_offset"
    tmin:       float, seconds
    tmax:       float, seconds
    '''

    # paths
    data_dir = '%s/data' % (root_dir)
    logfile_dir = '%s/logfiles' % (data_dir)

    # loop through blocks to get epochs
    epoch_list = list()
    for block in blocks:

        print('Analysing data for %s B%s..' % (subject, block))

        # load log file
        log = pd.read_csv('%s/%s_sequence_B%s.csv' % (logfile_dir, subject, block))

        # clean log file to just get auditory trials
        log = log.query("stim_number > 0")

        # read nwb file
        nwb_path = '%s/ecog/%s_B%s/%s_B%s.nwb' % (data_dir, subject, block, subject, block)
        nwb_file_io = NWBHDF5IO(nwb_path, 'r')
        nwb_file = nwb_file_io.read()

        # load ecog
        if data_type == 'raw':
            ecog = nwb_file.acquisition['LFP']
        elif data_type == 'high_gamma':  # this is for data that has been preprocessed through ecogVis
            ecog = nwb_file.processing['ecephys']['high_gamma']
        elif data_type == 'LFP':
            ecog = nwb_file.processing['ecephys']['LFP']['preprocessed']
        ecog_array = ecog.data.__array__()

        # get params of the data
        sfreq = ecog.rate
        ch_names = nwb_file.electrodes.label.data.__array__().tolist()
        bads_bool = nwb_file.electrodes['bad'].data.__array__()

        # set bads to zero
        ecog_array[:, bads_bool] = 0

        # zscore raw
        ecog_array = zscore(ecog_array, axis=0)

        # get time stamps (i think these aren't perfect -- check them)
        event_samples = nwb_file.intervals['TimeIntervals_speaker']['start_time'].data.__array__()
        event_times = event_samples * sfreq

        # time lock options
        if time_lock == 'POD' or time_lock == 'word_offset':
            event_shifts = list()
            for word, amb_level in log[['auditory_word', 'ambiguity']].values:
                shift = event_timing(word, amb_level, textgrid_dir, time_lock)
                event_shifts.append(int(shift*sfreq))
            event_shifts = np.array(event_shifts)

        else:
            event_shifts = np.zeros(len(log))

        # put this into an mne raw object
        info = create_info(ch_names, sfreq, ch_types='ecog')
        raw = RawArray(ecog_array.T, info)
        raw.info['bads'] = np.array(raw.info['ch_names'])[bads_bool].tolist()

        # make epochs
        events = np.array([[0, 0, 0]]*len(event_times))
        events[:, 0] = [int(t) for t in event_times] + event_shifts
        epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, metadata=log, preload=True,
                        baseline=None)
        epoch_list.append(epochs)

    # need to fix this -- bads need to be the same over blocks
    ep_info = epoch_list[0].info
    for eps in epoch_list:
        eps.info = ep_info

    # concatenate the epochs together
    epochs = concatenate_epochs(epoch_list)

    # add properties to metadata
    
    # mirror slider so that the numbers match the acoustics not the lexical
    acoustic_slider = list()
    binned_responses = list()
    flips = ['desolate', 'beneficial', 'mountains']
    for word_end, slider_resp, block_type in epochs.metadata[['word_end',
                                                              'slider.response',
                                                              'block_type']].values:

        # depends on the block type
        if block_type == 1:
            if word_end in flips:
                val = np.abs(slider_resp - 11)
            else:
                val = slider_resp
            acoustic_slider.append(val)

        # depends on block type
        if block_type == 2:
            if word_end in flips:
                val = slider_resp
            else:
                val = np.abs(slider_resp - 11)
            acoustic_slider.append(val)

        # a categorisation of behaviour into three classes
        if val < 2:
            binned_responses.append(1)
        elif val >= 2 and val < 8:
            binned_responses.append(2)
        elif val >= 8:
            binned_responses.append(3)

    epochs.metadata['binned_responses'] = binned_responses
    epochs.metadata['acoustic_slider'] = acoustic_slider

    return epochs
