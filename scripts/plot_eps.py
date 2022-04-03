# analyse barakeet data

import numpy as np
import matplotlib.pyplot as plt
from data_io import load_epochs

# paths
root_dir = '/Users/lauragwilliams/Dropbox (UCSF Department of Neurological Surgery)/lg/barakeet'
plot_dir = '%s/vis' % (root_dir)
textgrid_dir = '%s/external/TextGrids' % (root_dir)

# params
# subject = 'EC237'
# blocks = ['28']

subject = 'EC243'
blocks = ['39', '40', '47', '48']
data_type = 'high_gamma'
time_lock = 'word_onset'
tmin = -0.5
tmax = 1.0

# roi dict
roi_dict = {'EC237': {'Suprasylvian': [1, 128],
                  'Lateral Temporal': [129, 256],
                  'Posterior SubTemporal': [257, 320],
                  'Insula':	[321, 330],
                  'Hippocampus': [331, 340],
                  'Amygdala': [341, 350]},

            'EC243': {'pSTG': [80, 77],
                      'STG': [91, 73],
                      'aSTG': [68, 65],
                      'MTG': [43, 10]}

                  }

# load data
epochs = load_epochs(subject, blocks, data_type, time_lock, tmin, tmax)
pairs = epochs.metadata['phoneme_pair'].unique()
n_pairs = len(pairs)
morphs = np.sort(epochs.metadata['resampled'].unique())
n_morphs = len(morphs)

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

    if val < 2:
        binned_responses.append(1)
    elif val >= 2 and val < 8:
        binned_responses.append(2)
    elif val >= 8:
        binned_responses.append(3)

epochs.metadata['binned_responses'] = binned_responses
epochs.metadata['acoustic_slider'] = acoustic_slider

# plot the continuum
plt.close()
lim = 2
fig, axes = plt.subplots(n_pairs, n_morphs)
for pi, pair in enumerate(pairs):
    for mi, morph in enumerate(morphs):

        # get average for this morph and pair
        d_subset = epochs["phoneme_pair == '%s' and resampled == %s" % (pair, morph)]
        axes[pi, mi].imshow(d_subset._data.mean(0), cmap='RdBu_r', vmin=-lim, vmax=lim)
        axes[pi, mi].set_title('%s, %s' % (pair, morph))
plt.show()

# plot a select electrode
times = epochs.times
cols = plt.cm.RdBu_r(np.linspace(0.1, 0.9, n_morphs))

elec = 87

fig, ax = plt.subplots(n_pairs, 1)

# get roi
for roi in roi_dict[subject].keys():
    emin, emax = roi_dict[subject][roi]
    if elec+1 >= emin and elec < emax:
        this_roi = roi

this_roi = 'TBD'

for pi, pair in enumerate(pairs):
    for mi, morph in enumerate(morphs):

        # get average for this morph and pair
        d_subset = epochs["phoneme_pair == '%s' and resampled == %s" % (pair, morph)]
        ax[pi].plot(times, d_subset._data.mean(0)[elec, :], color=cols[mi])
        ax[pi].set_title('%s %s %s' % (pair, elec, this_roi))
plt.show()

elec = 102
elec = 87

# for elec in range(282):
for elec in [102, 87]:

    # plot split by behaviour
    fig, ax = plt.subplots(n_pairs, 1)
    cols = ['r', 'orange', 'green']
    for pi, pair in enumerate(pairs):

        # get average for this morph and pair

        # plot left
        for v, binned in enumerate([1, 2, 3]):
            d_subset = epochs["phoneme_pair == '%s' and binned_responses == %s" % (pair, binned)]
            ax[pi].plot(times, d_subset._data.mean(0)[elec, :], color=cols[v])
        ax[pi].set_title('%s %s %s' % (pair, elec, this_roi))
    plt.show()

# plot whole grid
