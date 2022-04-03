# analyse barakeet data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from data_io import load_epochs


# paths
root_dir = '/Users/lauragwilliams/Dropbox (UCSF Department of Neurological Surgery)/lg/barakeet'
plot_dir = '%s/vis' % (root_dir)
textgrid_dir = '%s/external/TextGrids' % (root_dir)

# params
subject = 'EC237'
blocks = ['28']
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
                  'Amygdala': [341, 350]}}

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
for word_end, slider_resp in epochs.metadata[['word_end', 'slider.response']].values:
    if word_end in flips:
        val = np.abs(slider_resp - 11)
    else:
        val = slider_resp
    acoustic_slider.append(val)
    if val < 2:
        binned_responses.append(1)
    elif val >= 2 and val < 8:
        binned_responses.append(2)
    elif val >= 8:
        binned_responses.append(3)
epochs.metadata['binned_responses'] = binned_responses
epochs.metadata['acoustic_slider'] = acoustic_slider
epochs.metadata['ambiguous'] = (epochs.metadata['binned_responses'] == 2)*1

# fit encoding model over time and space
features = ['ambiguous', 'slider.response', 'resampled', 'acoustic_slider']
X = epochs._data
Y = epochs.metadata[features].values

# make clf 
reg = make_pipeline(StandardScaler(), LinearRegression())
reg.score(X, y)

reg.coef_
