import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mayavi import mlab

import mne
from mne.viz import plot_alignment, snapshot_brain_montage

# params
subject = 'EC243'

# paths
base_dir = '/Users/lauragwilliams/Dropbox (UCSF Department of Neurological Surgery)/lg/barakeet/data'
subjects_dir = '%s/imaging' % (base_dir)

# path to electrodes
elec_fname = '%s/%s/elecs/TDT_elecs_all.mat' % (subjects_dir, subject)

# extract elec positions
mat = loadmat(elec_fname)
elec = mat['elecmatrix'] * 0.001
n_elecs, n_dims = elec.shape
ch_names = [str(i) for i in range(n_elecs)]

# ch_names = mat['eleclabels'].tolist()
# ch_names = [ch_names[ii][0][0] for ii in range(len(ch_names))]

# remove nans
nan_idx = np.isfinite(elec.mean(1))
elec = elec[nan_idx]
ch_names = np.array(ch_names)[nan_idx].tolist()

# make mne montage
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                        coord_frame='unknown')

# and info
info = mne.create_info(ch_names, 1000., 'ecog').set_montage(montage)

# We'll once again plot the surface, then take a snapshot.
# fig = mlab.figure(bgcolor=(1., 1., 1.))
fig_scatter = plot_alignment(info,
                             subject=subject,
                             subjects_dir=subjects_dir,
                             surfaces='pial')
mne.viz.set_3d_view(fig_scatter, 180, 80)
xy, im = snapshot_brain_montage(fig_scatter, montage)

# Convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in info['ch_names']])

# Define an arbitrary "activity" pattern for viz
# activity = np.linspace(180, 100, xy_pts.shape[0])
activity = np.zeros(xy_pts.shape[0])

anat_elecs = {
              # 'pSTG': [2, 3, 19, 20, 35, 36],
              # 'aSTG': [99, 100, 101, 115, 116, 117],
              # 'IFG': [200, 201, 202, 203, 204],
              # 'motor': [28, 29, 30, 44, 45, 46],
              'other': [#65, 66, 67, 78, 94, 93,
                        325]}

for roi in anat_elecs.keys():
    elecs = anat_elecs[roi]
    activity[elecs] = 1
    # This allows us to use matplotlib to create arbitrary 2d scatterplots
    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(*xy_pts.T, c=activity, s=100, cmap='coolwarm')
    ax.set_axis_off()
    # ax.imshow(im)
plt.show()
