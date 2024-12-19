import numpy as np
import os
from scipy.io import loadmat
import nibabel as nib
from scipy import stats
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Directories
base_dir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
group_dir = base_dir + 'highpass_filtered_intercept2/34groups/'
save_dir = group_dir + 'analyses_results/'
eventLoc_dir = save_dir + 'event_boundaries_binary_1swindow/'
SL_dir = base_dir + 'masks/searchlights/'

# base variables
ntime=192
nregs=5204
groups=34

# GSBS data
data = loadmat(save_dir + 'GSBS_obj.mat')
median_duration = data['median_duration']
bounds = data['bounds']
binbounds=bounds>0

# image shape for visualization
stride=2
radius=3
min_vox=15
params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)

img = nib.load(base_dir + 'masks/data_plus_GM_mask.nii')
affine = img.affine

# Get searchlight information
coordinates = np.load(SL_dir + 'SL_voxelCoordinates_' + params_name + '.npy', allow_pickle=True)
searchlights = np.load(SL_dir + 'SL_voxelsIndices_' + params_name + '.npy', allow_pickle=True)

# Load the saved variables for (non)events
# get 19 original behavioral boundaries
#ev_boundaries = loadmat(basedir + 'event_boundaries_subj.mat')
#event_boundaries_subj = ev_boundaries['event_boundaries_subj']
#event_boundaries = event_boundaries_subj.reshape((1, 191))
# with 1s window 19 -> 36 TRs
event_loc_all = np.load(os.path.join(eventLoc_dir, "event_loc_all_vector.npy"))
non_event_loc_all = np.load(os.path.join(eventLoc_dir, "non_event_loc_all_vector.npy"))

# Use the vectors where the events and non-events are to create 2 separate data frames
# All SL x age group x binary boundaries per event TRs
selected_binbounds_events = binbounds[:, :, event_loc_all]
# mean over event TRs -> SL x group
mean_selected_binbounds_events = np.mean(selected_binbounds_events, axis=2)

# All SL x age group x binary boundaries per non-event TRs
selected_binbounds_nonevents = binbounds[:, :, non_event_loc_all]
# mean over non-event TRs -> SL x group
mean_selected_binbounds_nonevents = np.mean(selected_binbounds_nonevents, axis=2)

# Create vector for age group
group = np.arange(groups)

# Empty arrays for median duration
age_boundaryPresence_events = np.full([nregs], 0).astype(float)
pval_age_boundaryPresence_events = np.full([nregs], 0).astype(float)
age_boundaryPresence_nonevents = np.full([nregs], 0).astype(float)
pval_age_boundaryPresence_nonevents = np.full([nregs], 0).astype(float)

# Loop over all SLs to calculate correlation age group and boundary presence
# Separately for TRs that overlap with events and not
for SL in range(nregs):
    # Calculate the Spearman correlation between the ordinal (group) and continuous variable (median duration)
    age_boundaryPresence_events[SL], pval_age_boundaryPresence_events[SL] = stats.spearmanr(group, mean_selected_binbounds_events[SL,:])
    age_boundaryPresence_nonevents[SL], pval_age_boundaryPresence_nonevents[SL] = stats.spearmanr(group,
                                                                                            mean_selected_binbounds_nonevents[
                                                                                            SL, :])

mean_age_boundaryPresence_events = np.mean(age_boundaryPresence_events)
print("Mean correlation Age group x Boundary Presence for Events:", mean_age_boundaryPresence_events)
mean_age_boundaryPresence_nonevents = np.mean(age_boundaryPresence_nonevents)
print("Mean correlation Age group x Boundary Presence for NonEvents:", mean_age_boundaryPresence_nonevents)

# PLot correlation
# Compute the mean boundary presence for each age group
mean_boundary_presence_events = mean_selected_binbounds_events.mean(axis=0)
mean_boundary_presence_nonevents = mean_selected_binbounds_nonevents.mean(axis=0)

# Create the scatter plot
plt.figure(figsize=(10, 6))

# Scatter points for events
scatter_events = plt.scatter(group, mean_boundary_presence_events, color='blue', alpha=0.7, label='Events')
# Scatter points for non-events
scatter_nonevents = plt.scatter(group, mean_boundary_presence_nonevents, color='green', alpha=0.7, label='Non-Events')

# Custom legend handles for correlation
correlation_events = plt.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=6,
                               label=r'Events $r_s$: {:.2f}'.format(mean_age_boundaryPresence_events))
correlation_nonevents = plt.Line2D([], [], color='green', marker='o', linestyle='None', markersize=6,
                                  label=r'Non-Events $r_s$: {:.2f}'.format(mean_age_boundaryPresence_nonevents))



# Combine handles for the legend
handles = [correlation_events, correlation_nonevents]

# Add combined legend with explicit handles
plt.legend(handles=handles, loc='upper right', frameon=True, fontsize=14)

plt.xlabel('Age group', fontsize=16)
plt.ylabel('Mean boundary presence', fontsize=16)
plt.title('Mean boundary presence across searchlights by age group for events and non-events', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(save_dir + 'mean_boundary_presence_by_age.pdf', format='pdf', dpi=300)
plt.show()


diff_boundaryPresence = age_boundaryPresence_events - age_boundaryPresence_nonevents

# Plot on brain
# From SL to voxel to create .niis
x_max, y_max, z_max = img.shape
counter = np.zeros((x_max, y_max, z_max))

ageEvents_sum = np.zeros((x_max, y_max, z_max))
ageNonEvents_sum = np.zeros((x_max, y_max, z_max))
ageDiffEvents_sum = np.zeros((x_max, y_max, z_max))

for SL_idx, voxel_indices in enumerate(searchlights):
    for vox in voxel_indices:
        x, y, z = coordinates[vox]
        counter[x, y, z] += 1
        ageEvents_sum[x, y, z] += age_boundaryPresence_events[SL_idx]
        ageNonEvents_sum[x, y, z] += age_boundaryPresence_nonevents[SL_idx]
        ageDiffEvents_sum[x, y, z] += diff_boundaryPresence[SL_idx]

# Take mean across searchlights
mean_ageEvents_sum = np.divide(ageEvents_sum, counter)
mean_ageNonEvents_sum = np.divide(ageNonEvents_sum, counter)
mean_ageDiffEvents_sum = np.divide(ageDiffEvents_sum, counter)

# Convert to nifti - unthresholded
map_nifti = nib.Nifti1Image(mean_ageEvents_sum, affine)
nib.save(map_nifti, save_dir + 'correlation_age_EventTRs.nii')

map_nifti = nib.Nifti1Image(mean_ageNonEvents_sum, affine)
nib.save(map_nifti, save_dir + 'correlation_age_NonEventTRs.nii')

map_nifti = nib.Nifti1Image(mean_ageDiffEvents_sum, affine)
nib.save(map_nifti, save_dir + 'correlation_age_Diff_EventNonEventTRs.nii')

# How do each of the correlation maps between age and boundary presence correlate with absolute overlap?

# Load the saved variable absolute overlap 1swindow
abs_overlap = np.load(os.path.join(eventLoc_dir, "abs_overlap_vector.npy"))

correlation_events, p_value_events = spearmanr(age_boundaryPresence_events, abs_overlap)
print("Correlation Events:", correlation_events)
print("P-value Events:", p_value_events)

correlation_nonevents, p_value_nonevents = spearmanr(age_boundaryPresence_nonevents, abs_overlap)
print("Correlation NonEvents:", correlation_nonevents)
print("P-value NonEvents:", p_value_nonevents)

