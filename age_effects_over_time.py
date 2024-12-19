from scipy import io
from scipy.io import loadmat
from scipy import stats
import sys
sys.path.append("..")
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
import pingouin as pg
from create_folder import create_folder
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

groups = 34

basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
ngroups_dir = basedir + 'highpass_filtered_intercept2/' + str(groups) + 'groups/' #TODO
datadir = ngroups_dir + 'GSBS_results/searchlights/'
savedir = ngroups_dir + 'analyses_results/'
SL_dir = basedir + 'masks/searchlights/'

# collect results
ntime=192
nregs=5204
kfold_data=groups
maxk=96

data = loadmat(savedir + 'GSBS_obj.mat')
median_duration = data['median_duration']
bounds = data['bounds']
binbounds=bounds>0

# Get searchlight information and mask
stride=2
radius=3
min_vox=15
params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)

img = nib.load(basedir + 'masks/data_plus_GM_mask.nii')
affine = img.affine

coordinates = np.load(SL_dir + 'SL_voxelCoordinates_' + params_name + '.npy', allow_pickle=True)
searchlights = np.load(SL_dir + 'SL_voxelsIndices_' + params_name + '.npy', allow_pickle=True)

# get behavioral boundaries
ev_boundaries = loadmat(basedir + 'event_boundaries_subj.mat')
event_boundaries_subj = ev_boundaries['event_boundaries_subj']
event_boundaries = event_boundaries_subj.reshape((1, 191))

# get abs overlap
abs_overlap = np.load(basedir + 'highpass_filtered_intercept2/34groups/analyses_results/event_boundaries_binary/average_absoverlap.npy', allow_pickle=True)

# correlation age and median state duration
group = np.arange(groups)

age_dur = np.full([nregs], 0).astype(float)
pval_age_dur = np.full([nregs], 0).astype(float)

for SL in range(nregs):
    age_dur[SL], pval_age_dur[SL] = stats.spearmanr(group, median_duration[SL, :])

# correct for multiple testing median duration - these are the SL to include
# based on reject FALSE/TRUE get the indices for the significant elements and extract the original corresponding p-values - TRUE means sig effect of age on duration
reject, pvals_cmt = pg.multicomp(pval_age_dur, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
false_indices = [index for index, value in enumerate(~reject) if value]

# from all SL to SL with sig age effect (FDR corrected) x age group x binary boundaries per TR
indexes = np.array(true_indices)
selected_binbounds = binbounds[indexes]

# from all SL to SL with NO sig age effect (FDR corrected) x age group x binary boundaries per TR
indexes_f = np.array(false_indices)
selected_binbounds_f = binbounds[indexes_f]

# from all SL to SL with highest 10% overlap x age group x binary boundaries per TR
top_10_percent = int(len(abs_overlap) * 0.1)
indexes_10p = np.argsort(abs_overlap)[-top_10_percent:]
selected_binbounds_10p = binbounds[indexes_10p]

# mean over all included SL per group x TR
mean_selected_binbounds = np.mean(selected_binbounds, axis=0)
mean_selected_binbounds_f = np.mean(selected_binbounds_f, axis=0)
mean_selected_binbounds_10p = np.mean(selected_binbounds_10p, axis=0)

# correlation mean boundary presence across SL with age
age_bounds = np.full([ntime], 0).astype(float)
pval_age_bounds = np.full([ntime], 0).astype(float)
age_bounds_f = np.full([ntime], 0).astype(float)
pval_age_bounds_f = np.full([ntime], 0).astype(float)
age_bounds_10p = np.full([ntime], 0).astype(float)
pval_age_bounds_10p = np.full([ntime], 0).astype(float)

for T in range(ntime):
    age_bounds[T], pval_age_bounds[T] = stats.spearmanr(group, mean_selected_binbounds[:, T])
    age_bounds_f[T], pval_age_bounds_f[T] = stats.spearmanr(group, mean_selected_binbounds_f[:, T])
    age_bounds_10p[T], pval_age_bounds_10p[T] = stats.spearmanr(group, mean_selected_binbounds_10p[:, T])

# correlate presence of boundaries across selected SL in the average of all groups with the correlation between age and boundary presence
av_mean_selected_binbounds = np.mean(mean_selected_binbounds, axis=0)
age_av_bounds, pval_age_av_bounds = stats.spearmanr(age_bounds[1:], av_mean_selected_binbounds[1:])
av_mean_selected_binbounds_f = np.mean(mean_selected_binbounds_f, axis=0)
age_av_bounds_f, pval_age_av_bounds_f = stats.spearmanr(age_bounds_f[1:], av_mean_selected_binbounds_f[1:])
av_mean_selected_binbounds_10p = np.mean(mean_selected_binbounds_10p, axis=0)
age_av_bounds_10p, pval_age_av_bounds_10p = stats.spearmanr(age_bounds_10p[1:], av_mean_selected_binbounds_10p[1:])

# correlation with boundary presence x age on event boundraries and non-event boundaries
event_ind = np.where(event_boundaries_subj == 1)[0]
event_ind2 = event_ind+1 #because the first TR is missing for subj event boundaries
mean_selected_binbounds_event = selected_binbounds[:,:,event_ind2]
mean_selected_binbounds_event_f = selected_binbounds_f[:,:,event_ind2]
mean_selected_binbounds_event_10p = selected_binbounds_10p[:,:,event_ind2]
nonevent_ind = np.where(event_boundaries_subj == 0)[0]
nonevent_ind2 = nonevent_ind+1
mean_selected_binbounds_nonevent = selected_binbounds[:,:,nonevent_ind2]
mean_selected_binbounds_nonevent_f = selected_binbounds_f[:,:,nonevent_ind2]
mean_selected_binbounds_nonevent_10p = selected_binbounds_10p[:,:,nonevent_ind2]

mean_selected_binbounds_event_2d = np.mean(mean_selected_binbounds_event, axis=0)
mean_selected_binbounds_nonevent_2d = np.mean(mean_selected_binbounds_nonevent, axis=0)
mean_selected_binbounds_event_1d = np.mean(mean_selected_binbounds_event_2d, axis=1)
mean_selected_binbounds_nonevent_1d = np.mean(mean_selected_binbounds_nonevent_2d, axis=1)
mean_selected_binbounds_event_2d_f = np.mean(mean_selected_binbounds_event_f, axis=0)
mean_selected_binbounds_nonevent_2d_f = np.mean(mean_selected_binbounds_nonevent_f, axis=0)
mean_selected_binbounds_event_1d_f = np.mean(mean_selected_binbounds_event_2d_f, axis=1)
mean_selected_binbounds_nonevent_1d_f = np.mean(mean_selected_binbounds_nonevent_2d_f, axis=1)
mean_selected_binbounds_event_2d_10p = np.mean(mean_selected_binbounds_event_10p, axis=0)
mean_selected_binbounds_nonevent_2d_10p = np.mean(mean_selected_binbounds_nonevent_10p, axis=0)
mean_selected_binbounds_event_1d_10p = np.mean(mean_selected_binbounds_event_2d_10p, axis=1)
mean_selected_binbounds_nonevent_1d_10p = np.mean(mean_selected_binbounds_nonevent_2d_10p, axis=1)

age_bounds_event, pval_age_bounds_event = stats.spearmanr(group, mean_selected_binbounds_event_1d)
age_bounds_nonevent, pval_age_bounds_nonevent = stats.spearmanr(group, mean_selected_binbounds_nonevent_1d)
age_bounds_event_f, pval_age_bounds_event_f = stats.spearmanr(group, mean_selected_binbounds_event_1d_f)
age_bounds_nonevent_f, pval_age_bounds_nonevent_f = stats.spearmanr(group, mean_selected_binbounds_nonevent_1d_f)
age_bounds_event_10p, pval_age_bounds_event_10p = stats.spearmanr(group, mean_selected_binbounds_event_1d_10p)
age_bounds_nonevent_10p, pval_age_bounds_nonevent_10p = stats.spearmanr(group, mean_selected_binbounds_nonevent_1d_10p)

## plot data per age group

# Generate x values (time points)
x_values = np.arange(mean_selected_binbounds.shape[1])

# set color scale
colors = plt.cm.rainbow(np.linspace(0, 1, mean_selected_binbounds.shape[0]))

# fig size
plt.figure(figsize=(10, 6))

# Plot each group's data
for i, color in zip(range(mean_selected_binbounds.shape[0]), colors):
    #plt.scatter(x_values, mean_selected_binbounds[i], label=f'Group {i+1}', s=5, color=color) # s is the size of the dots
    #plt.scatter(x_values, mean_selected_binbounds_f[i], label=f'Group {i + 1}', s=5, color=color)  # s is the size of the dots
    plt.scatter(x_values, mean_selected_binbounds_10p[i], label=f'Group {i + 1}', s=5, color=color)  # s is the size of the dots

# Set axis labels
plt.xlabel('TR')
plt.ylabel('Boundary presence across searchlights')

# Set y-axis limits
plt.ylim(0, 1)

# Create colorbar
norm = Normalize(vmin=18, vmax=88)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='rainbow'), label='Mean age group')
cbar.set_ticks([19.8, 36.1, 51.9, 67.4, 85.2])  # Set custom tick positions
cbar.set_ticklabels(['19.8', '36.1', '51.9', '67.4', '85.2'])  # Set custom tick labels

# # Create custom legend handles for the first and last groups
# legend_handles = [
#     plt.Line2D([], [], marker='o', color='w', label=f'Group 1', markerfacecolor=colors[0], markersize=5),
#     plt.Line2D([], [], marker='o', color='w', label=f'Group {mean_selected_binbounds.shape[0]}', markerfacecolor=colors[-1], markersize=5)
# ]
#
# # Add the custom legend to the plot
# plt.legend(handles=legend_handles, loc='upper left')

# Hide legend
plt.legend().remove()

plt.title('Searchlight with the highest overlap - top 10%')

# Insert vertical lines based on subjective event boundaries
for i, val in enumerate(event_boundaries_subj):
    if val == 1:
        plt.axvline(x=i+1, color='black', linestyle='--', linewidth=0.5)

# Show plot
plt.show()

# plot data correlation age
x_values = np.arange(len(age_bounds))

# fig size
plt.figure(figsize=(10, 6))

# Plot age_bounds
#plt.plot(x_values, age_bounds)
#plt.plot(x_values, age_bounds_f)
plt.plot(x_values, age_bounds_10p)

# Set x-axis label
plt.xlabel('TR')

# Set y-axis label
plt.ylabel('Correlation age group x boundary presence across searchlights')

plt.title('Searchlight with the highest overlap - top 10%')

# Set y-axis limits
plt.ylim(-1, 1)

# Insert vertical lines based on subjective event boundaries
for i, val in enumerate(event_boundaries_subj):
    if val == 1:
        plt.axvline(x=i+1, color='black', linestyle='--', linewidth=0.5) # x=i+1 as the first TR is missing for the subjective event boundaries

# Show plot
plt.show()

# plot correlation between boundary presence average over groups with the effect of age on boundary presence
#plt.scatter(age_bounds[1:], av_mean_selected_binbounds[1:])
#plt.scatter(age_bounds_f[1:], av_mean_selected_binbounds_f[1:])
plt.scatter(age_bounds_f[1:], av_mean_selected_binbounds_10p[1:])

# Set x-axis label
plt.xlabel('Correlation between age group and boundary presence')
plt.ylabel('Boundary presence average across all groups and searchlights')

plt.title('Searchlight with the highest overlap - top 10%')

# Show plot
plt.show()

# correlation age and event vs non-event TRs
# plt.scatter(group, mean_selected_binbounds_event_1d, label='Event TRs, rho = -.88')
# plt.scatter(group, mean_selected_binbounds_nonevent_1d, label='Non-Event TRs, rho = -.95')
# plt.scatter(group, mean_selected_binbounds_event_1d_f, label='Event TRs, rho = -.58')
# plt.scatter(group, mean_selected_binbounds_nonevent_1d_f, label='Non-Event TRs, rho = -.63')
plt.scatter(group, mean_selected_binbounds_event_1d_10p, label='Event TRs, rho = -.61')
plt.scatter(group, mean_selected_binbounds_nonevent_1d_10p, label='Non-Event TRs, rho = -.75')
plt.xlabel('Age group')
plt.ylabel('Boundary presence across TRs and searchlights')
plt.legend(loc='upper right')
plt.title('Searchlight with the highest overlap - top 10%')
plt.show()

print


