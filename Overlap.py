import numpy as np
import nibabel as nib
from scipy.io import loadmat
from scipy.stats import ranksums
from scipy import stats
import pingouin as pg
import os
import pandas as pd
from compute_overlap import compute_overlap
from compute_overlap_1swindow import compute_overlap_1swindow
from tqdm import tqdm
import matplotlib.pyplot as plt
from create_folder import create_folder

groups=34

Nregs = 5204
Ntime = 192
basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
ngroups_dir = basedir + 'highpass_filtered_intercept2/' + str(groups) + 'groups/'
savedir = ngroups_dir + 'analyses_results/event_boundaries_binary_1swindow/' #TODO
create_folder(savedir)

# get neural boundaries for all groups for all searchlights
data = loadmat(ngroups_dir + 'analyses_results/GSBS_obj.mat')
#for binary #TODO select right option for binary or strengths
bounds = data['bounds']
binbounds=bounds>0
#for strength
#binbounds = data['strengths']

# get behavioral boundaries
ev_boundaries = loadmat(basedir + 'event_boundaries_subj.mat')
event_boundaries_subj = ev_boundaries['event_boundaries_subj']
event_boundaries = event_boundaries_subj.reshape((1, 191))
# get behavioral boundaries +1s
ev_boundaries_p1s = loadmat(basedir + 'event_boundaries_subj_p1s.mat')
event_boundaries_subj_p1s = ev_boundaries_p1s['event_boundaries_subj_p1s']
event_boundaries_p1s = event_boundaries_subj_p1s.reshape((1, 191))
# get behavioral boundaries -1s
ev_boundaries_m1s = loadmat(basedir + 'event_boundaries_subj_m1s.mat')
event_boundaries_subj_m1s = ev_boundaries_m1s['event_boundaries_subj_m1s']
event_boundaries_m1s = event_boundaries_subj_m1s.reshape((1, 191))

events_loc_s0 = np.where(event_boundaries > 0)[1]
events_loc_sp1 = np.where(event_boundaries_p1s > 0)[1]
events_loc_sm1 = np.where(event_boundaries_m1s > 0)[1]

event_loc_all = np.unique(np.append(np.append(events_loc_s0,events_loc_sp1),events_loc_sm1))
non_event_loc_all = np.setdiff1d(np.arange(0,192),event_loc_all)
np.save(os.path.join(savedir, "event_loc_all_vector.npy"), event_loc_all)
np.save(os.path.join(savedir, "non_event_loc_all_vector.npy"), non_event_loc_all)

mean_evbounds = np.mean(binbounds[:,:,event_loc_all],2)
mean_nonevbounds = np.mean(binbounds[:,:,non_event_loc_all],2)
# use this as input for correlation with age

# get gray matter mask for dimensions to plot
img = nib.load(basedir + 'masks/data_plus_GM_mask.nii')
affine = img.affine

# get searchlight information
stride=2
radius=3
min_vox=15
params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)
SL_dir = basedir + 'masks/searchlights/'
coordinates = np.load(SL_dir + 'SL_voxelCoordinates_' + params_name + '.npy', allow_pickle=True)
searchlights = np.load(SL_dir + 'SL_voxelsIndices_' + params_name + '.npy', allow_pickle=True)

# get ISS per group per searchlight
ISS_dir = ngroups_dir + 'preGSBS/age_groups/'
ISS = np.load(ISS_dir + 'all_groups_mean_correlation_per_SL.npy')

# create group variable for correlation
group = np.arange(groups)

# create empty arrays for overlap
rel_overlap_events=np.full([Nregs, groups], 0).astype(float)
abs_overlap_events=np.full([Nregs, groups], 0).astype(float)
corr_events=np.full([Nregs, groups], 0).astype(float)
# first TR doesn't count because can't be a boundary
time_range = slice(1, None)
# correlation / overlap over time per group per searchlight
for gr in range(groups):
    group_data = binbounds[:, gr, time_range]
    for sl in range(Nregs):
        # simple correlation between neural boundaries and subj event boundaries
        correlations = np.corrcoef(group_data[sl,:], event_boundaries, rowvar=False)
        corr_events[sl, gr] = correlations[0, 1]
    # overlap between neural boundaries and subj event boundaries
    #rel_overlap = compute_overlap(1, group_data, event_boundaries)
    rel_overlap = compute_overlap_1swindow(1, group_data, event_boundaries,event_boundaries_p1s,event_boundaries_m1s)
    rel_overlap_events[:, gr] = rel_overlap
    #abs_overlap = compute_overlap(2, group_data, event_boundaries)
    abs_overlap = compute_overlap_1swindow(2, group_data, event_boundaries,event_boundaries_p1s,event_boundaries_m1s)
    abs_overlap_events[:, gr] = abs_overlap

# get average values for overlap across groups
tot_corr_events = np.mean(corr_events, axis=1)
tot_rel_overlap_events= np.mean(rel_overlap_events, axis=1)
tot_abs_overlap_events = np.mean(abs_overlap_events, axis=1)

# save absolute overlap per SL to correlate with age_dur
np.save(os.path.join(savedir, "abs_overlap_vector.npy"), tot_abs_overlap_events)

# which regions show significant overlap with events across groups?
pval_corr_events = np.zeros(Nregs)
pval_rel_events = np.zeros(Nregs)
pval_abs_events = np.zeros(Nregs)

zero = np.zeros(groups)

for i in range(Nregs):
    pval_corr_event = ranksums(corr_events[i, :], zero)
    pval_corr_events[i] = pval_corr_event[1]
    pval_rel_event = ranksums(rel_overlap_events[i, :], zero)
    pval_rel_events[i] = pval_rel_event[1]
    pval_abs_event = ranksums(abs_overlap_events[i, :], zero)
    pval_abs_events[i] = pval_abs_event[1]

reject, pval_cmt = pg.multicomp(pval_corr_events, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_corr_events = [pval_corr_events[index] for index in true_indices]

reject2, pval_cmt2 = pg.multicomp(pval_rel_events, method='fdr_bh')
true_indices2 = [index for index, value in enumerate(reject2) if value]
cmt_pval_rel_events = [pval_rel_events[index] for index in true_indices2]

reject3, pval_cmt3 = pg.multicomp(pval_abs_events, method='fdr_bh')
true_indices3 = [index for index, value in enumerate(reject3) if value]
cmt_pval_abs_events = [pval_abs_events[index] for index in true_indices3]

# from SL to voxel
x_max, y_max, z_max = img.shape # based on gray matter img
counter = np.zeros((x_max, y_max, z_max))
# for all 3 variables an empty array for sum for the correlation and sum for the p-value
tot_corr_events_sum = np.zeros((x_max, y_max, z_max))
pval_corr_events_sum = np.zeros((x_max, y_max, z_max))
tot_rel_events_sum = np.zeros((x_max, y_max, z_max))
pval_rel_events_sum = np.zeros((x_max, y_max, z_max))
tot_abs_events_sum = np.zeros((x_max, y_max, z_max))
pval_abs_events_sum = np.zeros((x_max, y_max, z_max))

for SL_idx, voxel_indices in enumerate(tqdm(searchlights)):
    for vox in voxel_indices:
        x, y, z = coordinates[vox]
        counter[x, y, z] += 1
        tot_corr_events_sum[x, y, z] += tot_corr_events[SL_idx]
        pval_corr_events_sum[x,y,z] += pval_corr_events[SL_idx]
        tot_rel_events_sum[x, y, z] += tot_rel_overlap_events[SL_idx]
        pval_rel_events_sum[x,y,z] += pval_rel_events[SL_idx]
        tot_abs_events_sum[x, y, z] += tot_abs_overlap_events[SL_idx]
        pval_abs_events_sum[x,y,z] += pval_abs_events[SL_idx]

# Take mean across searchlights
mean_corr_events = np.divide(tot_corr_events_sum, counter)
mean_pval_corr_events = np.divide(pval_corr_events_sum, counter)
mean_rel_events = np.divide(tot_rel_events_sum, counter)
mean_pval_rel_events = np.divide(pval_rel_events_sum, counter)
mean_abs_events = np.divide(tot_abs_events_sum, counter)
mean_pval_abs_events = np.divide(pval_abs_events_sum, counter)

map_nifti = nib.Nifti1Image(mean_abs_events, affine)
nib.save(map_nifti, savedir + 'analysis_abs_events.nii')
map_nifti = nib.Nifti1Image(mean_rel_events, affine)
nib.save(map_nifti, savedir + 'analysis_rel_events.nii')

# Create map of <0.05 - corrected for multiple testing at SL level
if cmt_pval_corr_events:
    idx_keep = np.where(mean_pval_corr_events < max(cmt_pval_corr_events))
    mean_corr_events_cmt = np.zeros_like(mean_corr_events) * np.nan
    mean_corr_events_cmt[idx_keep] = mean_corr_events[idx_keep]
    # save as .nii
    map_nifti = nib.Nifti1Image(mean_corr_events_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_corr_events_cmt.nii')

if cmt_pval_rel_events:
    idx_keep = np.where(mean_pval_rel_events < max(cmt_pval_rel_events))
    mean_rel_events_cmt = np.zeros_like(mean_rel_events) * np.nan
    mean_rel_events_cmt[idx_keep] = mean_rel_events[idx_keep]
    # save as .nii
    map_nifti = nib.Nifti1Image(mean_rel_events_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_rel_events_cmt.nii')

if cmt_pval_abs_events:
    idx_keep = np.where(mean_pval_abs_events < max(cmt_pval_abs_events))
    mean_abs_events_cmt = np.zeros_like(mean_abs_events) * np.nan
    mean_abs_events_cmt[idx_keep] = mean_abs_events[idx_keep]
    # save as .nii
    map_nifti = nib.Nifti1Image(mean_abs_events_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_abs_events_cmt.nii')

#from here correlate with age and ISS as covar
# correlation between age group and events - 6 in total
# create empty arrays
age_corr_events=np.full([Nregs], 0).astype(float)
age_rel_events=np.full([Nregs], 0).astype(float)
age_abs_events=np.full([Nregs], 0).astype(float)
pval_age_corr_events=np.full([Nregs], 0).astype(float)
pval_age_rel_events=np.full([Nregs], 0).astype(float)
pval_age_abs_events=np.full([Nregs], 0).astype(float)
# with ISS as covariate
age_corr_events_iss=np.full([Nregs], 0).astype(float)
age_rel_events_iss=np.full([Nregs], 0).astype(float)
age_abs_events_iss=np.full([Nregs], 0).astype(float)
pval_age_corr_events_iss=np.full([Nregs], 0).astype(float)
pval_age_rel_events_iss=np.full([Nregs], 0).astype(float)
pval_age_abs_events_iss=np.full([Nregs], 0).astype(float)
# correlation per SL
for SL in range(Nregs):
    # Calculate the Spearman correlation between the ordinal (group) and continuous variable ( 3 types of event overlap)
    age_corr_events[SL], pval_age_corr_events[SL] = stats.spearmanr(group, corr_events[SL, :])
    age_rel_events[SL], pval_age_rel_events[SL] = stats.spearmanr(group, rel_overlap_events[SL, :])
    age_abs_events[SL], pval_age_abs_events[SL] = stats.spearmanr(group, abs_overlap_events[SL, :])
    age_abs_events_nonevent[SL], pval_age_abs_events_nonevent[SL] = stats.spearmanr(group, mean_nonevbounds[SL, :])
    age_abs_events_event[SL], pval_age_abs_events_event[SL] = stats.spearmanr(group, mean_evbounds[SL, :])
    # Calculate partial correlation with ISS as covariate
    data = {'x':group, 'y':corr_events[SL, :], 'cv1':ISS[SL,:]}
    df = pd.DataFrame(data)
    partial_spearman_corr = pg.partial_corr(data=df, x='x', y='y', covar='cv1', method='spearman')
    age_corr_events_iss[SL] = partial_spearman_corr['r']
    pval_age_corr_events_iss[SL] = partial_spearman_corr['p-val']
    data = {'x':group, 'y':rel_overlap_events[SL, :], 'cv1':ISS[SL,:]}
    df = pd.DataFrame(data)
    partial_spearman_corr = pg.partial_corr(data=df, x='x', y='y', covar='cv1', method='spearman')
    age_rel_events_iss[SL] = partial_spearman_corr['r']
    pval_age_rel_events_iss[SL] = partial_spearman_corr['p-val']
    data = {'x':group, 'y':abs_overlap_events[SL, :], 'cv1':ISS[SL,:]}
    df = pd.DataFrame(data)
    partial_spearman_corr = pg.partial_corr(data=df, x='x', y='y', covar='cv1', method='spearman')
    age_abs_events_iss[SL] = partial_spearman_corr['r']
    pval_age_abs_events_iss[SL] = partial_spearman_corr['p-val']

# correct for multiple testing for all 6 variables (3 different methods of overlap, followed by with ISS as covar)
reject, pval_cmt = pg.multicomp(pval_age_corr_events, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_corr_events = [pval_age_corr_events[index] for index in true_indices]

reject, pval_cmt = pg.multicomp(pval_age_rel_events, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_rel_events = [pval_age_rel_events[index] for index in true_indices]

reject, pval_cmt = pg.multicomp(pval_age_abs_events, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_abs_events = [pval_age_abs_events[index] for index in true_indices]

reject, pval_cmt = pg.multicomp(pval_age_corr_events_iss, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_corr_events_iss = [pval_age_corr_events_iss[index] for index in true_indices]

reject, pval_cmt = pg.multicomp(pval_age_rel_events_iss, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_rel_events_iss = [pval_age_rel_events_iss[index] for index in true_indices]

reject, pval_cmt = pg.multicomp(pval_age_abs_events_iss, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_abs_events_iss = [pval_age_abs_events_iss[index] for index in true_indices]

# from SL to voxel
x_max, y_max, z_max = img.shape # based on gray matter img
counter = np.zeros((x_max, y_max, z_max))
# for all 6 variables an empty array for sum for the correlation and sum for the p-value
age_corr_events_sum = np.zeros((x_max, y_max, z_max))
pval_age_corr_events_sum = np.zeros((x_max, y_max, z_max))
age_rel_events_sum = np.zeros((x_max, y_max, z_max))
pval_age_rel_events_sum = np.zeros((x_max, y_max, z_max))
age_abs_events_sum = np.zeros((x_max, y_max, z_max))
pval_age_abs_events_sum = np.zeros((x_max, y_max, z_max))
age_corr_events_iss_sum = np.zeros((x_max, y_max, z_max))
pval_age_corr_events_iss_sum = np.zeros((x_max, y_max, z_max))
age_rel_events_iss_sum = np.zeros((x_max, y_max, z_max))
pval_age_rel_events_iss_sum = np.zeros((x_max, y_max, z_max))
age_abs_events_iss_sum = np.zeros((x_max, y_max, z_max))
pval_age_abs_events_iss_sum = np.zeros((x_max, y_max, z_max))

for SL_idx, voxel_indices in enumerate(tqdm(searchlights)):
    for vox in voxel_indices:
        x, y, z = coordinates[vox]
        counter[x, y, z] += 1
        age_corr_events_sum[x, y, z] += age_corr_events[SL_idx]
        pval_age_corr_events_sum[x,y,z] += pval_age_corr_events[SL_idx]
        age_rel_events_sum[x, y, z] += age_rel_events[SL_idx]
        pval_age_rel_events_sum[x,y,z] += pval_age_rel_events[SL_idx]
        age_abs_events_sum[x, y, z] += age_abs_events[SL_idx]
        pval_age_abs_events_sum[x,y,z] += pval_age_abs_events[SL_idx]
        age_corr_events_iss_sum[x, y, z] += age_corr_events_iss[SL_idx]
        pval_age_corr_events_iss_sum[x,y,z] += pval_age_corr_events_iss[SL_idx]
        age_rel_events_iss_sum[x, y, z] += age_rel_events_iss[SL_idx]
        pval_age_rel_events_iss_sum[x,y,z] += pval_age_rel_events_iss[SL_idx]
        age_abs_events_iss_sum[x, y, z] += age_abs_events_iss[SL_idx]
        pval_age_abs_events_iss_sum[x,y,z] += pval_age_abs_events_iss[SL_idx]

# Take mean across searchlights
mean_age_corr_events = np.divide(age_corr_events_sum, counter)
mean_pval_age_corr_events = np.divide(pval_age_corr_events_sum, counter)
mean_age_rel_events = np.divide(age_rel_events_sum, counter)
mean_pval_age_rel_events = np.divide(pval_age_rel_events_sum, counter)
mean_age_abs_events = np.divide(age_abs_events_sum, counter)
mean_pval_age_abs_events = np.divide(pval_age_abs_events_sum, counter)
mean_age_corr_events_iss = np.divide(age_corr_events_iss_sum, counter)
mean_pval_age_corr_events_iss = np.divide(age_corr_events_iss_sum, counter)
mean_age_rel_events_iss = np.divide(age_rel_events_iss_sum, counter)
mean_pval_age_rel_events_iss = np.divide(pval_age_rel_events_iss_sum, counter)
mean_age_abs_events_iss = np.divide(age_abs_events_iss_sum, counter)
mean_pval_age_abs_events_iss = np.divide(pval_age_abs_events_iss_sum, counter)

# # Create map of <0.05 - uncorrected for multiple testing
idx_keep = np.where(mean_pval_age_corr_events < 0.05)
mean_age_corr_events_pthresh = np.zeros_like(mean_age_corr_events) * np.nan
mean_age_corr_events_pthresh[idx_keep] = mean_age_corr_events[idx_keep]
# save as .nii
map_nifti = nib.Nifti1Image(mean_age_corr_events_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_corr_events_pthresh.nii')

idx_keep = np.where(mean_pval_age_rel_events < 0.05)
mean_age_rel_events_pthresh = np.zeros_like(mean_age_rel_events) * np.nan
mean_age_rel_events_pthresh[idx_keep] = mean_age_rel_events[idx_keep]
# save as .nii
map_nifti = nib.Nifti1Image(mean_age_rel_events_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_rel_events_pthresh.nii')

idx_keep = np.where(mean_pval_age_abs_events < 0.05)
mean_age_abs_events_pthresh = np.zeros_like(mean_age_abs_events) * np.nan
mean_age_abs_events_pthresh[idx_keep] = mean_age_abs_events[idx_keep]
# save as .nii
map_nifti = nib.Nifti1Image(mean_age_abs_events_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_abs_events_pthresh.nii')

idx_keep = np.where(mean_pval_age_corr_events_iss < 0.05)
mean_age_corr_events_iss_pthresh = np.zeros_like(mean_age_corr_events_iss) * np.nan
mean_age_corr_events_iss_pthresh[idx_keep] = mean_age_corr_events_iss[idx_keep]
# save as .nii
map_nifti = nib.Nifti1Image(mean_age_corr_events_iss_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_corr_events_iss_pthresh.nii')

idx_keep = np.where(mean_pval_age_rel_events_iss < 0.05)
mean_age_rel_events_iss_pthresh = np.zeros_like(mean_age_rel_events_iss) * np.nan
mean_age_rel_events_iss_pthresh[idx_keep] = mean_age_rel_events_iss[idx_keep]
# save as .nii
map_nifti = nib.Nifti1Image(mean_age_rel_events_iss_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_rel_events_iss_pthresh.nii')

idx_keep = np.where(mean_pval_age_abs_events_iss < 0.05)
mean_age_abs_events_iss_pthresh = np.zeros_like(mean_age_abs_events_iss) * np.nan
mean_age_abs_events_iss_pthresh[idx_keep] = mean_age_abs_events_iss[idx_keep]
# save as .nii
map_nifti = nib.Nifti1Image(mean_age_abs_events_iss_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_abs_events_iss_pthresh.nii')

# Create map of <0.05 - corrected for multiple testing at SL level
if cmt_pval_age_corr_events:
    idx_keep = np.where(mean_pval_age_corr_events < max(cmt_pval_age_corr_events))
    mean_age_corr_events_cmt = np.zeros_like(mean_age_corr_events) * np.nan
    mean_age_corr_events_cmt[idx_keep] = mean_age_corr_events[idx_keep]
    # save as .nii
    map_nifti = nib.Nifti1Image(mean_age_corr_events_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_corr_events_cmt.nii')

if cmt_pval_age_rel_events:
    idx_keep = np.where(mean_pval_age_rel_events < max(cmt_pval_age_rel_events))
    mean_age_rel_events_cmt = np.zeros_like(mean_age_rel_events) * np.nan
    mean_age_rel_events_cmt[idx_keep] = mean_age_rel_events[idx_keep]
    # save as .nii
    map_nifti = nib.Nifti1Image(mean_age_rel_events_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_rel_events_cmt.nii')

if cmt_pval_age_abs_events:
    idx_keep = np.where(mean_pval_age_abs_events < max(cmt_pval_age_abs_events))
    mean_age_abs_events_cmt = np.zeros_like(mean_age_abs_events) * np.nan
    mean_age_abs_events_cmt[idx_keep] = mean_age_abs_events[idx_keep]
    # save as .nii
    map_nifti = nib.Nifti1Image(mean_age_abs_events_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_abs_events_cmt.nii')


if cmt_pval_age_corr_events_iss:
    idx_keep = np.where(mean_pval_age_corr_events_iss < max(cmt_pval_age_corr_events_iss))
    mean_age_corr_events_iss_cmt = np.zeros_like(mean_age_corr_events_iss) * np.nan
    mean_age_corr_events_iss_cmt[idx_keep] = mean_age_corr_events_iss[idx_keep]
    # save as .nii
    map_nifti = nib.Nifti1Image(mean_age_corr_events_iss_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_corr_events_iss_cmt.nii')

if cmt_pval_age_rel_events_iss:
    idx_keep = np.where(mean_pval_age_rel_events_iss < max(cmt_pval_age_rel_events_iss))
    mean_age_rel_events_iss_cmt = np.zeros_like(mean_age_rel_events_iss) * np.nan
    mean_age_rel_events_iss_cmt[idx_keep] = mean_age_rel_events_iss[idx_keep]
    # save as .nii
    map_nifti = nib.Nifti1Image(mean_age_rel_events_iss_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_rel_events_iss_cmt.nii')

if cmt_pval_age_abs_events_iss:
    idx_keep = np.where(mean_pval_age_abs_events_iss < max(cmt_pval_age_abs_events_iss))
    mean_age_abs_events_iss_cmt = np.zeros_like(mean_age_abs_events_iss) * np.nan
    mean_age_abs_events_iss_cmt[idx_keep] = mean_age_abs_events_iss[idx_keep]
    # save as .nii
    map_nifti = nib.Nifti1Image(mean_age_abs_events_iss_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_abs_events_iss_cmt.nii')

print('done')