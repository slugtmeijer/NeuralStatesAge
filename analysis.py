from scipy import io
from scipy.io import loadmat
from scipy import stats
import sys
sys.path.append("..")
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
import pingouin as pg
from create_folder import create_folder

groups = 34 #TODO change for 1 or 34 groups - for 1 group don't run age analyses

basedir = '/home/lingee/wrkgrp/Selma/CamCAN_movie/'
ngroups_dir = basedir + 'highpass_filtered_intercept2/' + str(groups) + 'groups/'
datadir = ngroups_dir + 'GSBS_results/searchlights/'
savedir = ngroups_dir + 'analyses_results/'
create_folder(savedir)
save_maineffects = 'figsmaineffects/'
SL_dir = basedir + 'masks/searchlights/'
ISS_dir = ngroups_dir + 'preGSBS/age_groups/'

# collect results
ntime=192
nregs=5204
kfold_data=groups
maxk=96

# TODO till line 84 only run first time if you still need to extract all data from the GSBS files - this step takes long
# nstates = np.full([nregs,kfold_data], 0).astype(int)
# tdists = np.full([nregs,kfold_data, maxk + 2], 0).astype(float)
# bounds = np.full([nregs,kfold_data, ntime],0).astype(int)
# strengths = np.full([nregs,kfold_data, ntime],0).astype(float)
# deltas = np.full([nregs,kfold_data, ntime],0).astype(int)
# median_duration = np.full([nregs,kfold_data], 0).astype(float)
# variability_duration =np.full([nregs,kfold_data], 0).astype(float)
#
# # load data
# for k in range(kfold_data):
#     print(k)
#     for r in range(nregs): #SL
#         print(r)
#         filename = (datadir + 'GR' + str(k) + '/' + 'GSBS_GR' + str(k) + '_stride2_radius3_minvox15_SL' + str(r) +
#                     '.npy')
#         GSBS_obj = np.load(filename, allow_pickle=True).item()
#
#         nstates[r,k] = GSBS_obj.nstates
#         tdists[r,k,:] = GSBS_obj.tdists
#         bounds[r, k, :] = GSBS_obj.bounds
#         strengths[r, k,:] = GSBS_obj.strengths
#         deltas[r,k,:] = GSBS_obj.deltas
#
#         durations = [] # in TR
#         count = 0
#         for d in deltas[r,k,:]:
#              if d:
#                  durations.append(count)
#                  count = 1
#              else:
#                  count += 1
#         median_duration[r,k] = np.median(durations)
#         q1= np.percentile(durations, 25)
#         q3 = np.percentile(durations, 75)
#         iqr_duration = q3 - q1
#         variability_duration[r,k] = iqr_duration/median_duration[r,k]
#
# io.savemat(savedir + 'GSBS_obj.mat',
#            {'nstates': nstates, 'tdists': tdists, 'bounds': bounds, 'strengths': strengths, 'deltas': deltas, 'median_duration': median_duration, 'variability_duration': variability_duration})
#
# # Creating a dictionary containing all variables
# data_to_save = {
#     'nstates': nstates,
#     'tdists': tdists,
#     'bounds': bounds,
#     'strengths': strengths,
#     'deltas': deltas,
#     'median_duration': median_duration,
#     'variability_duration': variability_duration
# }
# # Saving all variables in a single .npy file
# np.savez(savedir + 'key_GSBS_output.npz', **data_to_save)

# TODO next 5 lines only if you only run the second half - load variables
data = loadmat(savedir + 'GSBS_obj.mat')
median_duration = data['median_duration']
variability_duration = data['variability_duration']
bounds = data['bounds']
binbounds=bounds>0

# compare youngest group with 11 older groups on median duration for 4 SLs used in simulations
SL_ind = [692, 2463, 1874, 2466]
n_binbounds = np.sum(binbounds, axis=2)
values_for_SLindices = n_binbounds[SL_ind, :]
values_for_SLindices_gr0 = values_for_SLindices[:,0]
values_for_SLindices_grOlder = values_for_SLindices[:,-11:]
means_old = np.mean(values_for_SLindices_grOlder, axis=1)
mins_old = np.min(values_for_SLindices_grOlder, axis=1)
maxs_old = np.max(values_for_SLindices_grOlder, axis=1)

# edit Linda - 04/09/24 check group similarity
cs = np.corrcoef(np.transpose(median_duration))
ind=np.where(np.triu(np.ones(cs.shape)))
np.mean(cs[ind])

stride=2
radius=3
min_vox=15
params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)

img = nib.load(basedir + 'masks/data_plus_GM_mask.nii')
affine = img.affine

# Get searchlight information
coordinates = np.load(SL_dir + 'SL_voxelCoordinates_' + params_name + '.npy', allow_pickle=True)
searchlights = np.load(SL_dir + 'SL_voxelsIndices_' + params_name + '.npy', allow_pickle=True)

# Correlation between median number of states (+iqr) and group with ISS as covariate
ISS = np.load(ISS_dir + 'all_groups_mean_correlation_per_SL.npy')
group = np.arange(groups)

# for median duration
age_dur = np.full([nregs], 0).astype(float)
pval_age_dur = np.full([nregs], 0).astype(float)
age_dur_iss = np.full([nregs], 0).astype(float)
pval_age_dur_iss = np.full([nregs], 0).astype(float)

# for variability in duration
age_vardur = np.full([nregs], 0).astype(float)
pval_age_vardur = np.full([nregs], 0).astype(float)
age_vardur_iss = np.full([nregs], 0).astype(float)
pval_age_vardur_iss = np.full([nregs], 0).astype(float)

# for correlation between age and ISS
age_iss = np.full([nregs], 0).astype(float)
pval_age_iss = np.full([nregs], 0).astype(float)

for SL in range(nregs):
     # Calculate the Spearman correlation between the ordinal (group) and continuous variable (median duration)
     age_dur[SL], pval_age_dur[SL] = stats.spearmanr(group, median_duration[SL,:])
     # Calculate the Spearman correlation between the ordinal (group) and continuous variable (ISS)
     age_iss[SL], pval_age_iss[SL] = stats.spearmanr(group, ISS[SL,:])
     # Calculate partial correlation with ISS as covariate
     data = {'x':group, 'y':median_duration[SL,:], 'cv1':ISS[SL,:]}
     df = pd.DataFrame(data)
     spearman_age_dur_iss = pg.partial_corr(data=df, x='x', y='y', covar='cv1', method='spearman')
     age_dur_iss[SL] = spearman_age_dur_iss['r']
     pval_age_dur_iss[SL] = spearman_age_dur_iss['p-val']
     # Calculate the Spearman correlation between the ordinal (group) and continuous variable (variability in median duration)
     age_vardur[SL], pval_age_vardur[SL] = stats.spearmanr(group, variability_duration[SL,:])
     # Calculate partial correlation with ISS as covariate
     data_vardur = {'x':group, 'y':variability_duration[SL,:], 'cv1':ISS[SL,:]}
     df_vardur = pd.DataFrame(data_vardur)
     spearman_age_vardur_iss = pg.partial_corr(data=df_vardur, x='x', y='y', covar='cv1', method='spearman')
     age_vardur_iss[SL] = spearman_age_vardur_iss['r']
     pval_age_vardur_iss[SL] = spearman_age_vardur_iss['p-val']

# save age_dur to calculate correlation with overlap event and neural state boundaries
np.save(os.path.join(savedir, "age_dur_vector.npy"), age_dur)
np.save(os.path.join(savedir, "age_iss_vector.npy"), age_iss)

# correct for multiple testing median duration
# based on reject FALSE/TRUE you get the indices for the significant elements and extract the original corresponding p-values
reject, pvals_cmt = pg.multicomp(pval_age_dur, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_dur = [pval_age_dur[index] for index in true_indices]

reject, pvals_cmt = pg.multicomp(pval_age_dur_iss, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_dur_iss = [pval_age_dur_iss[index] for index in true_indices]

# correct for multiple testing age - iss
reject, pvals_cmt = pg.multicomp(pval_age_iss, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_iss = [pval_age_iss[index] for index in true_indices]

# correct for multiple testing variability duration
reject, pvals_cmt = pg.multicomp(pval_age_vardur, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_vardur = [pval_age_vardur[index] for index in true_indices]

reject, pvals_cmt = pg.multicomp(pval_age_vardur_iss, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_vardur_iss = [pval_age_vardur_iss[index] for index in true_indices]

print("Start plotting")

# from SL to voxel
x_max, y_max, z_max = img.shape
counter = np.zeros((x_max, y_max, z_max))

dur_sum = np.zeros((x_max, y_max, z_max))
vardur_sum = np.zeros((x_max, y_max, z_max))
age_dur_sum = np.zeros((x_max, y_max, z_max))
pval_age_dur_sum = np.zeros((x_max, y_max, z_max))
age_iss_sum = np.zeros((x_max, y_max, z_max))
pval_age_iss_sum = np.zeros((x_max, y_max, z_max))
age_dur_iss_sum = np.zeros((x_max, y_max, z_max))
pval_age_dur_iss_sum = np.zeros((x_max, y_max, z_max))
age_vardur_sum = np.zeros((x_max, y_max, z_max))
pval_age_vardur_sum = np.zeros((x_max, y_max, z_max))
age_vardur_iss_sum = np.zeros((x_max, y_max, z_max))
pval_age_vardur_iss_sum = np.zeros((x_max, y_max, z_max))

# create .nii for median duration and variability per group
for k in range(kfold_data):
    for SL_idx, voxel_indices in enumerate(tqdm(searchlights)):
        median_dur = median_duration[SL_idx][k]
        var_dur = variability_duration[SL_idx][k]

        for vox in voxel_indices:
            x, y, z = coordinates[vox]
            counter[x, y, z] += 1
            dur_sum[x, y, z] += median_dur
            vardur_sum[x, y, z] += var_dur

    # Take mean across searchlights
    mean_durations = np.divide(dur_sum, counter)
    mean_variability = np.divide(vardur_sum, counter)

    # Convert to nifti
    map_nifti = nib.Nifti1Image(mean_durations, affine)
    nib.save(map_nifti, savedir + 'analysis_GR' + str(k) + '_durations.nii')

    map_nifti = nib.Nifti1Image(mean_variability, affine)
    nib.save(map_nifti, savedir + 'analysis_GR' + str(k) + '_variability.nii')

# go from SL to voxels for median duration and variability and iss
counter = np.zeros((x_max, y_max, z_max))
for SL_idx, voxel_indices in enumerate(tqdm(searchlights)):
    for vox in voxel_indices:
        x, y, z = coordinates[vox]
        counter[x, y, z] += 1
        age_dur_sum[x, y, z] += age_dur[SL_idx]
        pval_age_dur_sum[x,y,z] += pval_age_dur[SL_idx]
        age_iss_sum[x, y, z] += age_iss[SL_idx]
        pval_age_iss_sum[x,y,z] += pval_age_iss[SL_idx]
        age_dur_iss_sum[x, y, z] += age_dur_iss[SL_idx]
        pval_age_dur_iss_sum[x,y,z] += pval_age_dur_iss[SL_idx]
        age_vardur_sum[x, y, z] += age_vardur[SL_idx]
        pval_age_vardur_sum[x,y,z] += pval_age_vardur[SL_idx]
        age_vardur_iss_sum[x, y, z] += age_vardur_iss[SL_idx]
        pval_age_vardur_iss_sum[x,y,z] += pval_age_vardur_iss[SL_idx]

# Take mean across searchlights for median duration - age - (ISS)
mean_age_dur = np.divide(age_dur_sum, counter)
mean_pval_age_dur = np.divide(pval_age_dur_sum, counter)
mean_age_dur_iss = np.divide(age_dur_iss_sum, counter)
mean_pval_age_dur_iss = np.divide(pval_age_dur_iss_sum, counter)
# Take mean across searchlights for iss - age
mean_age_iss = np.divide(age_iss_sum, counter)
mean_pval_age_iss = np.divide(pval_age_iss_sum, counter)
# Take mean across searchlights for variability in duration - age (ISS)
mean_age_vardur = np.divide(age_vardur_sum, counter)
mean_pval_age_vardur = np.divide(pval_age_vardur_sum, counter)
mean_age_vardur_iss = np.divide(age_vardur_iss_sum, counter)
mean_pval_age_vardur_iss = np.divide(pval_age_vardur_iss_sum, counter)

# Convert to nifti unthresholded
map_nifti = nib.Nifti1Image(mean_age_dur, affine)
nib.save(map_nifti, savedir + 'analysis_age_durations.nii')
map_nifti = nib.Nifti1Image(mean_age_dur_iss, affine)
nib.save(map_nifti, savedir + 'analysis_age_durations_iss.nii')
map_nifti = nib.Nifti1Image(mean_age_vardur, affine)
nib.save(map_nifti, savedir + 'analysis_age_variability.nii')
map_nifti = nib.Nifti1Image(mean_age_vardur_iss, affine)
nib.save(map_nifti, savedir + 'analysis_age_variability_iss.nii')
map_nifti = nib.Nifti1Image(mean_age_iss, affine)
nib.save(map_nifti, savedir + 'analysis_age_iss.nii')

# Create map of <0.05 - uncorrected for multiple testing
idx_keep = np.where(mean_pval_age_dur < 0.05)
mean_age_dur_pthresh = np.zeros_like(mean_age_dur) * np.nan
mean_age_dur_pthresh[idx_keep] = mean_age_dur[idx_keep]
map_nifti = nib.Nifti1Image(mean_age_dur_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_durations_pthresh.nii')

idx_keep = np.where(mean_pval_age_dur_iss < 0.05)
mean_age_dur_iss_pthresh= np.zeros_like(mean_age_dur_iss) * np.nan
mean_age_dur_iss_pthresh[idx_keep] = mean_age_dur_iss[idx_keep]
map_nifti = nib.Nifti1Image(mean_age_dur_iss_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_durations_iss_pthresh.nii')

idx_keep = np.where(mean_pval_age_vardur < 0.05)
mean_age_vardur_pthresh = np.zeros_like(mean_age_vardur) * np.nan
mean_age_vardur_pthresh[idx_keep] = mean_age_vardur[idx_keep]
map_nifti = nib.Nifti1Image(mean_age_vardur_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_variability_pthresh.nii')

idx_keep = np.where(mean_pval_age_vardur_iss < 0.05)
mean_age_vardur_iss_pthresh = np.zeros_like(mean_age_vardur_iss) * np.nan
mean_age_vardur_iss_pthresh[idx_keep] = mean_age_vardur_iss[idx_keep]
map_nifti = nib.Nifti1Image(mean_age_vardur_iss_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_variability_iss_pthresh.nii')

idx_keep = np.where(mean_pval_age_iss < 0.05)
mean_age_iss_pthresh = np.zeros_like(mean_age_iss) * np.nan
mean_age_iss_pthresh[idx_keep] = mean_age_iss[idx_keep]
map_nifti = nib.Nifti1Image(mean_age_iss_pthresh, affine)
nib.save(map_nifti, savedir + 'analysis_age_iss_pthresh.nii')

# Create map of <0.05 - corrected for multiple testing at SL level for median duration (without and with ISS) and variability
if cmt_pval_age_dur:
    idx_keep = np.where(mean_pval_age_dur < max(cmt_pval_age_dur))
    mean_age_dur_cmt = np.zeros_like(mean_age_dur) * np.nan
    mean_age_dur_cmt[idx_keep] = mean_age_dur[idx_keep]
    #map_nifti = nib.Nifti1Image(mean_age_dur_cmt, affine)
    #nib.save(map_nifti, savedir + 'analysis_age_durations_cmt.nii')
if cmt_pval_age_dur_iss:
    idx_keep = np.where(mean_pval_age_dur_iss < max(cmt_pval_age_dur_iss))
    mean_age_dur_iss_cmt = np.zeros_like(mean_age_dur_iss) * np.nan
    mean_age_dur_iss_cmt[idx_keep] = mean_age_dur_iss[idx_keep]
    map_nifti = nib.Nifti1Image(mean_age_dur_iss_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_durations_iss_cmt.nii')
if cmt_pval_age_iss:
    idx_keep = np.where(mean_pval_age_iss < max(cmt_pval_age_iss))
    mean_age_iss_cmt = np.zeros_like(mean_age_iss) * np.nan
    mean_age_iss_cmt[idx_keep] = mean_age_iss[idx_keep]
    map_nifti = nib.Nifti1Image(mean_age_iss_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_iss_cmt.nii')
if cmt_pval_age_vardur:
    idx_keep = np.where(mean_pval_age_vardur < max(cmt_pval_age_vardur))
    mean_age_vardur_cmt = np.zeros_like(mean_age_vardur) * np.nan
    mean_age_vardur_cmt[idx_keep] = mean_age_vardur[idx_keep]
    map_nifti = nib.Nifti1Image(mean_age_vardur_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_variability_cmt.nii')
if cmt_pval_age_vardur_iss:
    idx_keep = np.where(mean_pval_age_vardur_iss < max(cmt_pval_age_vardur_iss))
    mean_age_vardur_iss_cmt = np.zeros_like(mean_age_vardur_iss) * np.nan
    mean_age_vardur_iss_cmt[idx_keep] = mean_age_vardur_iss[idx_keep]
    map_nifti = nib.Nifti1Image(mean_age_vardur_iss_cmt, affine)
    nib.save(map_nifti, savedir + 'analysis_age_variability_iss_cmt.nii')

print("correlations with age done")

#Get figures for main effects of duration, variability
#get mean of median_durations across groups per searchlight
mean_medianduration = np.mean(np.asarray(median_duration), axis=1)
mean_variabilitymedianduration = np.mean(np.asarray(variability_duration), axis=1)

# For median_duration for young - middle - old
mean_medianduration_groupy = np.mean(np.asarray(median_duration)[:, :11], axis=1)
mean_medianduration_groupm = np.mean(np.asarray(median_duration)[:, 11:23], axis=1)
mean_medianduration_groupo = np.mean(np.asarray(median_duration)[:, 23:], axis=1)

# For variability_duration for young - middle - old
mean_variabilitymedianduration_groupy = np.mean(np.asarray(variability_duration)[:, :11], axis=1)
mean_variabilitymedianduration_groupm = np.mean(np.asarray(variability_duration)[:, 11:23], axis=1)
mean_variabilitymedianduration_groupo = np.mean(np.asarray(variability_duration)[:, 23:], axis=1)

x_max, y_max, z_max = img.shape
counter = np.zeros((x_max, y_max, z_max))

sum_dur = np.zeros((x_max, y_max, z_max))
sum_vardur = np.zeros((x_max, y_max, z_max))
sum_dur_gry = np.zeros((x_max, y_max, z_max))
sum_vardur_gry = np.zeros((x_max, y_max, z_max))
sum_dur_grm = np.zeros((x_max, y_max, z_max))
sum_vardur_grm = np.zeros((x_max, y_max, z_max))
sum_dur_gro = np.zeros((x_max, y_max, z_max))
sum_vardur_gro = np.zeros((x_max, y_max, z_max))

for SL_idx, voxel_indices in enumerate(tqdm(searchlights)):
    dur = mean_medianduration[SL_idx]
    vardur = mean_variabilitymedianduration[SL_idx]
    dur_gry = mean_medianduration_groupy[SL_idx]
    vardur_gry = mean_variabilitymedianduration_groupy[SL_idx]
    dur_grm = mean_medianduration_groupm[SL_idx]
    vardur_grm = mean_variabilitymedianduration_groupm[SL_idx]
    dur_gro = mean_medianduration_groupo[SL_idx]
    vardur_gro = mean_variabilitymedianduration_groupo[SL_idx]

    for vox in voxel_indices:
        x, y, z = coordinates[vox]
        counter[x, y, z] += 1
        sum_dur[x, y, z] += dur
        sum_vardur[x, y, z] += vardur
        sum_dur_gry[x, y, z] += dur_gry
        sum_vardur_gry[x, y, z] += vardur_gry
        sum_dur_grm[x, y, z] += dur_grm
        sum_vardur_grm[x, y, z] += vardur_grm
        sum_dur_gro[x, y, z] += dur_gro
        sum_vardur_gro[x, y, z] += vardur_gro

# Take mean across searchlights
mean_sum_dur = np.divide(sum_dur, counter)
mean_sum_vardur = np.divide(sum_vardur, counter)
mean_sum_dur_gry = np.divide(sum_dur_gry, counter)
mean_sum_vardur_gry = np.divide(sum_vardur_gry, counter)
mean_sum_dur_grm = np.divide(sum_dur_grm, counter)
mean_sum_vardur_grm = np.divide(sum_vardur_grm, counter)
mean_sum_dur_gro = np.divide(sum_dur_gro, counter)
mean_sum_vardur_gro = np.divide(sum_vardur_gro, counter)

# Convert to nifti
map_nifti = nib.Nifti1Image(mean_sum_dur, affine)
nib.save(map_nifti, savedir + 'mediandurations.nii')
map_nifti = nib.Nifti1Image(mean_sum_vardur, affine)
nib.save(map_nifti, savedir + 'variability_mediandurations.nii')
map_nifti = nib.Nifti1Image(mean_sum_dur_gry, affine)
nib.save(map_nifti, savedir + 'mediandurations_gry.nii')
map_nifti = nib.Nifti1Image(mean_sum_vardur_gry, affine)
nib.save(map_nifti, savedir + 'variability_mediandurations_gry.nii')
map_nifti = nib.Nifti1Image(mean_sum_dur_grm, affine)
nib.save(map_nifti, savedir + 'mediandurations_grm.nii')
map_nifti = nib.Nifti1Image(mean_sum_vardur_grm, affine)
nib.save(map_nifti, savedir + 'variability_mediandurations_grm.nii')
map_nifti = nib.Nifti1Image(mean_sum_dur_gro, affine)
nib.save(map_nifti, savedir + 'mediandurations_gro.nii')
map_nifti = nib.Nifti1Image(mean_sum_vardur_gro, affine)
nib.save(map_nifti, savedir + 'variability_mediandurations_gro.nii')