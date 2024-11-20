import numpy as np
import nibabel as nib
from scipy.stats import pearsonr
import os
from scipy import io
from tqdm import tqdm
from scipy import stats
from scipy.stats import spearmanr
import pingouin as pg
import pandas as pd
from scipy.spatial.distance import cdist

#TODO read me: these time x xtime correlations are based on whole group hyperaligned individual data

ntime = 192

# Set paths
basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
denoiseddir = basedir + 'highpass_filtered_intercept2/1groups/'
datadir = denoiseddir + 'preGSBS/age_groups/GR0/hyperaligned/'
GSBSdir = denoiseddir + 'GSBS_results/searchlights/GR0/'
ISSdir = denoiseddir + '/preGSBS/age_groups/'
savedir = denoiseddir + 'analyses_results/'
SL_dir = basedir + 'masks/searchlights/'
savedir_cor = savedir + 'time_correlations/'

# Get subjects age - vector is in the same order as the subject files in the datadir
CBU_info = io.loadmat(basedir + 'subinfo_CBU_age.mat')
var = 'subinfo_CBU_age'
CBU_age = CBU_info[var]
age = CBU_age[:, 1]

# Get searchlight info
nregs=5204
stride=2
radius=3
min_vox=15
params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)

coordinates = np.load(SL_dir + 'SL_voxelCoordinates_' + params_name + '.npy', allow_pickle=True)
searchlights = np.load(SL_dir + 'SL_voxelsIndices_' + params_name + '.npy', allow_pickle=True)

# GM mask
img = nib.load(basedir + 'masks/data_plus_GM_mask.nii')
affine = img.affine

# Get list of subjects
allfiles = os.listdir(datadir)
subjects=[]
for names in allfiles:
    if names.endswith(".nii"):
        subjects.append(names)

def load_data_subject_perSL(subject):
    subject_data_per_SL = dict((SL, []) for SL in range(len(searchlights)))

    # Load whole-brain data and convert to np
    filename = datadir + subject
    data_wholebrain = nib.load(filename).get_fdata()

    # Loop through all searchlights
    for SL, voxel_indices in enumerate(searchlights):

        # Loop through all voxels within this searchlight
        for voxel_idx in voxel_indices:
            # Get x y z indices of this voxel
            x,y,z = coordinates[voxel_idx]

            # Get timeseries of this voxel from the whole-brain data using the x y z
            data_voxel = data_wholebrain[x,y,z,:]

            # Store
            subject_data_per_SL[SL].append(data_voxel)

    # Per searchlight, transform data into pre-GSBS data
    for SL in range(len(searchlights)):
        # To numpy and Time x Voxel
        subject_data_per_SL[SL] = np.asarray(subject_data_per_SL[SL]).T

        # Z-score, as would normally happen within GSBS
        subject_data_per_SL[SL] = zscore(subject_data_per_SL[SL])

    return subject_data_per_SL

def zscore(x): # Copy from GSBS, thus x should be time by voxel
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True, ddof=1)


# If getting the strengths per subject has already been done before, skip this part as it can take 24h
if not os.path.exists(denoiseddir + 'GSBS_results/searchlights/' + 'individual_strengths_per_SL.npy'):
    # Loop over subjects to gather individual strength information
    # First loop over subjects and then loop over searchlights, as you would otherwise have to load the whole-brain nifti per subject for each searchlight again and again.
    within_cor = np.full([len(age), len(searchlights)], 0).astype(float)
    between_cor = np.full([len(age), len(searchlights)], 0).astype(float)
    dif_cor = np.full([len(age), len(searchlights)], 0).astype(float)
    strengths_individual_all = dict((SL, []) for SL in range(len(searchlights)))
    sub = 0
    for subject in tqdm(subjects):

        # Get subject-level z-scored data per subject (per SL, time x voxel)
        subject_data_per_SL = load_data_subject_perSL(subject)

        # Loop over searchlights
        for SL in np.arange(len(searchlights)):
            # Load GSBS results
            filename = GSBSdir + 'GSBS_GR0_stride2_radius3_minvox15_SL' + str(SL) + '.npy'
            GSBS_obj = np.load(filename, allow_pickle=True).item()

            # Load group-level strengths/boundaries
            strengths_group = GSBS_obj.get_strengths()
            deltas_group = GSBS_obj.get_deltas()
            boundary_idx = np.where(deltas_group)[0]
            bounds = GSBS_obj.get_bounds()

            # Setup strength timeline
            strengths_sub = np.zeros_like(strengths_group)
            strengths_sub[boundary_idx] = np.nan

            # Get searchlight data of this subject
            data_sub = subject_data_per_SL[SL]

            # Get time x time within and between state correlations
            ind = np.triu(np.ones((ntime, ntime), dtype=bool), 1)
            z = (data_sub - data_sub.mean(1, keepdims=True)) / data_sub.std(1, keepdims=True, ddof=1)
            t = np.cov(z)[ind]
            bounds = bounds > 0
            states = np.expand_dims(np.cumsum(bounds) + 1, 1)
            c_diff, same = (lambda c: (c == 1, c == 0))(cdist(states, states, "cityblock")[ind])
            within_cor[sub, SL] = np.mean(t[same])
            between_cor[sub, SL] = np.mean(t[c_diff])
            dif_cor[sub, SL] = within_cor[sub, SL] - between_cor[sub, SL]

            # Loop through boundaries
            for delta_idx in range(len(boundary_idx)):
                # Get start an end indices of current neural states
                if delta_idx == 0: # This is the first delta -> start of state 1 is 0
                    state1_start = 0
                else:
                    state1_start = boundary_idx[delta_idx-1]
                if delta_idx == len(boundary_idx) -1: # This is the last delta -> end of state 2 is end of stimulus
                    state2_end = len(strengths_sub)
                else:
                    state2_end = boundary_idx[delta_idx+1]
                middle = boundary_idx[delta_idx]

                # Get state patterns
                state1_pattern = data_sub[state1_start:middle, :]
                state2_pattern = data_sub[middle:state2_end, :]

                # Compute strength of this boundary, by computed the average pattern per state and computing pearonr, and subtracting it from 1
                r, _ = pearsonr(np.mean(state1_pattern, axis=0), np.mean(state2_pattern, axis=0))
                strength = 1 - r

                # Store the strength
                strengths_sub[middle] = strength

            # Store this subject's strength timeline
            strengths_individual_all[SL].append(strengths_sub)
        sub += 1

    # Loop over searchlights to do analysis
    for SL in range(len(searchlights)):
        strengths_individual_all[SL] = np.asarray(strengths_individual_all[SL])

    # Save individual strengths
    np.save(denoiseddir + 'GSBS_results/searchlights/' + 'individual_strengths_per_SL.npy', strengths_individual_all)
    np.save(denoiseddir + 'GSBS_results/searchlights/' + 'indiv_within_cor_per_SL.npy', within_cor)
    np.save(denoiseddir + 'GSBS_results/searchlights/' + 'indiv_between_cor_per_SL.npy', between_cor)
    np.save(denoiseddir + 'GSBS_results/searchlights/' + 'indiv_dif_cor_per_SL.npy', dif_cor)

# Load individual strengths and time*time correlations
strengths_individual_all = np.load(denoiseddir + 'GSBS_results/searchlights/' + 'individual_strengths_per_SL.npy', allow_pickle=True).item()
within_cor = np.load(denoiseddir + 'GSBS_results/searchlights/' + 'indiv_within_cor_per_SL.npy', allow_pickle=True)
between_cor = np.load(denoiseddir + 'GSBS_results/searchlights/' + 'indiv_between_cor_per_SL.npy', allow_pickle=True)
dif_cor = np.load(denoiseddir + 'GSBS_results/searchlights/' + 'indiv_dif_cor_per_SL.npy', allow_pickle=True)

# Create .nii for within_cor, between_cor and difference based on mean of all participants

x_max, y_max, z_max = img.shape
counter = np.zeros((x_max, y_max, z_max))

# Create empty arrays for main effects of timextime correlations
within_cor_sum = np.zeros((x_max, y_max, z_max))
between_cor_sum = np.zeros((x_max, y_max, z_max))
dif_cor_sum = np.zeros((x_max, y_max, z_max))

# Mean data for correlations across all participants
within_cor_meanAll = np.mean(within_cor, 0)
between_cor_meanAll = np.mean(between_cor, 0)
dif_cor_meanAll = np.mean(dif_cor, 0)

# Loop over SLs
for SL_idx, voxel_indices in enumerate(searchlights):
    within_cor_SL = within_cor_meanAll[SL_idx]
    between_cor_SL = between_cor_meanAll[SL_idx]
    dif_cor_SL = dif_cor_meanAll[SL_idx]

    for vox in voxel_indices:
        x, y, z = coordinates[vox]
        counter[x, y, z] += 1
        within_cor_sum[x, y, z] += within_cor_SL
        between_cor_sum[x, y, z] += between_cor_SL
        dif_cor_sum[x, y, z] += dif_cor_SL

# Take mean across searchlights
mean_within_cor_sum = np.divide(within_cor_sum, counter)
mean_between_cor_sum = np.divide(between_cor_sum, counter)
mean_dif_cor_sum = np.divide(dif_cor_sum, counter)

# Convert to nifti
map_nifti = nib.Nifti1Image(mean_within_cor_sum, affine)
nib.save(map_nifti, savedir_cor + 'allsubj_within_cor.nii')
map_nifti = nib.Nifti1Image(mean_between_cor_sum, affine)
nib.save(map_nifti, savedir_cor + 'allsubj_between_cor.nii')
map_nifti = nib.Nifti1Image(mean_dif_cor_sum, affine)
nib.save(map_nifti, savedir_cor + 'allsubj_dif_cor.nii')

# Correlations with age

# Load individual ISS
ISS = np.load(ISSdir + 'ISS_perSL_perSubj.npy')

# Create empty matrices
age_strength = np.full([nregs], 0).astype(float)
pval_age_strength = np.full([nregs], 0).astype(float)
age_strength_ISS = np.full([nregs], 0).astype(float)
pval_age_strength_ISS = np.full([nregs], 0).astype(float)
dif_cor_age = np.full([nregs],0).astype(float)
pval_dif_cor_age = np.full([nregs],0).astype(float)
within_cor_age = np.full([nregs],0).astype(float)
pval_within_cor_age = np.full([nregs],0).astype(float)
between_cor_age = np.full([nregs],0).astype(float)
pval_between_cor_age = np.full([nregs],0).astype(float)
age_strength_within_cor = np.full([nregs],0).astype(float)
pval_age_strength_within_cor = np.full([nregs],0).astype(float)
age_between_within_cor = np.full([nregs],0).astype(float)
pval_age_between_within_cor = np.full([nregs],0).astype(float)

# Switch subject and SL position so SL comes first
within_cor_switched = within_cor.transpose()
between_cor_switched = between_cor.transpose()
dif_cor_switched = dif_cor.transpose()

# Loop over searchlights
for SL in strengths_individual_all.keys():
    strengths_this_searchlight = strengths_individual_all[SL]

    # The strength timelines now contains a lot of zeros for timepoints without boundary at the group level. Here I am removing them
    timepoints_without_boundary = np.where(strengths_this_searchlight[0,:] == 0)[0]
    strengths_this_searchlight = np.delete(strengths_this_searchlight, timepoints_without_boundary, 1) # Subject x Boundary

    # Get the mean boundary strength for this searchlight per subject
    mean_strengths_this_searchlight = np.mean(strengths_this_searchlight, axis=1)

    # Spearman correlation age and strength and time x time correlations
    age_strength[SL], pval_age_strength[SL] = stats.spearmanr(age, mean_strengths_this_searchlight)
    dif_cor_age[SL], pval_dif_cor_age[SL] = stats.spearmanr(age, dif_cor_switched[SL, :])
    within_cor_age[SL], pval_within_cor_age[SL] = stats.spearmanr(age, within_cor_switched[SL, :])
    between_cor_age[SL], pval_between_cor_age[SL] = stats.spearmanr(age, between_cor_switched[SL, :])

    #edits Linda 29-08-2024 check partial correlation, what is explain by within/between state correlations?
    data = {'x': age, 'y': mean_strengths_this_searchlight, 'cv1': within_cor_switched[SL, :]}
    df = pd.DataFrame(data)
    spearman_age_strength_within_cor = pg.partial_corr(data=df, x='x', y='y', covar='cv1', method='spearman')
    age_strength_within_cor[SL] = spearman_age_strength_within_cor['r']
    pval_age_strength_within_cor[SL] = spearman_age_strength_within_cor['p-val']

    #edits Linda 29-08-2024 check partial correlation, what is explain by within/between state correlations?
    data = {'x': age, 'y': between_cor_switched[SL, :], 'cv1': within_cor_switched[SL, :]}
    df = pd.DataFrame(data)
    spearman_age_between_within_cor = pg.partial_corr(data=df, x='x', y='y', covar='cv1', method='spearman')
    age_between_within_cor[SL] = spearman_age_between_within_cor['r']
    pval_age_between_within_cor[SL] = spearman_age_between_within_cor['p-val']

    # Calculate partial correlation with ISS as covariate
    data = {'x': age, 'y': mean_strengths_this_searchlight, 'cv1': ISS[:, SL]}
    df = pd.DataFrame(data)
    spearman_age_strength_ISS = pg.partial_corr(data=df, x='x', y='y', covar='cv1', method='spearman')
    age_strength_ISS[SL] = spearman_age_strength_ISS['r']
    pval_age_strength_ISS[SL] = spearman_age_strength_ISS['p-val']

# save age_strength to calculate correlation with age_dur
np.save(os.path.join(savedir, "age_strength_vector.npy"), age_strength)

# correct for multiple testing
# based on reject FALSE/TRUE you get the indices for the significant elements and extract the original corresponding p-values
reject, pvals_cmt = pg.multicomp(pval_age_strength, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_strength = [pval_age_strength[index] for index in true_indices]

reject, pvals_ISS_cmt = pg.multicomp(pval_age_strength_ISS, method='fdr_bh')
true_indices_ISS = [index for index, value in enumerate(reject) if value]
cmt_pval_age_strength_ISS = [pval_age_strength_ISS[index] for index in true_indices_ISS]

reject, pvals_cmt = pg.multicomp(pval_dif_cor_age, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_dif_cor_age = [pval_dif_cor_age[index] for index in true_indices]

reject, pvals_cmt = pg.multicomp(pval_age_strength_within_cor, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_age_strength_within_cor = [pval_age_strength_within_cor[index] for index in true_indices]

reject, pvals_cmt = pg.multicomp(pval_within_cor_age, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_within_cor_age = [pval_within_cor_age[index] for index in true_indices]

reject, pvals_cmt = pg.multicomp(pval_between_cor_age, method='fdr_bh')
true_indices = [index for index, value in enumerate(reject) if value]
cmt_pval_between_cor_age = [pval_between_cor_age[index] for index in true_indices]

# from SL to voxel
counter = np.zeros((x_max, y_max, z_max))

age_strength_sum = np.zeros((x_max, y_max, z_max))
pval_age_strength_sum = np.zeros((x_max, y_max, z_max))

age_strength_ISS_sum = np.zeros((x_max, y_max, z_max))
pval_age_strength_ISS_sum = np.zeros((x_max, y_max, z_max))

age_strength_within_cor_sum = np.zeros((x_max, y_max, z_max))
pval_age_strength_within_cor_sum = np.zeros((x_max, y_max, z_max))

dif_cor_age_sum = np.zeros((x_max, y_max, z_max))
pval_dif_cor_age_sum = np.zeros((x_max, y_max, z_max))
within_cor_age_sum = np.zeros((x_max, y_max, z_max))
pval_within_cor_age_sum = np.zeros((x_max, y_max, z_max))
between_cor_age_sum = np.zeros((x_max, y_max, z_max))
pval_between_cor_age_sum = np.zeros((x_max, y_max, z_max))

for SL, voxel_indices in enumerate(tqdm(searchlights)):
    for vox in voxel_indices:
        x, y, z = coordinates[vox]
        counter[x, y, z] += 1
        age_strength_sum[x, y, z] += age_strength[SL]
        pval_age_strength_sum[x,y,z] += pval_age_strength[SL]
        age_strength_ISS_sum[x, y, z] += age_strength_ISS[SL]
        pval_age_strength_ISS_sum[x,y,z] += pval_age_strength_ISS[SL]
        dif_cor_age_sum[x, y, z] += dif_cor_age[SL]
        pval_dif_cor_age_sum[x, y, z] += pval_dif_cor_age[SL]
        within_cor_age_sum[x, y, z] += within_cor_age[SL]
        pval_within_cor_age_sum[x, y, z] += pval_within_cor_age[SL]
        between_cor_age_sum[x, y, z] += between_cor_age[SL]
        pval_between_cor_age_sum[x, y, z] += pval_between_cor_age[SL]
        age_strength_within_cor_sum[x, y, z] += age_strength_within_cor[SL]
        pval_age_strength_within_cor_sum[x,y,z] += pval_age_strength_within_cor[SL]

mean_age_strength = np.divide(age_strength_sum, counter)
mean_pval_age_strength = np.divide(pval_age_strength_sum, counter)
mean_age_strength_ISS = np.divide(age_strength_ISS_sum, counter)
mean_pval_age_strength_ISS = np.divide(pval_age_strength_ISS_sum, counter)
mean_age_strength_within_cor = np.divide(age_strength_within_cor_sum, counter)
mean_pval_age_strength_within_cor = np.divide(pval_age_strength_within_cor_sum, counter)

mean_dif_cor_age_sum = np.divide(dif_cor_age_sum, counter)
mean_pval_dif_cor_age_sum = np.divide(pval_dif_cor_age_sum, counter)
mean_within_cor_age_sum = np.divide(within_cor_age_sum, counter)
mean_pval_within_cor_age_sum = np.divide(pval_within_cor_age_sum, counter)
mean_between_cor_age_sum = np.divide(between_cor_age_sum, counter)
mean_pval_between_cor_age_sum = np.divide(pval_between_cor_age_sum, counter)

# Save .nii

# Unthresholded
map_nifti = nib.Nifti1Image(mean_age_strength, affine)
nib.save(map_nifti, savedir + 'correlation_age_strength.nii')
map_nifti = nib.Nifti1Image(mean_age_strength_ISS, affine)
nib.save(map_nifti, savedir + 'correlation_age_strength_ISS.nii')
map_nifti = nib.Nifti1Image(mean_age_strength_within_cor, affine)
nib.save(map_nifti, savedir + 'correlation_age_strength_within_cor.nii')

map_nifti = nib.Nifti1Image(mean_dif_cor_age_sum, affine)
nib.save(map_nifti, savedir_cor + 'dif_cor_agecont.nii')
map_nifti = nib.Nifti1Image(mean_within_cor_age_sum, affine)
nib.save(map_nifti, savedir_cor + 'within_cor_agecont.nii')
map_nifti = nib.Nifti1Image(mean_between_cor_age_sum, affine)
nib.save(map_nifti, savedir_cor + 'between_cor_agecont.nii')

# Create map of <0.05 - uncorrected for multiple testing
idx_keep = np.where(mean_pval_age_strength < 0.05)
mean_age_strength_pthresh = np.zeros_like(mean_age_strength) * np.nan
mean_age_strength_pthresh[idx_keep] = mean_age_strength[idx_keep]
map_nifti = nib.Nifti1Image(mean_age_strength_pthresh, affine)
nib.save(map_nifti, savedir + 'correlation_age_strength_pthresh.nii')

idx_keep = np.where(mean_pval_age_strength_ISS < 0.05)
mean_age_strength_ISS_pthresh = np.zeros_like(mean_age_strength_ISS) * np.nan
mean_age_strength_ISS_pthresh[idx_keep] = mean_age_strength_ISS[idx_keep]
map_nifti = nib.Nifti1Image(mean_age_strength_ISS_pthresh, affine)
nib.save(map_nifti, savedir + 'correlation_age_strength_ISS_pthresh.nii')

# Create map of <0.05 - corrected for multiple testing
if cmt_pval_age_strength:
    idx_keep = np.where(mean_pval_age_strength < max(cmt_pval_age_strength))
    mean_age_strength_cmt = np.zeros_like(mean_age_strength) * np.nan
    mean_age_strength_cmt[idx_keep] = mean_age_strength[idx_keep]
    map_nifti = nib.Nifti1Image(mean_age_strength_cmt, affine)
    nib.save(map_nifti, savedir + 'correlation_age_strength_cmt.nii')

if cmt_pval_age_strength_ISS:
    idx_keep = np.where(mean_pval_age_strength_ISS < max(cmt_pval_age_strength_ISS))
    mean_age_strength_ISS_cmt = np.zeros_like(mean_age_strength_ISS) * np.nan
    mean_age_strength_ISS_cmt[idx_keep] = mean_age_strength_ISS[idx_keep]
    map_nifti = nib.Nifti1Image(mean_age_strength_ISS_cmt, affine)
    nib.save(map_nifti, savedir + 'correlation_age_strength_ISS_cmt.nii')

if cmt_pval_age_strength_within_cor:
    idx_keep = np.where(mean_pval_age_strength_within_cor < max(cmt_pval_age_strength_within_cor))
    mean_age_strength_within_cor_cmt = np.zeros_like(mean_age_strength_within_cor) * np.nan
    mean_age_strength_within_cor_cmt[idx_keep] = mean_age_strength_within_cor[idx_keep]
    map_nifti = nib.Nifti1Image(mean_age_strength_within_cor_cmt, affine)
    nib.save(map_nifti, savedir + 'correlation_age_strength_within_cor_cmt.nii')

if cmt_pval_dif_cor_age:
    idx_keep = np.where(mean_pval_dif_cor_age_sum < max(cmt_pval_dif_cor_age))
    mean_dif_cor_age_cmt = np.zeros_like(mean_dif_cor_age_sum) * np.nan
    mean_dif_cor_age_cmt[idx_keep] = mean_dif_cor_age_sum[idx_keep]
    map_nifti = nib.Nifti1Image(mean_dif_cor_age_cmt, affine)
    nib.save(map_nifti, savedir_cor + 'dif_cor_agecont_cmt.nii')

if cmt_pval_within_cor_age:
    idx_keep = np.where(mean_pval_within_cor_age_sum < max(cmt_pval_within_cor_age))
    mean_within_cor_age_cmt = np.zeros_like(mean_within_cor_age_sum) * np.nan
    mean_within_cor_age_cmt[idx_keep] = mean_within_cor_age_sum[idx_keep]
    map_nifti = nib.Nifti1Image(mean_within_cor_age_cmt, affine)
    nib.save(map_nifti, savedir_cor + 'within_cor_agecont_cmt.nii')

if cmt_pval_between_cor_age:
    idx_keep = np.where(mean_pval_between_cor_age_sum < max(cmt_pval_between_cor_age))
    mean_between_cor_age_cmt = np.zeros_like(mean_between_cor_age_sum) * np.nan
    mean_between_cor_age_cmt[idx_keep] = mean_between_cor_age_sum[idx_keep]
    map_nifti = nib.Nifti1Image(mean_between_cor_age_cmt, affine)
    nib.save(map_nifti, savedir_cor + 'between_cor_agecont_cmt.nii')
