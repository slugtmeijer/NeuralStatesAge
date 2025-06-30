import numpy as np
import nibabel as nib
import os
from scipy.io import loadmat
from scipy.stats import ranksums
from scipy import stats
from scipy import io
from statesegmentation import GSBS
from joblib import Parallel, delayed
from compute_overlap_1swindow import compute_overlap_1swindow
import random
import pandas as pd
import pingouin as pg

#these time x time correlations are based on whole group hyperaligned individual data

ntime = 192
nr_parallel_jobs = 30


# Set paths
basedir = '/home/lingee/wrkgrp/Selma/CamCAN_movie/'
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
nsub=len(age)


sub_groupID = np.full([nsub], 0).astype(float)
subgroup = loadmat('/home/lingee/wrkgrp/Selma/ids_34x577.mat')
for sub in np.arange(0,nsub):
    ID = CBU_info['subinfo_CBU_age'][sub,0]
    for gr in np.arange(0,34):
        if ID in subgroup['groupIDs'][0][gr]:
            sub_groupID[sub]=gr

# Get searchlight info
nregs=5204
stride=2
radius=3
min_vox=15
params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)

coordinates = np.load(SL_dir + 'SL_voxelCoordinates_' + params_name + '.npy', allow_pickle=True)
searchlights = np.load(SL_dir + 'SL_voxelsIndices_' + params_name + '.npy', allow_pickle=True)

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


def ss_GSBS(SL:int,subjects:list, GSBSdir):
    # Get subject-level z-scored data per subject (per SL, time x voxel)

    savename_strengths = denoiseddir + 'GSBS_results/searchlights/ind_GSBS/strenghts_SL' + str(SL) + '.npy'
    savename_tdists = denoiseddir + 'GSBS_results/searchlights/ind_GSBS/tdists_SL' + str(SL) + '.npy'

    filename = GSBSdir + 'GSBS_GR0_stride2_radius3_minvox15_SL' + str(SL) + '.npy'
    GSBS_obj = np.load(filename, allow_pickle=True).item()
    nbounds = GSBS_obj.nstates

    strengths_individual_all = np.zeros((len(subjects), 192))
    #tdists_individual_all = np.zeros((len(subjects), nbounds+5))

    sloop=np.arange(len(subjects))
    random.shuffle(sloop)
    # Loop over subjects
    for sub in sloop:
        # Load GSBS results
        subject_data_per_SL = load_data_subject_perSL(subjects[sub])
        # Get searchlight data of this subject
        data_sub = subject_data_per_SL[SL]

        GSBS_obj = GSBS(x=data_sub, kmax=nbounds + 1, finetune=1, statewise_detection=True)
        GSBS_obj.fit(False)
        strengths = GSBS_obj.get_strengths(nbounds)
        #tdists = GSBS_obj._tdists

        # Store this subject's strength timeline
        strengths_individual_all[sub]=strengths
        #tdists_individual_all[sub,0:tdists.shape[0]]=tdists

    np.save(savename_strengths,strengths_individual_all)
    #np.save(savename_tdists, tdists_individual_all)

# This is where you actually run the single subject GSBS
#SLs=np.array([1874,2466]) #visual and vmPFC SLs
#SLs=np.array([692,2463]) #superior temporal gyrus and superior frontal gyrus SLs
SLs=np.array([692,1874,2463,2466])
SL_names = ['STG', 'visual', 'SFG', 'vmPFC']
#Parallel(n_jobs=3)(delayed(ss_GSBS)(SL=SL, subjects=subjects, GSBSdir = GSBSdir) for SL in SLs)

# create empty arrays for overlap
rel_overlap_events_sub = np.full([len(SLs), nsub], 0).astype(float)
abs_overlap_events_sub = np.full([len(SLs), nsub], 0).astype(float)
boundISS = np.full([len(SLs), nsub], 0).astype(float)
group_boundISS = np.full([len(SLs), 34], 0).astype(float)
age_corr_abs_overlap = np.full([len(SLs)], 0).astype(float)
age_corr_rel_overlap = np.full([len(SLs)], 0).astype(float)
pval_age_corr_abs_overlap = np.full([len(SLs)], 0).astype(float)
pval_age_corr_rel_overlap = np.full([len(SLs)], 0).astype(float)

for indSL, SL in enumerate(SLs):
    savename_strengths = denoiseddir + 'GSBS_results/searchlights/ind_GSBS/strenghts_SL' + str(SL) + '.npy'
    strengths_individual_all=np.load(savename_strengths)
    binbounds = np.double(strengths_individual_all>0)

    # first TR doesn't count because can't be a boundary
    time_range = slice(1, None)
    # correlation / overlap over time per group per searchlight

    data = binbounds[:, time_range]
    # overlap between neural boundaries and subj event boundaries
    # rel_overlap = compute_overlap(1, group_data, event_boundaries)
    rel_overlap = compute_overlap_1swindow(1, data, event_boundaries, event_boundaries_p1s,
                                           event_boundaries_m1s)
    rel_overlap_events_sub[indSL, :] = rel_overlap
    # abs_overlap = compute_overlap(2, group_data, event_boundaries)
    abs_overlap = compute_overlap_1swindow(2, data, event_boundaries, event_boundaries_p1s,
                                           event_boundaries_m1s)
    abs_overlap_events_sub[indSL, :] = abs_overlap

    #correlation with age
    age_corr_abs_overlap[indSL], pval_age_corr_abs_overlap[indSL] = stats.spearmanr(age, abs_overlap_events_sub[indSL, :])
    age_corr_rel_overlap[indSL], pval_age_corr_rel_overlap[indSL] = stats.spearmanr(age, rel_overlap_events_sub[indSL, :])

    #boundary ISS
    for sub in np.arange(0,nsub):
        group = sub_groupID[sub]
        avgbounds = np.sum(data[sub_groupID==group,:],0)
        subbounds =  data[sub,:]
        avgbounds = (avgbounds - subbounds) / (np.sum(sub_groupID==group)-1)
        boundISS[indSL,sub]=np.corrcoef(avgbounds, subbounds)[0,1]

    #boundary ISS group
    for gr in np.arange(0,34):
        group_boundISS[indSL, gr] = np.mean(boundISS[indSL,sub_groupID == gr])

# spearman correlation age group x median duration without and with group_boundISS as covariate for 4 selected SLs
# load group data
data = loadmat('/home/lingee/wrkgrp/Selma/CamCAN_movie/highpass_filtered_intercept2/34groups/analyses_results/GSBS_obj.mat')
median_duration = data['median_duration']
median_duration = median_duration[SLs,:]
ngroups = 34
groups = np.arange(ngroups)
nregs = 4
nregs = 4

# empty arrays
age_dur = np.full([nregs], 0).astype(float)
pval_age_dur = np.full([nregs], 0).astype(float)
age_dur_iss = np.full([nregs], 0).astype(float)
pval_age_dur_iss = np.full([nregs], 0).astype(float)
age_stateISS = np.full([nregs], 0).astype(float)
pval_age_stateISS = np.full([nregs], 0).astype(float)

for SL in range(nregs):
     # Calculate the Spearman correlation between the ordinal (group) and continuous variable (median duration)
     age_dur[SL], pval_age_dur[SL] = stats.spearmanr(groups, median_duration[SL,:])
          # Calculate partial correlation with ISS as covariate
     data = {'x':groups, 'y':median_duration[SL,:], 'cv1':group_boundISS[SL,:]}
     df = pd.DataFrame(data)
     spearman_age_dur_iss = pg.partial_corr(data=df, x='x', y='y', covar='cv1', method='spearman')
     age_dur_iss[SL] = spearman_age_dur_iss['r']
     pval_age_dur_iss[SL] = spearman_age_dur_iss['p-val']
     age_stateISS[SL], pval_age_stateISS[SL] = stats.spearmanr(groups, group_boundISS[SL, :])

print("Age x Duration:", age_dur)
print("P-value Age x Duration:", pval_age_dur)
print("Age x Duration - covar boundary ISS:", age_dur_iss)
print("P-value Age x Duration - covar boundary ISS:", pval_age_dur_iss)
print("Age x State ISS:", age_stateISS)
print("P-value Age x State ISS:", pval_age_stateISS)