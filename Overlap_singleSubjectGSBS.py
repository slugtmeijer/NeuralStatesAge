import numpy as np
import os
import nibabel as nib
from scipy.io import loadmat
from scipy import stats
from compute_overlap_1swindow import compute_overlap_1swindow
from create_folder import create_folder


def run_analysis(subs_range, savedir_suffix):
    subs = len(subs_range)

    # create empty arrays for overlap
    abs_overlap_events = np.full([Nregs, subs], 0).astype(float)

    # first TR doesn't count because can't be a boundary
    time_range = slice(1, None)
    # correlation / overlap over time per subject per searchlight
    for s_idx, s in enumerate(subs_range):
        s_data = binbounds[:, s, time_range]
        abs_overlap = compute_overlap_1swindow(2, s_data, event_boundaries, event_boundaries_p1s,
                                               event_boundaries_m1s)
        abs_overlap_events[:, s_idx] = abs_overlap

    # get correlation age and overlap per SL
    # empty arrays
    age_overlap = np.full([Nregs], 0).astype(float)
    pval_age_overlap = np.full([Nregs], 0).astype(float)
    results_text = ""
    for SL in range(Nregs):
        # spearmanr
        age_overlap[SL], pval_age_overlap[SL] = stats.spearmanr(age, abs_overlap_events[SL,:])
        #print(f"SL_{SL_ind[SL]}: r = {age_overlap[SL]:.3f}, p = {pval_age_overlap[SL]:.3f}")
        line = f"SL_{SL_ind[SL]}: r = {age_overlap[SL]:.3f}, p = {pval_age_overlap[SL]:.3f}\n"
        print(line, end='')  # Print to console
        results_text += line  # Add to string

        # Save to text file
    with open(savedir + f'correlation_results_{savedir_suffix}.txt', 'w') as f:
        f.write(results_text)

    # Return the abs_overlap_events array for further use
    return abs_overlap_events

# Main script
groups = 1
subs = 577
SL_ind = [692, 2463, 1874, 2466]
Nregs = len(SL_ind)
Ntime = 192
basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
ngroups_dir = basedir + 'highpass_filtered_intercept2/' + str(groups) + 'groups/'
datadir = ngroups_dir + 'GSBS_results/searchlights/ind_GSBS/'
savedir = ngroups_dir + 'analyses_results/event_boundaries_binary_1swindow/'
create_folder(savedir)

# Initialize binbounds array (4 regions × 577 subjects × 192 timepoints)
binbounds = np.zeros((Nregs, subs, Ntime))

# Load data
for n in range(Nregs):
    strengths = np.load(os.path.join(datadir, f'strenghts_SL{SL_ind[n]}.npy'))
    binbounds_n = strengths > 0
    binbounds[n, :, :] = binbounds_n

# Load age
# Get subjects age - vector is in the same order as the subject files in the datadir
CBU_info = loadmat(basedir + 'subinfo_CBU_age.mat')
var = 'subinfo_CBU_age'
CBU_age = CBU_info[var]
age = CBU_age[:, 1]

ev_boundaries = loadmat(basedir + 'event_boundaries_subj.mat')
event_boundaries_subj = ev_boundaries['event_boundaries_subj']
event_boundaries = event_boundaries_subj.reshape((1, 191))

ev_boundaries_p1s = loadmat(basedir + 'event_boundaries_subj_p1s.mat')
event_boundaries_subj_p1s = ev_boundaries_p1s['event_boundaries_subj_p1s']
event_boundaries_p1s = event_boundaries_subj_p1s.reshape((1, 191))

ev_boundaries_m1s = loadmat(basedir + 'event_boundaries_subj_m1s.mat')
event_boundaries_subj_m1s = ev_boundaries_m1s['event_boundaries_subj_m1s']
event_boundaries_m1s = event_boundaries_subj_m1s.reshape((1, 191))

# Load gray matter mask
img = nib.load(basedir + 'masks/data_plus_GM_mask.nii')
affine = img.affine

# Load searchlight information
stride = 2
radius = 3
min_vox = 15
params_name = f'stride{stride}_radius{radius}_minvox{min_vox}'
SL_dir = basedir + 'masks/searchlights/'
coordinates = np.load(SL_dir + f'SL_voxelCoordinates_{params_name}.npy', allow_pickle=True)
searchlights = np.load(SL_dir + f'SL_voxelsIndices_{params_name}.npy', allow_pickle=True)

# Run: Analysis for all subjects
abs_overlap_events = run_analysis(range(577), 'indiv_subs')

mean_abs_overlap = np.mean(abs_overlap_events, axis=1)
