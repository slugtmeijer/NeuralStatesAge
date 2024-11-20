from scipy.io import loadmat
from scipy import stats
import sys
sys.path.append("..")
import numpy as np
import os

groups = 34

basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
ngroups_dir = basedir + 'highpass_filtered_intercept2/' + str(groups) + 'groups/'
GSBSdir = ngroups_dir + 'analyses_results/'

ntime=192
nregs=5204

# Load data GSBS
data = loadmat(GSBSdir + 'GSBS_obj.mat')
median_duration = data['median_duration']

# Create vector for groups for correlation
group = np.arange(groups)

# Empty array for correlation age and median duration
age_dur = np.full([nregs], 0).astype(float)
pval_age_dur = np.full([nregs], 0).astype(float)

for SL in range(nregs):
     # Calculate the Spearman correlation between the ordinal (group) and continuous variable (median duration)
     age_dur[SL], pval_age_dur[SL] = stats.spearmanr(group, median_duration[SL,:])

# Get indexes of searchlights with highest correlation with age

# Replace NaN values with 0
age_dur2 = np.nan_to_num(age_dur, nan=0.0)

# Get the indices that would sort the array in descending order
sorted_indices = np.argsort(age_dur2)[::-1]

# Select the top 5 indices
top_10_indices = sorted_indices[:10]

# Get the top 5 values
top_10_values = age_dur2[top_10_indices]



# same for overlap events - neural states
overlap_dir = GSBSdir + 'event_boundaries_binary_1swindow/'

# Load the saved variables
abs_overlap = np.load(os.path.join(overlap_dir, "abs_overlap_vector.npy"))

# Replace NaN values with 0
abs_overlap2 = np.nan_to_num(abs_overlap, nan=0.0)

# Get the indices that would sort the array in descending order
sorted_indices = np.argsort(abs_overlap2)[::-1]

# Select the top 5 indices
top_10_indices = sorted_indices[:10]

# Get the top 5 values
top_10_values = abs_overlap2[top_10_indices]

print


