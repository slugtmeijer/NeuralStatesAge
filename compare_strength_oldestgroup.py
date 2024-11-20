import numpy as np
from scipy.io import loadmat
from scipy import stats
import matplotlib.pyplot as plt

subs = 577
groups=34
Nregs = 5204
Ntime = 192
basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
ngroups_dir = basedir + 'highpass_filtered_intercept2/' + str(groups) + 'groups/'
onegroup_dir = basedir + 'highpass_filtered_intercept2/1groups/'


# get data from the oldest group (age group GSBS)
# get boundary strength for all groups for all searchlights
data = loadmat(ngroups_dir + 'analyses_results/GSBS_obj.mat')
strengths = data['strengths']
# only data from the last group
strengths_gr34 = strengths[:,33,:]

# mean across boundary TRs for all SLs for group 34 - (age group GSBS)
strengths_gr34_allSL = np.zeros((Nregs), dtype=float)
for SL in range(Nregs):
    strengths_this_searchlight = strengths_gr34[SL]

    # The strength timelines now contains a lot of zeros for timepoints without boundary at the group level. Here I am removing them
    timepoints_without_boundary = np.where(strengths_this_searchlight == 0)[0]
    strengths_this_searchlight = np.delete(strengths_this_searchlight, timepoints_without_boundary)

    # Get the mean boundary strength for this searchlight
    mean_strengths_this_searchlight = np.mean(strengths_this_searchlight)
    strengths_gr34_allSL[SL] = mean_strengths_this_searchlight

strengths_gr34_mean = strengths_gr34_allSL

# Get data from individual strengths based on whole group GSBS
strengths_individual_all = np.load(onegroup_dir + 'GSBS_results/searchlights/' + 'individual_strengths_per_SL.npy', allow_pickle=True).item()

# Loop over searchlights
strengths_577_allSL = np.zeros((Nregs, subs), dtype=float)
for SL in strengths_individual_all.keys():
    strengths_this_searchlight = strengths_individual_all[SL]

    # The strength timelines now contains a lot of zeros for timepoints without boundary at the group level. Here I am removing them
    timepoints_without_boundary = np.where(strengths_this_searchlight[0, :] == 0)[0]
    strengths_this_searchlight = np.delete(strengths_this_searchlight, timepoints_without_boundary,
                                           1)  # Subject x Boundary

    # Get the mean boundary strength for this searchlight per subject
    mean_strengths_this_searchlight = np.mean(strengths_this_searchlight, axis=1)
    strengths_577_allSL[SL, :] = mean_strengths_this_searchlight

# Get only the subjects that are in group 34
# Get subjects groups - vector is in the same order as the subject files in the datadir
CBU_info = loadmat(basedir + 'subinfo_CBU_age_group.mat')
var = 'info_CBU_age_group'
CBU_var = CBU_info[var]
group = CBU_var[:, 3]
indices_group_34 = np.where(group == 34)[0]
strengths_indiv34 = strengths_577_allSL[:, indices_group_34]
# Take mean over subjects
strengths_indiv34_mean = np.mean(strengths_indiv34, axis=1)

# ttest difference between 2 groups
t_stat, p_value = stats.ttest_ind(strengths_gr34_mean, strengths_indiv34_mean)
print("Mean strength across all SLs based on age group GSBS - gr34: ", np.mean(strengths_gr34_mean))
print("Mean strength across all SLs - mean of individuals in group 34 based on 1 group GSBS: ", np.mean(strengths_indiv34_mean))
print("T-statistic:", t_stat)
print("P-value:", p_value)


# approach 2, test mean strength over SL for the 17 oldest subjects from whole group GSBS against mean of group 34
strengths_gr34_meansubs = np.mean(strengths_gr34_mean)

strengths_indiv = np.mean(strengths_indiv34, axis=0)

t_statistic, p_value = stats.ttest_1samp(strengths_indiv, strengths_gr34_meansubs)
print("Mean strength, mean over SLs, based on age group GSBS - gr34 test value: ", np.mean(strengths_gr34_mean))
print("Mean strength, mean over SLs, test across 17 individuals in group 34 based on 1 group GSBS: ", np.mean(strengths_indiv34_mean))
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# test assumption of normality
# 1. Histogram
plt.figure(figsize=(10, 4))
plt.hist(strengths_indiv34_mean, bins=10, color='skyblue', edgecolor='black')
plt.title("Histogram of the Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# 2. Q-Q Plot
plt.figure(figsize=(6, 6))
stats.probplot(strengths_indiv34_mean, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()

# 3. Shapiro-Wilk Test
shapiro_stat, p_value = stats.shapiro(strengths_indiv34_mean)
print("Shapiro-Wilk Test Statistic:", shapiro_stat)
print("p-value:", p_value)


print