import nibabel as nib
import numpy as np
import os

groups = 1 #TODO change for 1 or 34 groups

basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
ngroups_dir = basedir + ('highpass_filtered_intercept2/') + str(groups) + 'groups/'
groupdir = ngroups_dir + 'preGSBS/age_groups/'
meanfolder = 'mean/'

SL = 5204

stride = 2
radius = 3
min_vox = 15

params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)
coordinates = np.load(basedir + 'masks/searchlights/SL_voxelCoordinates_' + params_name + '.npy',allow_pickle=True)
searchlights = np.load(basedir + 'masks/searchlights/SL_voxelsIndices_'+ params_name + '.npy',allow_pickle= True)

ISS = np.full([SL,groups],0).astype(float)

for GR in range(groups):

    datadir = groupdir + 'GR' + str(GR) + '/hyperaligned/'

    # get mean image
    mean_file = (datadir + meanfolder + 'mean_wholeBrain_allSubs_GR' + str(GR) + '.npy')
    mean_img = np.load(mean_file)

    # get nr of subjects per group based on hyperaligned .nii files in one group folder
    # Get list of subjects
    allfiles = os.listdir(datadir)
    namelist = []
    for names in allfiles:
        if names.endswith(".nii"):
            namelist.append(names)
    namelist.sort()
    nr_sub = len(namelist)

    correlation_coefficients = np.full([nr_sub, len(searchlights)], 0).astype(float)

    for s in range(nr_sub):
        sub = namelist[s]
        img = nib.load(datadir + sub)
        img_sub = img.get_fdata()

        # Loop through all searchlights
        for SL_idx, voxel_indices in enumerate(searchlights):
            print('group ' + str(GR) + ' subject ' + str(s) + ' SLindex ' + str(SL_idx))
            vox_coords = coordinates[voxel_indices]
            data_SL_mean = []
            data_SL_subj = []
            for x, y, z in vox_coords:
                data_SL_mean.append(mean_img[x, y, z, :])
                data_SL_subj.append(img_sub[x, y, z, :])

            data_SL_mean = np.transpose(np.asarray(data_SL_mean))  # Go to time x voxel
            data_SL_subj = np.transpose(np.asarray(data_SL_subj))

            multiplied_img = data_SL_mean * nr_sub
            leave1out = multiplied_img - data_SL_subj

            #get correlation between 1 subject and the group minus that subject
            correlation_coefficient = np.corrcoef(data_SL_subj.flatten(), leave1out.flatten())[0,1]

            correlation_coefficients[s, SL_idx] = correlation_coefficient
    # Get mean correlation across subjects in this group
    group_mean_correlation = np.mean(correlation_coefficients, axis=0)
    #np.save(datadir + meanfolder + 'group_mean_correlation_per_SL_GR' + str(GR), group_mean_correlation)
    ISS[:,GR] = group_mean_correlation
np.save(groupdir + 'all_groups_mean_correlation_per_SL', ISS)

# Save individual ISS values per SL if doing analyses over 1 group
if groups == 1:
    np.save(groupdir + 'ISS_perSL_perSubj', correlation_coefficients)

print("Done")


