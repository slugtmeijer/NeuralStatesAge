#from nltools.data import Brain_Data
from nibabel import load, save
from statesegmentation import GSBS
import nibabel as nib
import numpy as np
from tqdm import tqdm
import os
from create_folder import create_folder
groups = 34

basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/highpass_filtered_intercept2/'
ngroups_dir = basedir + str(groups) + 'groups/'
datadir = ngroups_dir + 'preGSBS/age_groups/'

for GR in range(groups):
    data_dir = datadir + 'GR' + str(GR) + "/hyperaligned/"
    save_dir = data_dir + "mean/"
    create_folder(save_dir)

    # Get list of subjects
    allfiles = os.listdir(data_dir)
    namelist=[]
    for names in allfiles:
        if names.endswith(".nii"):
         namelist.append(names)
    namelist.sort()

    data_allSubs = []

    sub = namelist[0]
    img = nib.load(data_dir + sub)

    # Get dimensions
    img_shape = nib.load(data_dir + namelist[0]).shape
    num_subjects = len(namelist)
    x_dim, y_dim, z_dim, time_dim = img_shape

    # Create empty array
    data_allSubs = np.zeros((num_subjects, x_dim, y_dim, z_dim, time_dim))

    # Load data for each subject
    for i, subject in enumerate(tqdm(namelist)):
        img = nib.load(data_dir + subject)
        data_allSubs[i] = img.get_fdata()

    # Take average and save
    print("Saving...")
    data_mean = np.mean(np.asarray(data_allSubs), axis=0)
    np.save(save_dir + 'mean_wholeBrain_allSubs_GR' + str(GR), data_mean)
    print("Done")