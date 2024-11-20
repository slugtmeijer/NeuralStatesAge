import numpy as np
from mvpa2.suite import *
from nibabel import save
import pickle
import time
import scipy.io
import os

ngroups = 34 #TODO change 1 (whole group) or 34 (age groups)
subjs = 'ids_34x577.mat' #TODO change (ids_1x577 or ids_34x577)

# Directories
home_dir = '/home/sellug/wrkgrp/Selma/'
basedir = home_dir + 'CamCAN_movie/'
maskdir = basedir + 'masks/'
datadir = basedir + 'highpass_CSF_WM_motion_filtered2/'
ngroups_dir = datadir + str(ngroups) + 'groups/'
preGSBS_dir = ngroups_dir + 'preGSBS/age_groups/'

def create_folder(folder):
    # Check if the folder already exists
    if not os.path.exists(folder):
        # If it doesn't exist, create it
        os.makedirs(folder)
        print(f"Folder '{folder}' created.")
    else:
        print(f"Folder '{folder}' already exists.")


# Function to compute the reference subject based on ISS
def compute_ISS(run_datasets,nTime):
    subsim = np.empty((len(run_datasets), len(run_datasets)))
    for i in range(len(run_datasets)):
        for j in range(i, len(run_datasets)):
            subsim[i, j] = np.mean(nTime * np.mean(np.multiply(run_datasets[i], run_datasets[j]), 0)/(nTime-1))
            subsim[j, i] = subsim[i, j]
    refsub = np.argmax(np.mean(subsim, 0))
    return refsub


for g in range(ngroups):
    print(g)
    saved_variables_dir = preGSBS_dir + 'GR' + str(g) + '/hyperaligned/'
    create_folder(saved_variables_dir)
    # load subject information based on id list in mat
    data = scipy.io.loadmat(home_dir + subjs)
    ids = data["groupIDs"]
    ids_vec = np.squeeze(ids[0, g])
    full_file_names = []
    suffix = "_s0w_ME_denoised_nr_HP.nii"
    for file_name in ids_vec:
        full_file_name = os.path.join(datadir, str("CBU") + str(file_name))
        full_file_name += suffix
        full_file_names.append(full_file_name)
    namelist = full_file_names
    #namelist = namelist[0:2] #switch on to run on subset of 2 subjects

    mask_file = maskdir + 'data_plus_GM_mask.nii'
    mask = fmri_dataset(samples=mask_file, mask=mask_file)
    mask = np.where(mask.samples>0)[1]

    # Load all the data for the SL hyperalignment
    run_datasets = []

    for count, name in enumerate(namelist):
        print(count/len(namelist)*100)
        print("Loading " + name)

        filename = name
        alldata = fmri_dataset(samples=filename, mask=mask_file)

        nTimePoints = alldata.samples.shape[0]

        alldata.sa._uniform_length = nTimePoints
        alldata.sa['time_indices'] = range(nTimePoints)
        alldata.sa['time_coords'] = np.zeros(nTimePoints)
        zscore(alldata, chunks_attr=None, param_est=None)
        run_datasets.append(alldata)

    # Reference subject
    refsub = compute_ISS(run_datasets, nTimePoints)
    refsub_name = namelist[refsub]
    print("Reference subject: " + refsub_name)

    # Save name of reference subject
    with open(saved_variables_dir + 'refsub.pkl', 'wb') as f:
        pickle.dump(refsub_name, f)

    # Settings for hyperalignment
    hyper = Hyperalignment(level1_equal_weight=True)

    # Settings for SL hyperalignment
    slhyper = SearchlightHyperalignment(radius=3, sparse_radius=2, ref_ds=refsub, nblocks=50, compute_recon=False, hyperalignment=hyper, mask_node_ids=mask)

    # Compute hyperalignment parameters
    print("Computing hyperalignment parameters")
    start_time = time.time()
    slhypmaps = slhyper(run_datasets)
    end_time = time.time()
    print("Done")
    print("Computation time: %s seconds" % (end_time - start_time))

    # Save computed parameters and list of subject names
    with open(saved_variables_dir + 'slhypmaps.pkl', 'wb') as f:
        pickle.dump([slhypmaps, namelist], f)

    # Apply hypperalignment parameters on all data
    for count, name in enumerate(namelist):
        print(count / len(namelist) * 100)
        filename = name
        data_run = fmri_dataset(samples=filename, mask=mask_file)

        nTimePoints = data_run.samples.shape[0]
        data_run.sa._uniform_length = nTimePoints
        data_run.sa['time_indices'] = range(nTimePoints)
        data_run.sa['time_coords'] = np.zeros(nTimePoints)
        zscore(data_run, chunks_attr=None, param_est=None)

        # Apply parameters
        print("Applying parameters on " + name + " run " + str(run))
        data_run_hyper = slhypmaps[count].forward(data_run)

        # Save hyperaligned data
        print("Saving hyperaligned data " + name[-32:-4] + " run " + str(run))
        img = map2nifti(data_run_hyper)
        save(img, saved_variables_dir + name[-32:-4] + '_hyperaligned.nii')