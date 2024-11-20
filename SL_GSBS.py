import matplotlib.pyplot as plt

from statesegmentation import GSBS
import numpy as np
from tqdm import tqdm
import config as cfg
import multiprocessing
from create_folder import create_folder

import nibabel as nib

nr_parallel_jobs = 30

groups = 34 # TODO change for 1 or 34 groups

basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
SL_dir = basedir + 'masks/searchlights/'
ngroups_dir = basedir + 'highpass_filtered_intercept2/' + str(groups) + 'groups/'
datadir = ngroups_dir + 'preGSBS/age_groups/'

stride = 2
radius = 3
min_vox = 15

params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)
coordinates = np.load(SL_dir + 'SL_voxelCoordinates_' + params_name + '.npy',allow_pickle=True)
searchlights = np.load(SL_dir + 'SL_voxelsIndices_'+ params_name + '.npy',allow_pickle= True)

#def GSBS_SL(data_SL, SL_idx, run_nr):
def GSBS_SL(data_SL, SL_idx):
    GSBS_SL_obj = GSBS(x=data_SL, kmax=int(data_SL.shape[0] * .5), finetune=1, blocksize=50, statewise_detection=True)
    GSBS_SL_obj.fit(False)
    savename = save_dir + 'GSBS_GR' + str(GR) + '_' + params_name + '_SL' + str(SL_idx)
    np.save(savename, GSBS_SL_obj)
    return


for GR in range(groups):
    cur_dir = 'GR' + str(GR) + '/'
    save_dir = ngroups_dir + 'GSBS_results/searchlights/' + cur_dir
    create_folder(save_dir)

    if __name__ == '__main__':
        processes = []

        #for run_nr in cfg.run_numbers:
        print("Start GSBS")

        # Load mean data
        data = np.load(datadir + cur_dir + 'hyperaligned/mean/mean_wholeBrain_allSubs_GR' + str(GR) + '.npy', allow_pickle=True)
        # Loop through all search lights
        for SL_idx, voxel_indices in enumerate(searchlights):
            print("RUNNING gsbs" + str(SL_idx))
            vox_coords = coordinates[voxel_indices]
            data_SL = []
            for x,y,z in vox_coords:
                data_SL.append(data[x,y,z,:])

            data_SL = np.transpose(np.asarray(data_SL)) # Go to time x voxel
            p = multiprocessing.Process(target=GSBS_SL, args=(data_SL, SL_idx,))
            processes.append(p)
            p.start()

            # To avoid too many jobs at the same time
            if len(processes) >= nr_parallel_jobs:
                for process in processes:
                    process.join()
                processes = []