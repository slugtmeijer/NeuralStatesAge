#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:11:48 2021

@author: dorgoz but actually also Djamari
"""

import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

datadir = "/home/sellug/wrkgrp/Selma/CamCAN_movie/highpass_filtered_intercept2/34groups/preGSBS/age_groups/GR0/hyperaligned/"
savedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/masks/searchlights_temp/'

# Settings - does not need to be the same as hyperalignment; about 100 voxels for GSBS - depends on voxel size
stride = 2
radius = 3
min_vox = 15

def create_searchLights(dataset, mask_path, stride, radius, min_vox):
    """Returns coordinates of every voxel and voxel indices of every searchlight.
    Parameters
    ----------
    dataset : Dataset
        3D Voxel Space
    mask_path : Path String
        Path to the mask to be used
    stride : int, optional
        Specifies amount by which searchlights move across data
    radius : int, optional
        Specifies radius of each searchlight
    min_vox : int, optional
        Indicates the minimum number of elements with data for each searchlight
    Returns
    -------
    coords : list
        Coordinates of voxel indices
    SL_allvox : list
        Voxels in each searchlight
    """

    # Load the Mask
    print('Loading mask')
    img = nib.load(mask_path)
    mask = img.get_fdata()
    print('Mask Loaded')

    print('Creating searchlights')
    #coords in index values (not MNI)
    coords = np.transpose(np.where(dataset < np.max(dataset) + 1))
    SL_allvox = []
    centers = np.zeros_like(mask)
    nr_centers = np.zeros_like(mask)
    for x in tqdm(range(0, np.max(coords, axis=0)[0], stride)):
        for y in range(0, np.max(coords, axis=0)[1], stride):
            for z in range(0, np.max(coords, axis=0)[2], stride):
                if mask[x,y,z] > 0:
                    dists = cdist(coords, np.array([[x, y, z]]))[:, 0]
                    SL_vox = np.where(dists <= radius)[0]

                    xs = []
                    ys = []
                    zs = []
                    # take the overlap with mask
                    SL_vox_masked = []
                    for vox in SL_vox:
                        x_vox, y_vox, z_vox = coords[vox]

                        if mask[x_vox,y_vox,z_vox] > 0:
                            SL_vox_masked.append(vox)
                            xs.append(x_vox)
                            ys.append(y_vox)
                            zs.append(z_vox)


                    # Only add SL_vox_masked to SL_allvox if length is above min_vox
                    if len(SL_vox_masked) >= min_vox:
                        SL_allvox.append(SL_vox_masked)

                        # Compute center and add to image
                        x_center = int(np.mean(xs))
                        y_center = int(np.mean(ys))
                        z_center = int(np.mean(zs))
                        centers[x_center, y_center, z_center] = len(SL_allvox) - 1  # Index number of searchlight
                        nr_centers[x_center, y_center, z_center] +=1

    print('Searchlights are created')
    # Save center image
    map_SL_indices = nib.Nifti1Image(centers, img.affine)
    nib.save(map_SL_indices, savedir + 'SL_indices.nii')

    # create and save sum image
    nr_of_SLs_per_vox = np.zeros_like(mask)
    for SL_vox in SL_allvox:
        for vox in SL_vox:
            x, y, z = coords[vox]
            nr_of_SLs_per_vox[x,y,z] += 1
    nib.save(nib.Nifti1Image(nr_of_SLs_per_vox, img.affine), savedir + 'nr_of_SLs_per_vox.nii')

    return coords, SL_allvox


# Set the mask path
mask_path = '/home/sellug/wrkgrp/Selma/CamCAN_movie/masks/data_plus_GM_mask.nii'

# Load fMRI data and get only 1 TR for dimensions, needs to be x, y, z, t #TODO SELMA check if this is okay, was reading in a npy
fmri = nib.load(datadir + '110544_s0w_ME_denoised_nr_HP_hyperaligned.nii')
data = fmri.get_fdata()

data_to_get_searchlight_coordinates = data[:, :, :, 0]

# Create searchlights
coordinates, searchlights = create_searchLights(data_to_get_searchlight_coordinates, mask_path, stride, radius, min_vox)


print('Number of searchlights:')
print(len(searchlights))

# Save voxel coordinates by index and searchlight voxels by index
params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)
np.save(savedir + 'SL_voxelCoordinates_' + params_name + '.npy', coordinates, )
np.save(savedir + 'SL_voxelsIndices_'+ params_name + '.npy', searchlights, )
