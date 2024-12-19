import nibabel as nib
import numpy as np

# #compare files
# HP = np.load('/home/sellug/wrkgrp/Selma/CamCAN_movie/highpass_filtered_only/34groups/preGSBS/age_groups/GR0/hyperaligned/mean/mean_wholeBrain_allSubs_GR0.npy')
# HPM = np.load('/home/sellug/wrkgrp/Selma/CamCAN_movie/highpass_filtered_extramotiondenoising/34groups/preGSBS/age_groups/GR0/hyperaligned/mean/mean_wholeBrain_allSubs_GR0.npy')
# if np.array_equal(HP, HPM):
#     identical = 1
#     print("The arrays are identical.")
# else:
#     identical = 2
#     print("The arrays are not identical.")


def compare_nifti_files(file1_path, file2_path):
    # Load the NIfTI files
    img1 = nib.load(file1_path)
    img2 = nib.load(file2_path)

    # Get the data arrays from the images
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # Check if the data arrays are identical
    if np.array_equal(data1, data2):
        identical = 1
        print("The NIfTI files are identical.")
    else:
        identical = 2
        print("The NIfTI files are not identical.")


# Example usage:
file1_path = "/home/sellug/wrkgrp/Selma/CamCAN_movie/highpass_motion_filtered_orig/CBU110252_s0w_ME_denoised_nr_HP.nii"
file2_path = "/home/sellug/wrkgrp/Selma/CamCAN_movie/used_for_analyses/highpass_filtered_intercept/CBU110252_s0w_ME_denoised_nr_HP.nii"
compare_nifti_files(file1_path, file2_path)