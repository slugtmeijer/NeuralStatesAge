%% denoising high pass filter
clear

%add paths
addpath(genpath('/home/sellug/wrkgrp/Cambridge_data/Toolboxes/SPM12/'))

%specify relevant directories
data_folder='/home/sellug/wrkgrp/Cambridge_data/Movie_HMM/Data_for_Donders/'; %ME_denoised data
out_folder='/home/sellug/wrkgrp/Selma/CamCAN_movie/highpass_filtered_intercept2/';

%data characteristics
fname='_s0w_ME_denoised.nii';
nTRs=192;
TR=2.47;
nsubs=length(dir([data_folder '*_s0w_ME_denoised.nii']));

%choose between a high pass or bandpass filter
filt='HP'; %'BP'

%start a pool of parallel workers
parpool(5)

parfor s=1:nsubs
%for s=1:1
    disp(s)

    subIDs=dir([data_folder '*_s0w_ME_denoised.nii']);
    subID=subIDs(s).name;
    
    %ME-ICA denoised data
    data_file=[data_folder subID];
    
    % Create a DCT filter
    Kall   = spm_dctmtx(nTRs,nTRs);
    nHP = fix(2*(nTRs*TR)/(1/0.008) + 1);
    
    if strcmp(filt,'HP')
        K   = Kall(:,[2:nHP]);
    elseif strcmp(filt,'BP')
        nLP = fix(2*(nTRs*TR)/(1/0.1) + 1);
        K   = Kall(:,[2:nHP nLP:nTRs]);
    end
    
    %add the intercept
    X0r=[ones(nTRs,1) K]; 

    %load the nifti file
    hdr=spm_vol(data_file);
    dat=spm_read_vols(hdr);
    dat=permute(dat(:,:,:,1:nTRs), [4 1 2 3]);
    
    %do the regression and keep the residuals
    R  = eye(nTRs) - X0r*pinv(X0r);
    aY = R*dat(:,:);
    
    %save the data in whole brain images
    for i=1:nTRs
        hdr_new=hdr(i);
        hdr_new.fname=[out_folder subID(1:9) '_s0w_ME_denoised_nr_' filt '.nii'];
        spm_write_vol(hdr_new, reshape(aY(i,:), [size(dat,2) size(dat,3) size(dat,4)]));
    end
    
end


