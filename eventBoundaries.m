% get subjective event boundaries for HRF 5s (and for 4 and 6 seconds)

dir1 = '/home/sellug/wrkgrp/Selma/scripts/NestedHierarchy-main/Helper files/';
cd(dir1)
savedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/';

Ntime=192;
TR=2.47;

%load event info
load(['subjective_event_onsets.mat'])
for i = 1:3
    if i == 1 % orig HRF
        savename = [savedir, 'event_boundaries_subj.mat'];
        n = 5;
    elseif i == 2 % HRF -1s
        savename = [savedir, 'event_boundaries_subj_m1s.mat'];
        n = 4;
    elseif i == 3 % HRF +1s
        savename = [savedir, 'event_boundaries_subj_p1s.mat'];
        n = 6;
    end
    event_onsets_subjective=floor((event_onsets+n)./TR)+1;
    event_boundaries_subj=zeros(Ntime,1); %which TR
    event_boundaries_subj(event_onsets_subjective)=1;
    event_boundaries_subj=event_boundaries_subj(2:Ntime);
    
    %save event_boundaries_subj
    save(savename, 'event_boundaries_subj');
end