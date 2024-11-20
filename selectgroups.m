%Select age groups

subinfo = open('/home/sellug/wrkgrp/Selma/subinfo_CBU_CC_age_groups.mat');
subinfo = subinfo.subinfo_CBU_CC_age_groups;

[~,idx] = sort(subinfo(:,3)); %sort on age
sortinfo = subinfo(idx,:);

n = [(zeros(1,1)+16),(zeros(1,33)+17)]; %number of subjects per group, in this case 1 group of 16 and 33 groups of 17
grs = (1:34); %number of groups
gr = repelem(grs, n)';

info = [sortinfo(:,1:3),gr];

info_CBU_age_group = sortrows(info, 1); %sort back on CBUid

[mean,std,min,max] = grpstats(info(:,3),info(:,4),{"mean","std","min","max"});

%get which CBUIDs are in each group
for i = 1:length(grs)
    IDs=info(info(:,4)==i); %CHECK WHICH COLUMN
    groupIDs{i} = IDs;
end

save('/home/sellug/wrkgrp/Selma/ids_34x577.mat', 'groupIDs')
save('/home/sellug/wrkgrp/Selma/CamCAN_movie/subinfo_CBU_age_group.mat', 'info_CBU_age_group')

% now save groupIDs for 1 group of all participants
ID_all =  info_CBU_age_group(:,1);
groupIDs = {ID_all};
save('/home/sellug/wrkgrp/Selma/CamCAN_movie/ids_1x577.mat', 'groupIDs')