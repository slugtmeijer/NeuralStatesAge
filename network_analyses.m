% neural states
% 1) correlation between age and median neural state duration per network
% 2) overlap perceived boundaries and state boundaries per network

clear

folder = '/Users/selmalugtmeijer/Library/CloudStorage/OneDrive-UniversityofBirmingham/PostDoc_Brock/Project2_movie/neuralstates/writing/communicationsBiology/Revision/Analyses/';
% read in atlas
hdr=spm_vol([folder, 'rSchaefer2018_200Parcels_1mm.nii']);
img_atlas=spm_read_vols(hdr); 

% read in sig effect age on state duration
hdr=spm_vol([folder, 'r_analysis_age_durations.nii']);
img_age_dur=spm_read_vols(hdr); 

% read in sig effect age on overlap event boundaries and neural states
hdr=spm_vol([folder, 'r_analysis_abs_events_all_groups.nii']);
img_overlap=spm_read_vols(hdr); 

% read in parcel labels and network mapping
parcel_labels_file = [folder, 'rSchaefer2018_200Parcels_1mm.txt'];
fid = fopen(parcel_labels_file, 'r');
if fid == -1
    error('Could not open file: %s', parcel_labels_file);
end

% Read the file line by line
parcel_ids = [];
parcel_names = {};
line_count = 0;

while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(line)
        line_count = line_count + 1;
        parts = strsplit(line, '|');
        if length(parts) == 2
            parcel_ids(end+1) = str2double(parts{1});
            parcel_names{end+1} = parts{2};
        else
            fprintf('Warning: Could not parse line %d: %s\n', line_count, line);
        end
    end
end
fclose(fid);

fprintf('Successfully read %d parcels from file\n', length(parcel_ids));

% Extract network information from parcel names and create mapping
parcel_to_network = containers.Map('KeyType', 'int32', 'ValueType', 'any');
network_list = {};

for i = 1:length(parcel_names)
    name_parts = split(parcel_names{i}, '_');
    % Extract network name
    if length(name_parts) >= 3
        network_name = name_parts{3};
        parcel_to_network(parcel_ids(i)) = network_name;
        
        % Keep track of unique networks
        if ~ismember(network_name, network_list)
            network_list{end+1} = network_name;
        end
    end
end

fprintf('Found %d unique networks: %s\n', length(network_list), strjoin(network_list, ', '));

% Count total parcels per network
network_totals = containers.Map();
for network = network_list
    network_totals(network{1}) = 0;
end

for i = 1:length(parcel_ids)
    parcel_id = parcel_ids(i);
    if parcel_to_network.isKey(parcel_id)
        network_name = parcel_to_network(parcel_id);
        network_totals(network_name) = network_totals(network_name) + 1;
    end
end

%% Calculate average values per network for age-duration and abs overlap 

% Initialize containers for storing results
network_age_dur_avg = containers.Map();
network_overlap_avg = containers.Map();
network_voxel_counts = containers.Map();

fprintf('Creating network masks and calculating averages...\n');

% Process each network
for i = 1:length(network_list)
    network_name = network_list{i};
    fprintf('Processing network: %s\n', network_name);
    
    % Find all parcels belonging to this network
    parcels_in_network = [];
    for j = 1:length(parcel_ids)
        parcel_id = parcel_ids(j);
        if parcel_to_network.isKey(parcel_id) && strcmp(parcel_to_network(parcel_id), network_name)
            parcels_in_network(end+1) = parcel_id;
        end
    end
    
    % Create a mask for all voxels in this network
    network_mask = false(size(img_atlas));
    for parcel_id = parcels_in_network
        network_mask = network_mask | (img_atlas == parcel_id);
    end
    
    % Count voxels in this network
    n_voxels = sum(network_mask(:));
    network_voxel_counts(network_name) = n_voxels;
    
    % Extract all values for this network from both statistical maps
    age_dur_values = img_age_dur(network_mask);
    overlap_values = img_overlap(network_mask);

    % Calculate single average across ALL voxels in the network
    % (excluding NaN values)
    valid_dur = age_dur_values(~isnan(age_dur_values));
    valid_overlap = overlap_values(~isnan(overlap_values));

    if ~isempty(valid_dur)
        network_age_dur_avg(network_name) = mean(valid_dur);
    else
        network_age_dur_avg(network_name) = NaN;
    end

    if ~isempty(valid_overlap)
        network_overlap_avg(network_name) = mean(valid_overlap);
    else
        network_overlap_avg(network_name) = NaN;
    end

    fprintf('  - %d voxels found\n', n_voxels);
end

%% Display results
fprintf('\n=== RESULTS ===\n');
fprintf('\n--- Age Effect on State Duration (r_analysis_age_durations) ---\n');
fprintf('Network\t\t\tAverage Value\tN Voxels\tN Parcels\n');
fprintf('-------\t\t\t-------------\t---------\t---------\n');

for i = 1:length(network_list)
    network_name = network_list{i};
    avg_val = network_age_dur_avg(network_name);
    n_voxels = network_voxel_counts(network_name);
    n_parcels = network_totals(network_name);
    fprintf('%-15s\t%.4f\t\t%d\t\t%d\n', network_name, avg_val, n_voxels, n_parcels);
end

fprintf('\n--- Event Boundary Overlap (r_analysis_abs_events_all_groups) ---\n');
fprintf('Network\t\t\tAverage Value\tN Voxels\tN Parcels\n');
fprintf('-------\t\t\t-------------\t---------\t---------\n');

for i = 1:length(network_list)
    network_name = network_list{i};
    avg_val = network_overlap_avg(network_name);
    n_voxels = network_voxel_counts(network_name);
    n_parcels = network_totals(network_name);
    fprintf('%-15s\t%.4f\t\t%d\t\t%d\n', network_name, avg_val, n_voxels, n_parcels);
end

%% Create summary table
network_names_cell = network_list';
age_dur_averages = cellfun(@(x) network_age_dur_avg(x), network_names_cell);
overlap_averages = cellfun(@(x) network_overlap_avg(x), network_names_cell);
n_voxels_array = cellfun(@(x) network_voxel_counts(x), network_names_cell);
n_parcels_array = cellfun(@(x) network_totals(x), network_names_cell);

% Create summary table
results_table = table(network_names_cell, age_dur_averages, overlap_averages, ...
                     n_voxels_array, n_parcels_array, ...
                     'VariableNames', {'Network', 'AgeDuration_Avg', 'Overlap_Avg', ...
                                      'N_Voxels', 'N_Parcels'});

fprintf('\n--- Summary Table ---\n');
disp(results_table);

%% Create plots for network correlations

% Prepare data for plotting
network_names_cell = network_list';
age_dur_averages = cellfun(@(x) network_age_dur_avg(x), network_names_cell);
overlap_averages = cellfun(@(x) network_overlap_avg(x), network_names_cell);

% Remove any networks with NaN values for cleaner plots
valid_dur_idx = ~isnan(age_dur_averages);
valid_overlap_idx = ~isnan(overlap_averages);

% Create figure with two subplots
figure('Position', [100, 100, 1200, 800]);

%% Plot 1: Age vs State Duration
subplot(1, 2, 1);
valid_networks_dur = network_names_cell(valid_dur_idx);
valid_values_dur = age_dur_averages(valid_dur_idx);

% Create horizontal bar plot
barh(1:length(valid_networks_dur), valid_values_dur);
set(gca, 'YTick', 1:length(valid_networks_dur));
set(gca, 'YTickLabel', valid_networks_dur);
xlabel('Correlation age x state duration', 'FontSize', 16);
%ylabel('Networks');
%title('Age Effect on Neural State Duration by Network');
set(gca, 'FontSize', 14); % Increase axis numbers and tick labels
grid off;

% Add a vertical line at x=0 for reference
hold on;
plot([0 0], [0.5, length(valid_networks_dur)+0.5], 'k--', 'LineWidth', 1);
hold off;

% Color bars based on positive/negative values
h1 = gca;
bars1 = h1.Children(end); % Get the bar object (last child before the reference line)
for i = 1:length(valid_values_dur)
    if valid_values_dur(i) > 0
        bars1.CData(i,:) = [0.2 0.6 0.8]; % Blue for positive
    else
        bars1.CData(i,:) = [0.8 0.3 0.3]; % Red for negative
    end
end

%% Plot 2: Age vs Event Boundary Overlap
subplot(1, 2, 2);
valid_networks_overlap = network_names_cell(valid_overlap_idx);
valid_values_overlap = overlap_averages(valid_overlap_idx);

% Create horizontal bar plot
barh(1:length(valid_networks_overlap), valid_values_overlap);
set(gca, 'YTick', 1:length(valid_networks_overlap));
set(gca, 'YTickLabel', valid_networks_overlap);
xlabel('Overlap perceived event boundaries and neural states', 'FontSize', 16);
%ylabel('Networks');
%title('Age Effect on Event Boundary Overlap by Network');
set(gca, 'FontSize', 14); % Increase axis numbers and tick labels
grid off;

% Add a vertical line at x=0 for reference
hold on;
plot([0 0], [0.5, length(valid_networks_overlap)+0.5], 'k--', 'LineWidth', 1);
hold off;

% Color bars based on positive/negative values
h2 = gca;
bars2 = h2.Children(end); % Get the bar object
for i = 1:length(valid_values_overlap)
    if valid_values_overlap(i) > 0
        bars2.CData(i,:) = [0.2 0.6 0.8]; % Blue for positive
    else
        bars2.CData(i,:) = [0.8 0.3 0.3]; % Red for negative
    end
end

%% Adjust layout
% Make sure network labels are fully visible
subplot(1, 2, 1);
pos1 = get(gca, 'Position');
set(gca, 'Position', [pos1(1)+0.05, pos1(2), pos1(3)-0.05, pos1(4)]);

subplot(1, 2, 2);
pos2 = get(gca, 'Position');
set(gca, 'Position', [pos2(1)+0.05, pos2(2), pos2(3)-0.05, pos2(4)]);

% Add overall title
sgtitle('Network involvement', 'FontSize', 16, 'FontWeight', 'bold');

%% Optional: Save the figure
% saveas(gcf, [folder, 'network_age_correlations.png']);
% saveas(gcf, [folder, 'network_age_correlations.fig']);

%% Display summary statistics
fprintf('\n=== PLOT SUMMARY ===\n');
fprintf('State Duration Plot:\n');
fprintf('  - Networks with positive correlations: %d\n', sum(valid_values_dur > 0));
fprintf('  - Networks with negative correlations: %d\n', sum(valid_values_dur < 0));
fprintf('  - Range: %.4f to %.4f\n', min(valid_values_dur), max(valid_values_dur));

fprintf('\nEvent Boundary Overlap Plot:\n');
fprintf('  - Networks with positive correlations: %d\n', sum(valid_values_overlap > 0));
fprintf('  - Networks with negative correlations: %d\n', sum(valid_values_overlap < 0));
fprintf('  - Range: %.4f to %.4f\n', min(valid_values_overlap), max(valid_values_overlap));