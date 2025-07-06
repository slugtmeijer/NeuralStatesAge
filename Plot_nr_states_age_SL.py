from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn

groups = 34

basedir = '/home/lingee/wrkgrp/Selma/CamCAN_movie/'
ngroups_dir = basedir + 'highpass_filtered_intercept2/' + str(groups) + 'groups/'
datadir = ngroups_dir + 'analyses_results/'
savedir = ngroups_dir + 'analyses_results/'

data = loadmat(datadir + 'GSBS_obj.mat')
bounds = data['bounds']
binbounds=bounds>0

# Step 1: Select SL 1874 and 2466 from all 5204 SLs
selected_sls = binbounds[[1874, 2466], :, :]

# Step 2: Count the number of TRUEs (boundaries) for each of the 34 groups for both SLs
boundary_counts = np.sum(selected_sls, axis=2)

# Step 3: boundaries to states
state_counts = boundary_counts + 1

# Step 4: Plot
plt.figure(figsize=(12, 10))
plt.rcParams['font.size'] = 18

# Create x-axis values (group numbers)
x_values = np.arange(groups)

# Create scatter plot
plt.scatter(x_values, state_counts[0], color='orange', label='SOG')
plt.scatter(x_values, state_counts[1], color='red', label='vmPFC')

# Set x-axis to show group numbers
plt.xticks([0, 5, 10, 15, 20, 25, 30])
plt.xlabel('Age group')
plt.ylabel('Number of states')

# Add legend
plt.legend(loc='upper right', fontsize=18,
          prop=font_manager.FontProperties(weight='bold'))
for spine in plt.gca().spines.values():
   spine.set_linewidth(2)
plt.tight_layout()

# Save the plot
name = 'group' + str(groups) + '_SLs_nr_boundaries_age_scatter'
plt.savefig(savedir + name + '.png')
plt.close()



# Step 4: Line Plot
plt.figure()
plt.rcParams['font.size'] = 18

# Transpose so groups are on x-axis and we have 2 lines (one for each SL)
data_transposed = np.transpose(state_counts)
bp = seaborn.lineplot(data=data_transposed)

# Set line styles and colors manually
lines = bp.get_lines()
lines[0].set_color('orange')
lines[0].set_linestyle('--')  # dashed
lines[1].set_color('red')
lines[1].set_linestyle(':')   # dotted (uneven dashed)

# Set x-axis to show group numbers
bp.set_xticks([0, 5, 10, 15, 20, 25, 30])
bp.set_xlabel('Age group')
bp.set_ylabel('Number of states')

# Update legend with SL labels
line_labels = ['SOG', 'vmPFC']
plt.legend(lines, line_labels, loc='lower left')

plt.tight_layout()

# Save the plot
name = 'group' + str(groups) + '_SLs_nr_boundaries_age_line'
plt.savefig(savedir + name + '.pdf')
plt.close()


# Get more details on decline in number of states between youngest and oldest group
# Get state counts for all 5204 SLs (assuming you have the full binbounds array)
# Not selected for significant age effects!
all_boundary_counts = np.sum(binbounds, axis=2)  # Shape: (5204, 34)
all_state_counts = all_boundary_counts + 1  # Convert boundaries to states

# Get difference between oldest group (33) and youngest group (0) for each SL
state_differences = all_state_counts[:, 0] - all_state_counts[:, 33]  # Shape: (5204,)

# Get range and median of differences
diff_range = np.max(state_differences) - np.min(state_differences)
diff_median = np.median(state_differences)

print(f"Range of state differences across SLs: {diff_range}")
print(f"Median state difference: {diff_median}")
print(f"Min difference: {np.min(state_differences)}")
print(f"Max difference: {np.max(state_differences)}")

# Create a histogram/distribution plot of the state differences
plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 18

# Create histogram
plt.hist(state_differences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

# Add vertical lines for median and mean
plt.axvline(diff_median, color='red', linestyle='--', linewidth=2, label=f'Median: {diff_median:.2f}')
plt.axvline(np.mean(state_differences), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(state_differences):.2f}')

plt.xlabel('State difference (oldest - youngest group)')
plt.ylabel('Frequency')
plt.title('Distribution of State Differences Across SLs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig(savedir + 'state_differences_distribution.pdf')
plt.show()