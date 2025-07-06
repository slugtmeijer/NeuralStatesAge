import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

basedir = '/home/lingee/wrkgrp/Selma/CamCAN_movie/highpass_filtered_intercept2/34groups/'
datadir = basedir + 'GSBS_results/searchlights/'
savedir = basedir + 'analyses_results/'


def plot_time_correlation_boundaries(ax, data, GSBS=None, nstates=None):
    # ax: where it is plotted
    # data: 2D matrix, time x voxels
    # GSBS (opt): GSBS object
    # Compute corrcoef
    corr = np.corrcoef(data)
    # Plot the matrix
    ax.imshow(corr, interpolation='none')
    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Time')
    # Plot the boundaries
    if GSBS is not None:
        strengths = GSBS.get_strengths(nstates)
        bounds = np.where(strengths > 0)[0]
        # Add start and end
        bounds = np.insert(bounds, 0, 0)
        bounds = np.append(bounds, len(strengths))
        for i in range(len(bounds) - 1):
            rect = patches.Rectangle(
                (bounds[i], bounds[i]),
                bounds[i + 1] - bounds[i],
                bounds[i + 1] - bounds[i],
                linewidth=1, edgecolor='w', facecolor='none'
            )
            ax.add_patch(rect)


# Define searchlights and age groups
searchlights = ['SL692', 'SL2463']  # 692=STS 1874=SOG 2463=SFG 2466=vmPFC
searchlight_names = {'SL692': 'STS', 'SL1874': 'SOG', 'SL2463': 'SFG', 'SL2466': 'vmPFC'}
age_groups = ['young', 'middle', 'old']
group_dirs = ['GR0', 'GR16', 'GR33']

# Create subplots with horizontal orientation: 2 rows (searchlights), 3 columns (age groups)
f, ax = plt.subplots(2, 3, figsize=(12, 7))

# Column titles for each age group
col_titles = age_groups
# Row titles for each searchlight (using anatomical names)
row_titles = [searchlight_names[sl] for sl in searchlights]

# Generate filenames and plot data
for sl_idx, searchlight in enumerate(searchlights):
    for age_idx, (age_group, group_dir) in enumerate(zip(age_groups, group_dirs)):
        filename = f"{datadir}{group_dir}/GSBS_{group_dir}_stride2_radius3_minvox15_{searchlight}.npy"

        GSBS_obj = np.load(filename, allow_pickle=True).item()
        plot_time_correlation_boundaries(ax=ax[sl_idx, age_idx], data=GSBS_obj.x, GSBS=GSBS_obj)
        # plot_time_correlation_boundaries(ax=ax[sl_idx, age_idx], data=GSBS_obj.x, GSBS=None)

        # Only set column titles on the top row
        if sl_idx == 0:
            ax[sl_idx, age_idx].set_title(col_titles[age_idx], fontsize=14, fontweight='bold')

        # Rotate x-axis labels and adjust their position
        ax[sl_idx, age_idx].tick_params(axis='x', rotation=45, labelsize=8)
        ax[sl_idx, age_idx].xaxis.set_label_coords(0.5, -0.2)

# Set row titles (searchlights) on the left
for i, row_title in enumerate(row_titles):
    f.text(0.08, 0.75 - i * 0.35, row_title, ha='right', va='center', fontsize=14, fontweight='bold',
           rotation='vertical')

# Set overall title
# plt.suptitle('Time Correlation', fontsize=20, y=0.95)

# Adjust layout
# Fine-tune subplot spacing
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.9, hspace=0.1, wspace=0.02)

# Create filename using searchlight anatomical names
sl_names = '_'.join([searchlight_names[sl] for sl in searchlights])
plt.savefig(f"{savedir}time_correlations_SLs_{sl_names}.png", bbox_inches='tight', dpi=300)

# Display the plot
plt.show()
print