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


# Create subplots with switched orientation: 3 rows, 2 columns
f, ax = plt.subplots(3, 2, figsize=(10, 15))

# Column titles for each searchlight
col_titles = ['SL1874', 'SL2466']  # 692=STS 1874=SOG 2463=SFG 2466=vmPFC
row_titles = ['young', 'middle', 'old']

# Load and plot data for each subplot - reorganized for new layout
filenames = [
    # Row 0 (young): SL1874, SL2466
    (datadir + 'GR0/GSBS_GR0_stride2_radius3_minvox15_SL1874.npy', 'young', 'SL1874'),
    (datadir + 'GR0/GSBS_GR0_stride2_radius3_minvox15_SL2466.npy', 'young', 'SL2466'),
    # Row 1 (middle): SL1874, SL2466
    (datadir + 'GR16/GSBS_GR16_stride2_radius3_minvox15_SL1874.npy', 'middle', 'SL1874'),
    (datadir + 'GR16/GSBS_GR16_stride2_radius3_minvox15_SL2466.npy', 'middle', 'SL2466'),
    # Row 2 (old): SL1874, SL2466
    (datadir + 'GR33/GSBS_GR33_stride2_radius3_minvox15_SL1874.npy', 'old', 'SL1874'),
    (datadir + 'GR33/GSBS_GR33_stride2_radius3_minvox15_SL2466.npy', 'old', 'SL2466')
]

# Plot data
for i, (filename, age_group, searchlight) in enumerate(filenames):
    row, col = divmod(i, 2)  # Now we have 2 columns instead of 3
    GSBS_obj = np.load(filename, allow_pickle=True).item()
    plot_time_correlation_boundaries(ax=ax[row, col], data=GSBS_obj.x, GSBS=GSBS_obj)
    # plot_time_correlation_boundaries(ax=ax[row, col], data=GSBS_obj.x, GSBS=None)

    # Only set column titles on the top row
    if row == 0:
        ax[row, col].set_title(col_titles[col], fontsize=14, fontweight='bold')

    # Rotate x-axis labels and adjust their position
    ax[row, col].tick_params(axis='x', rotation=45, labelsize=8)
    ax[row, col].xaxis.set_label_coords(0.5, -0.2)

# Set row titles (age groups) on the left
for i, row_title in enumerate(row_titles):
    f.text(0.08, 0.83 - i * 0.28, row_title, ha='right', va='center', fontsize=14, fontweight='bold',
           rotation='vertical')

# Set overall title
# plt.suptitle('Time Correlation', fontsize=20, y=0.95)

# Adjust layout
#plt.tight_layout(rect=[0.12, 0.05, 1, 0.95], w_pad=.1)

# Fine-tune subplot spacing
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.9, hspace=0.3, wspace=0.05)

# Adjust bottom margin for x-axis labels
f.subplots_adjust(bottom=0.1)

plt.savefig(savedir + 'time_correlations_SLs_1874_2466.pdf', bbox_inches='tight', dpi=300)

# Display the plot
plt.show()
print