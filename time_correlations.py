import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/highpass_filtered_intercept2/34groups/'
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
    ax.set_xlabel('Time')
    ax.set_ylabel('Time')

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


# Create subplots with adjusted layout
f, ax = plt.subplots(2, 3, figsize=(15, 20))

# Titles for each row
row_titles = ['SL1874', 'SL2466']

# Load and plot data for each subplot
filenames = [
    (datadir + 'GR0/GSBS_GR0_stride2_radius3_minvox15_SL1874.npy', 'youngest'),
    (datadir + 'GR17/GSBS_GR17_stride2_radius3_minvox15_SL1874.npy', 'middle'),
    (datadir + 'GR33/GSBS_GR33_stride2_radius3_minvox15_SL1874.npy', 'oldest'),
    (datadir + 'GR0/GSBS_GR0_stride2_radius3_minvox15_SL2466.npy', 'youngest'),
    (datadir + 'GR17/GSBS_GR17_stride2_radius3_minvox15_SL2466.npy', 'middle'),
    (datadir + 'GR33/GSBS_GR33_stride2_radius3_minvox15_SL2466.npy', 'oldest')
]

# Plot data
for i, (filename, title) in enumerate(filenames):
    row, col = divmod(i, 3)
    GSBS_obj = np.load(filename, allow_pickle=True).item()
    plot_time_correlation_boundaries(ax=ax[row, col], data=GSBS_obj.x, GSBS=GSBS_obj)
    #ax[row, col].set_title(title)

    # Rotate x-axis labels and adjust their position
    ax[row, col].tick_params(axis='x', rotation=45, labelsize=8)
    ax[row, col].xaxis.set_label_coords(0.5, -0.2)

# Set row titles
# for i, row_title in enumerate(row_titles):
#     f.text(0.08, 0.72 - i * 0.43, row_title, ha='right', va='center', fontsize=16, fontweight='bold',
#            rotation='vertical')

# Set overall title
# plt.suptitle('Time Correlation', fontsize=20, y=0.95)

# Adjust layout
plt.tight_layout(rect=[0.1, 0.05, 1, 0.93])

# Fine-tune subplot spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Adjust bottom margin for x-axis labels
f.subplots_adjust(bottom=0.1)

#plt.savefig(savedir + 'time_correlations_SLs_1874_2466.png', bbox_inches='tight', dpi=300)

# Display the plot
plt.show()