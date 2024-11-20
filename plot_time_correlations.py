import numpy as np

import matplotlib.patches as patches

def plot_time_correlation_boundaries(ax, data, GSBS=None, nstates=None):
    # ax: where it is plotted
    # data: 2D matrix, time x voxels
    # GSBS (opt): GSBS object

    # Compute corrcoef
    corr = np.corrcoef(data)

    # Plot the matrix
    ax.imshow(corr, interpolation='none')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Timepoint')

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
