import numpy as np
from statesegmentation import GSBS
from hrf_estimation import hrf
from scipy.stats import zscore
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/lingee/wrkgrp/Selma/scripts/Noise_simulation/')
from plot_time_correlations import plot_time_correlation_boundaries


def simulate_noise(group, SL, rep, noiseperc, noiseBOLD, noiseBOLD_name, peak_delay, peak_disp, TR, loaddir, savedir):
    # Load data
    gs = np.load(
        loaddir + 'GR' + str(group) + '/GSBS_GR' + str(group) + '_stride2_radius3_minvox15_SL' + str(SL) + '.npy',
        allow_pickle=True).item()
    data = gs.x
    ntime, nvox = np.shape(data)
    stdevs = np.std(data, 0)

    nstates = np.zeros((len(noiseperc), len(noiseBOLD)))

    for nb in noiseBOLD:
        name = 'group' + str(group) + ' SL' + str(SL) + noiseBOLD_name[nb] + ' noise'

        if rep == 0:
            f, ax = plt.subplots(1, len(noiseperc) + 1, figsize=(15, 3))
            plot_time_correlation_boundaries(ax=ax[0], data=data)
            ax[0].set_title('original')

            # Set overall title higher
            plt.suptitle(name, fontsize=20, y=0.9)  # Adjust the y position to move it higher

        for idx_nperc, nperc in enumerate(noiseperc):

            print([idx_nperc, nb, rep])

            # Add noise
            noise = np.random.randn(ntime, nvox)
            noise = zscore(noise, axis=0)
            scalednoise = noise * np.repeat(np.expand_dims(stdevs * nperc, 0), ntime, axis=0)

            if nb == 1:
                spmhrf = hrf.spm_hrf_compat(np.arange(0, 30, TR), peak_delay=peak_delay, peak_disp=peak_disp)
                BOLDnoise = np.zeros(np.shape(noise))
                for n in range(0, nvox):
                    BOLDnoise[:, n] = np.convolve(noise[:, n], spmhrf)[-192:]
                BOLDnoise = zscore(BOLDnoise, axis=0)
                scalednoise = BOLDnoise * np.repeat(np.expand_dims(stdevs * nperc, 0), ntime, axis=0)

            newdata = data + scalednoise

            # Run GSBS
            GSBS_states = GSBS(x=newdata, kmax=int(ntime * 0.5), finetune=1, statewise_detection=True)
            GSBS_states.fit()

            nstates[idx_nperc, nb] = GSBS_states.nstates

            if rep == 0:
                plot_time_correlation_boundaries(ax=ax[idx_nperc + 1], data=newdata)
                ax[idx_nperc + 1].set_title('noise ' + str(nperc * 100) + ' %')

        # Adjust layout to avoid overlap and make sure labels are visible
        if rep == 0:
            plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.1, right=0.9, top=0.7, bottom=0.2)

            # Save the figure
            plt.savefig(savedir + name + '.png')

    # Save the nstates data
    np.save(savedir + name + str(rep), nstates)

    return nstates


def simulate_time_variability(group, SL, loaddir, savedir):
    # Load data
    gs = np.load(
        loaddir + 'GR' + str(group) + '/GSBS_GR' + str(group) + '_stride2_radius3_minvox15_SL' + str(SL) + '.npy',
        allow_pickle=True).item()
    data = gs.x
    ntime, nvox = np.shape(data)

    nstates = np.zeros((3))
    name = 'group' + str(group) + ' SL' + str(SL) + ' offset '

    # Create subplots
    f, ax = plt.subplots(1, 3, figsize=(15, 3))

    # Move the overall title higher
    plt.suptitle(name, fontsize=20, y=0.9)

    for offset in range(0, 3):
        # Apply different offsets
        if offset == 0:
            newdata = data[2:ntime - 2, :]
        elif offset == 1:
            newdata = data[1:ntime - 3, :] + data[2:ntime - 2, :]
        elif offset == 2:
            newdata = data[0:ntime - 4, :] + data[2:ntime - 2, :]

        # Plot data for the current offset
        plot_time_correlation_boundaries(ax=ax[offset], data=newdata)
        ax[offset].set_title('offset ' + str(offset))

        # Run GSBS on the modified data
        GSBS_states = GSBS(x=newdata, kmax=int(ntime * 0.5), finetune=1, statewise_detection=True)
        GSBS_states.fit()

        nstates[offset] = GSBS_states.nstates

    # Adjust layout to avoid overlap of titles, labels, etc.
    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.1, right=0.9, top=0.7, bottom=0.2)

    # Save the figure
    plt.savefig(savedir + name + '.png')

    # Save the nstates data
    np.save(savedir + name, nstates)

    return nstates
