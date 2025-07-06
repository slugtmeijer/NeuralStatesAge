import numpy as np
import sys
sys.path.append('/home/lingee/wrkgrp/Selma/scripts/Noise_simulation/')
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from plot_time_correlations import plot_time_correlation_boundaries
from joblib import Parallel, delayed
import seaborn
from simulation_revis import simulate_noise, simulate_time_variability

# parameters
base_dir = '/home/lingee/wrkgrp/Selma/CamCAN_movie/highpass_filtered_intercept2/34groups/'
loaddir = base_dir + 'GSBS_results/searchlights/'
savedir = base_dir + 'noise_simulation/revision/'
noiseperc = [0.05, 0.5, 1]
noiseBOLD = [0]
noiseBOLD_name = [' nonBOLD', ' BOLD']
peak_delay = 6
peak_disp = 1
TR = 2.47
reps=100
group=0
SLs = [692, 2463, 1874, 2466]
sim_output_noise=[]

#simulate_noise(group, SL, 0, noiseperc, noiseBOLD, noiseBOLD_name, peak_delay, peak_disp, TR, loaddir, savedir)
for iSL, SL in enumerate(SLs):
    sim_output_noise.append(Parallel(n_jobs=10)(delayed(simulate_noise)(group, SL, rep, noiseperc, noiseBOLD, noiseBOLD_name, peak_delay, peak_disp, TR, loaddir, savedir) for rep in range(0, reps)))

sim_output_offset = Parallel(n_jobs=4)(delayed(simulate_time_variability)(group, SL, loaddir, savedir) for SL in SLs)

# plot results noise simulation
for iSL, SL in enumerate(SLs):
    results = np.squeeze(np.array(sim_output_noise[iSL]))
    plt.figure()

    # Create the plots
    bp = seaborn.boxplot(data=results)
    bp = seaborn.swarmplot(data=results, color=".25")

    # Set tick labels
    bp.set_xticklabels([str(noiseperc[0] * 100) + '%', str(noiseperc[1] * 100) + '%', str(noiseperc[2] * 100) + '%'])

    # Set font size for ticks to 18
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Remove axis labels (set to empty strings)
    plt.xlabel('')
    plt.ylabel('')

    # Create filename with SL included
    name = 'group' + str(group) + '_SL' + str(SL) + noiseBOLD_name[0] + '_noise'
    plt.savefig(savedir + name + '_nstates.png')

#plot results offset simulation
results = np.squeeze(np.mean(np.array(sim_output_offset),2))
plt.figure()
plt.rcParams['font.size']=18
bp = seaborn.lineplot(data=np.transpose(results))
bp.set_xticks(range(0,3))
#handles, labels=plt.gca().get_legend_handles_labels()
# Update legend with specific line names
handles, _ = plt.gca().get_legend_handles_labels()  # Get current handles
line_labels = ['STS', 'SOG', 'SFG', 'vmPFC']  # Custom line labels
plt.legend(handles, line_labels, loc='lower left')
#plt.legend(handles, SLs, loc='lower left')
plt.xlabel('offset')
plt.ylabel('number of states')
plt.tight_layout()
name = 'group' + str(group) + ' SLs' + ' offset_'
plt.savefig(savedir + name + 'nstates_c.png')

print