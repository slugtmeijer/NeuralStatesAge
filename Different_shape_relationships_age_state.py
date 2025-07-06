# =============================================================================
# MODE SELECTION: Choose analysis mode
# =============================================================================
# Mode 1: Categorical age (groups) vs median duration - use with 34 group means
# Mode 2: Continuous age vs boundary strength - use with individual subject data
MODE = 1  # Change this to 1 or 2

from scipy import io
from scipy.io import loadmat
from scipy import stats
import sys
from scipy.optimize import curve_fit

sys.path.append("..")
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
import pingouin as pg
from create_folder import create_folder
import matplotlib.pyplot as plt
from scipy.stats import sem
import warnings

warnings.filterwarnings('ignore')

groups = 34
basedir = '/home/lingee/wrkgrp/Selma/CamCAN_movie/'
ngroups_dir = basedir + 'highpass_filtered_intercept2/' + str(groups) + 'groups/'
datadir = ngroups_dir + 'analyses_results/'
datadir2 = basedir + 'highpass_filtered_intercept2/1groups/GSBS_results/searchlights/'
SL_dir = basedir + 'masks/searchlights/'

# TRs
ntime = 192
# Searchlights
nregs = 5204

# =============================================================================
# LOAD DATA BASED ON MODE
# =============================================================================
if MODE == 1:
    print("MODE 1: Categorical age vs median duration")
    # Load group-level data
    data = loadmat(datadir + 'GSBS_obj.mat')
    median_duration = data['median_duration']

    # Define age as categorical groups
    age = np.arange(groups)
    dependent_var = median_duration
    analysis_name = "age_duration_cat"  # Added '_cat' for categorical

    savedir = ngroups_dir + 'analyses_results/revision/'

elif MODE == 2:
    print("MODE 2: Continuous age vs boundary strength")
    # Load individual subject data
    strengths_individual_all = np.load(datadir2 + 'individual_strengths_per_SL.npy',
                                       allow_pickle=True).item()

    # Get subjects age as continuous variable
    CBU_info = io.loadmat(basedir + 'subinfo_CBU_age_group.mat')
    var = 'info_CBU_age_group'
    CBU_age = CBU_info[var]
    age = CBU_age[:, 2]

    # Process individual strengths data
    print("Processing individual boundary strengths...")
    n_subjects = len(age)
    individual_strengths = np.full([nregs, n_subjects], np.nan)

    for SL_idx, SL in enumerate(tqdm(strengths_individual_all.keys(), desc="Processing strengths")):
        strengths_this_searchlight = strengths_individual_all[SL]
        # Remove timepoints without boundary at the group level
        timepoints_without_boundary = np.where(strengths_this_searchlight[0, :] == 0)[0]
        strengths_this_searchlight = np.delete(strengths_this_searchlight, timepoints_without_boundary, 1)
        # Get the mean boundary strength for this searchlight per subject
        mean_strengths_this_searchlight = np.mean(strengths_this_searchlight, axis=1)
        individual_strengths[SL_idx, :] = mean_strengths_this_searchlight

    dependent_var = individual_strengths
    analysis_name = "age_strength_cont"  # Added '_cont' for continuous

    savedir = basedir + 'highpass_filtered_intercept2/1groups/analyses_results/revision/'

else:
    raise ValueError("MODE must be 1 or 2")


# Add the fitting functions after loading data but before calculations
# Define fitting functions
def linear_func(x, a, b):
    """Linear function for consistent comparison"""
    return a * x + b


def exponential_func(x, a, b, c):
    """Exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def logarithmic_func(x, a, b, c):
    """Logarithmic function: y = a * log(b * x + 1) + c"""
    return a * np.log(b * x + 1) + c


def quadratic_func(x, a, b, c):
    """Quadratic function: y = a * x^2 + b * x + c"""
    return a * x ** 2 + b * x + c


def calculate_r_squared(y_true, y_pred):
    """Calculate R-squared value"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def calculate_aic_bic(y_true, y_pred, n_params):
    """Calculate AIC and BIC"""
    n = len(y_true)
    mse = np.mean((y_true - y_pred) ** 2)

    if mse <= 0:
        return np.inf, np.inf

    log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - n / 2
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n)

    return aic, bic


def fit_relationship(x, y, func, func_name):
    """
    Fit a relationship and return R-squared and p-value
    Returns (r_squared, p_value, success)
    """
    try:
        # Fit the function with appropriate initial guesses
        if func_name == 'linear':
            # Initial guess for linear: slope around 0, intercept around mean
            popt, pcov = curve_fit(func, x, y, p0=[0, np.mean(y)], maxfev=1000)
        elif func_name == 'exponential':
            # Initial guess for exponential
            popt, pcov = curve_fit(func, x, y, p0=[1, 0.1, np.mean(y)], maxfev=1000)
        elif func_name == 'logarithmic':
            # Initial guess for logarithmic
            popt, pcov = curve_fit(func, x, y, p0=[1, 1, np.mean(y)], maxfev=1000)
        elif func_name == 'quadratic':
            # Initial guess for quadratic
            popt, pcov = curve_fit(func, x, y, p0=[0, 1, np.mean(y)], maxfev=1000)
        else:
            raise ValueError(f"Unknown function name: {func_name}")

        # Calculate predicted values
        y_pred = func(x, *popt)

        # Calculate R-squared using your original function
        r_squared = calculate_r_squared(y, y_pred)

        # Calculate p-value using F-test
        n = len(y)
        k = len(popt)  # number of parameters
        if n > k + 1 and r_squared < 1:
            f_stat = (r_squared / (k - 1)) / ((1 - r_squared) / (n - k))
            p_value = 1 - stats.f.cdf(f_stat, k - 1, n - k)
        else:
            p_value = 1.0

        # Calculate AIC
        n_params = len(popt)
        aic, _ = calculate_aic_bic(y, y_pred, n_params)

        return r_squared, p_value, aic, True

    except Exception as e:
        return 0.0, 1.0, np.inf, False


# Initialize arrays for all relationship types
relationships = ['spearman', 'pearson', 'linear', 'exponential', 'logarithmic', 'quadratic']

# Dictionary to store results
results = {}
for rel in relationships:
    results[rel] = {
        'coeff': np.full([nregs], 0).astype(float),
        'pval': np.full([nregs], 0).astype(float),
        'aic': np.full([nregs], np.inf).astype(float)
    }

# Searchlights
stride = 2
radius = 3
min_vox = 15
params_name = 'stride' + str(stride) + '_' + 'radius' + str(radius) + '_minvox' + str(min_vox)

# Searchlight mask
img = nib.load(basedir + 'masks/data_plus_GM_mask.nii')
affine = img.affine

# Get searchlight information
coordinates = np.load(SL_dir + 'SL_voxelCoordinates_' + params_name + '.npy', allow_pickle=True)
searchlights = np.load(SL_dir + 'SL_voxelsIndices_' + params_name + '.npy', allow_pickle=True)

# Correlation between age and dependent variable
print("Calculating correlations and fitting relationships...")

# Calculate all relationships
print("Calculating correlations and fitting relationships...")
for SL in tqdm(range(nregs), desc="Processing searchlights"):
    y_data = dependent_var[SL, :]

    # Spearman correlation (keep for comparison but don't use in final model selection)
    results['spearman']['coeff'][SL], results['spearman']['pval'][SL] = stats.spearmanr(age, y_data)
    results['spearman']['aic'][SL] = np.inf

    # Pearson correlation (keep for comparison but don't use in final model selection)
    results['pearson']['coeff'][SL], results['pearson']['pval'][SL] = stats.pearsonr(age, y_data)
    results['pearson']['aic'][SL] = np.inf

    # Now fit all relationships using curve fitting (including linear)
    # Linear relationship (equivalent to Pearson but using curve fitting)
    r_sq, p_val, aic, success = fit_relationship(age, y_data, linear_func, 'linear')
    results['linear']['coeff'][SL] = r_sq
    results['linear']['pval'][SL] = p_val
    results['linear']['aic'][SL] = aic

    # Exponential relationship
    r_sq, p_val, aic, success = fit_relationship(age, y_data, exponential_func, 'exponential')
    results['exponential']['coeff'][SL] = r_sq
    results['exponential']['pval'][SL] = p_val
    results['exponential']['aic'][SL] = aic

    # Logarithmic relationship
    r_sq, p_val, aic, success = fit_relationship(age, y_data, logarithmic_func, 'logarithmic')
    results['logarithmic']['coeff'][SL] = r_sq
    results['logarithmic']['pval'][SL] = p_val
    results['logarithmic']['aic'][SL] = aic

    # Quadratic relationship
    r_sq, p_val, aic, success = fit_relationship(age, y_data, quadratic_func, 'quadratic')
    results['quadratic']['coeff'][SL] = r_sq
    results['quadratic']['pval'][SL] = p_val
    results['quadratic']['aic'][SL] = aic


print("Applying multiple comparison correction...")

# Apply multiple comparison correction for each relationship type
corrected_results = {}
for rel in relationships:
    print(f"Processing {rel} relationship...")

    # Correct for multiple testing
    reject, pvals_cmt = pg.multicomp(results[rel]['pval'], method='fdr_bh')
    true_indices = [index for index, value in enumerate(reject) if value]
    cmt_pval = [results[rel]['pval'][index] for index in true_indices] if true_indices else []

    corrected_results[rel] = {
        'reject': reject,
        'pvals_cmt': pvals_cmt,
        'true_indices': true_indices,
        'cmt_pval': cmt_pval
    }

print("Start plotting for all relationship types...")

# from searchlights (SL) to voxels for plotting
x_max, y_max, z_max = img.shape

for rel in relationships:
    print(f"Processing {rel} maps...")

    counter = np.zeros((x_max, y_max, z_max))
    coeff_sum = np.zeros((x_max, y_max, z_max))
    pval_sum = np.zeros((x_max, y_max, z_max))

    for SL_idx, voxel_indices in enumerate(tqdm(searchlights, desc=f"Mapping {rel}")):
        for vox in voxel_indices:
            x, y, z = coordinates[vox]
            counter[x, y, z] += 1
            coeff_sum[x, y, z] += results[rel]['coeff'][SL_idx]
            pval_sum[x, y, z] += results[rel]['pval'][SL_idx]

    # Take mean across searchlights that contribute to each voxel
    mean_coeff = np.divide(coeff_sum, counter, out=np.zeros_like(coeff_sum), where=counter != 0)
    mean_pval = np.divide(pval_sum, counter, out=np.ones_like(pval_sum), where=counter != 0)

    # Save uncorrected maps
    map_nifti = nib.Nifti1Image(mean_coeff, affine)
    nib.save(map_nifti, savedir + f'{analysis_name}_{rel}_uncorrected.nii')

    # Create and save corrected maps
    if corrected_results[rel]['cmt_pval']:
        idx_keep = np.where(mean_pval < max(corrected_results[rel]['cmt_pval']))
        mean_coeff_cmt = np.full_like(mean_coeff, np.nan)
        mean_coeff_cmt[idx_keep] = mean_coeff[idx_keep]

        map_nifti = nib.Nifti1Image(mean_coeff_cmt, affine)
        nib.save(map_nifti, savedir + f'{analysis_name}_{rel}_cmt.nii')

        # Calculate min/max R² values for FDR-corrected map (excluding NaN values)
        valid_values = mean_coeff_cmt[~np.isnan(mean_coeff_cmt)]
        min_r2_corrected = np.min(valid_values)
        max_r2_corrected = np.max(valid_values)

        print(f"{rel}: {len(corrected_results[rel]['true_indices'])} significant searchlights after correction")
        print(f"{rel} FDR-corrected R² range: [{min_r2_corrected:.6f}, {max_r2_corrected:.6f}]")
    else:
        print(f"{rel}: No significant results after correction")

# Save summary statistics
print("Saving summary statistics...")
summary_stats = {}
for rel in relationships:
    summary_stats[rel] = {
        'n_significant_uncorrected': np.sum(results[rel]['pval'] < 0.05),
        'n_significant_corrected': len(corrected_results[rel]['true_indices']),
        'mean_coeff': np.mean(results[rel]['coeff']),
        'std_coeff': np.std(results[rel]['coeff']),
        'min_pval': np.min(results[rel]['pval']),
        'max_coeff': np.max(results[rel]['coeff']),
        'min_coeff': np.min(results[rel]['coeff'])
    }

# Print summary
print("\n=== SUMMARY STATISTICS ===")
for rel in relationships:
    stats = summary_stats[rel]
    print(f"\n{rel.upper()} RELATIONSHIP:")
    print(f"  Significant (uncorrected p<0.05): {stats['n_significant_uncorrected']}")
    print(f"  Significant (FDR corrected): {stats['n_significant_corrected']}")
    print(f"  Mean coefficient/R²: {stats['mean_coeff']:.4f}")
    print(f"  Coefficient/R² range: [{stats['min_coeff']:.4f}, {stats['max_coeff']:.4f}]")
    print(f"  Minimum p-value: {stats['min_pval']:.6f}")

# Save results to file
np.savez(savedir + f'{analysis_name}_relationship_analysis_results.npz',
         **{f'{rel}_coeff': results[rel]['coeff'] for rel in relationships},
         **{f'{rel}_pval': results[rel]['pval'] for rel in relationships},
         summary_stats=summary_stats,
         mode=MODE,
         analysis_name=analysis_name)

print(f"\nAnalysis complete! Results saved to {savedir}")
print("Files created:")
for rel in relationships:
    print(f"  - {analysis_name}_{rel}_uncorrected.nii")
    print(f"  - {analysis_name}_{rel}_cmt.nii (if significant results)")
print(f"  - {analysis_name}_relationship_analysis_results.npz (all numerical results)")


# Compare models
# For model comparison, use only the curve-fitted relationships (not correlations)
comparison_relationships = ['linear', 'exponential', 'logarithmic', 'quadratic']

print("Finding best model for each searchlight (CORRECTED)...")
best_models = []
model_performance = {'linear': 0, 'exponential': 0, 'logarithmic': 0, 'quadratic': 0}

for SL in range(nregs):
    # Get R² values for curve-fitted models only
    aic_values = {
        'linear': results['linear']['aic'][SL],
        'exponential': results['exponential']['aic'][SL],
        'logarithmic': results['logarithmic']['aic'][SL],
        'quadratic': results['quadratic']['aic'][SL]
    }

    # Find model with LOWEST AIC (best model)
    best_model = min(aic_values.keys(), key=lambda k: aic_values[k])
    best_models.append(best_model)
    model_performance[best_model] += 1

# Print simple summary
print("\n=== BEST MODEL SUMMARY (AIC) ===")
print("Best fitting relationship (using AIC - lower is better):")
total_searchlights = len(best_models)

for model, count in sorted(model_performance.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_searchlights) * 100
    print(f"{model}: {count}/{total_searchlights} searchlights ({percentage:.1f}%)")

# Find the overall winner
overall_best = max(model_performance.keys(), key=lambda k: model_performance[k])
print(f"\nOVERALL WINNER: {overall_best.upper()}")
print(f"Best fits {model_performance[overall_best]} out of {total_searchlights} searchlights ({(model_performance[overall_best] / total_searchlights) * 100:.1f}%)")

# Create a simple map showing which model is best for each voxel
print("Creating best model map...")

# Map values: 1=linear, 2=exponential, 3=logarithmic, 4=quadratic
model_to_number = {'linear': 1, 'exponential': 2, 'logarithmic': 3, 'quadratic': 4}

best_model_brain_map = np.zeros((x_max, y_max, z_max))
counter = np.zeros((x_max, y_max, z_max))

for SL_idx, voxel_indices in enumerate(tqdm(searchlights, desc="Creating best model map")):
    best_model = best_models[SL_idx]
    model_number = model_to_number[best_model]

    for vox in voxel_indices:
        x, y, z = coordinates[vox]
        best_model_brain_map[x, y, z] += model_number
        counter[x, y, z] += 1

# Take average (most frequent model per voxel)
final_map = np.divide(best_model_brain_map, counter, out=np.zeros_like(best_model_brain_map), where=counter != 0)
final_map = np.round(final_map).astype(np.int16)

# Save the map
map_nifti = nib.Nifti1Image(final_map, affine)
nib.save(map_nifti, savedir + f'{analysis_name}_best_relationship_map.nii')

print(f"\nSaved: best_relationship_map.nii")
print("Map legend: 1=Linear, 2=Exponential, 3=Logarithmic, 4=Quadratic")

# Save simple results
np.savez(savedir + f'{analysis_name}_simple_best_models.npz',
         best_models=best_models,
         model_counts=model_performance,
         overall_best=overall_best)

print(f"\nResults saved to: simple_best_models.npz")


# =============================================================================
# AGE EFFECTS VISUALIZATION (MODE 1)
# =============================================================================

if MODE == 1:
    print("Analyzing Spearman correlation directions...")

    # Get significant searchlights from Spearman correlation
    spearman_significant = corrected_results['spearman']['true_indices']

    if len(spearman_significant) > 0:
        # Get correlation coefficients for significant searchlights
        significant_coeffs = results['spearman']['coeff'][spearman_significant]

        # Count positive and negative correlations
        positive_correlations = np.sum(significant_coeffs > 0)
        negative_correlations = np.sum(significant_coeffs < 0)

        print(f"\n=== SPEARMAN CORRELATION DIRECTION ANALYSIS ===")
        print(f"Total significant searchlights (FDR corrected): {len(spearman_significant)}")
        print(
            f"Positive correlations (age ↑ → duration ↑): {positive_correlations} ({positive_correlations / len(spearman_significant) * 100:.1f}%)")
        print(
            f"Negative correlations (age ↑ → duration ↓): {negative_correlations} ({negative_correlations / len(spearman_significant) * 100:.1f}%)")

        # Save this information
        correlation_direction_stats = {
            'total_significant': len(spearman_significant),
            'positive_correlations': positive_correlations,
            'negative_correlations': negative_correlations,
            'significant_coefficients': significant_coeffs
        }

        np.savez(savedir + f'{analysis_name}_spearman_direction_analysis.npz',
                 **correlation_direction_stats)
    else:
        print("No significant Spearman correlations found for direction analysis")

# =============================================================================
# ENHANCED AGE EFFECTS VISUALIZATION (MODE 1)
# =============================================================================
if MODE == 1:
    print("Creating enhanced age effects visualization for MODE 1...")

    # Get significant searchlights from Spearman correlation
    spearman_significant = corrected_results['spearman']['true_indices']

    if len(spearman_significant) > 0:
        print(f"Found {len(spearman_significant)} significant searchlights for Spearman correlation")

        # Extract data for youngest (group 0) and oldest (group 33)
        youngest_group = 0
        oldest_group = 33

        # Convert from TRs to seconds (1 TR = 2470 ms = 2.47 seconds)
        TR_to_seconds = 2.47

        # Get median duration data for significant searchlights and convert to seconds
        young_data = median_duration[spearman_significant, youngest_group] * TR_to_seconds
        old_data = median_duration[spearman_significant, oldest_group] * TR_to_seconds

        # Calculate means and SEMs
        young_mean = np.mean(young_data)
        old_mean = np.mean(old_data)
        young_sem = sem(young_data)
        old_sem = sem(old_data)

        # Calculate mean difference
        mean_difference = old_mean - young_mean

        # Calculate individual differences for each searchlight
        individual_differences = old_data - young_data

        # Calculate additional statistics on duration differences
        diff_mean = np.mean(individual_differences)
        diff_std = np.std(individual_differences)
        diff_range = np.max(individual_differences) - np.min(individual_differences)

        print(f"Mean duration difference: {diff_mean:.3f} seconds")
        print(f"Standard deviation: {diff_std:.3f} seconds")
        print(f"Range of duration differences: {diff_range:.3f} seconds")
        print(f"Min difference: {np.min(individual_differences):.3f} seconds")
        print(f"Max difference: {np.max(individual_differences):.3f} seconds")

        # Count positive and negative differences
        positive_differences = np.sum(individual_differences > 0)
        negative_differences = np.sum(individual_differences < 0)
        zero_differences = np.sum(individual_differences == 0)

        print(f"\n=== SEARCHLIGHT DIFFERENCE BREAKDOWN ===")
        print(f"Searchlights with POSITIVE difference (Old > Young): {positive_differences}")
        print(f"Searchlights with NEGATIVE difference (Old < Young): {negative_differences}")
        print(f"Searchlights with NO difference (Old = Young): {zero_differences}")
        print(f"Total searchlights: {len(individual_differences)}")
        print(f"Percentage positive: {positive_differences / len(individual_differences) * 100:.1f}%")
        print(f"Percentage negative: {negative_differences / len(individual_differences) * 100:.1f}%")

        # Create the enhanced plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Bar positions with some spacing for jitter
        x_positions = [0, 1]
        group_labels = ['Youngest', 'Oldest']
        means = [young_mean, old_mean]
        sems = [young_sem, old_sem]

        # Create bars with SEM error bars - make them more prominent
        bars = ax.bar(x_positions, means, yerr=sems, capsize=10,
                      color=['lightblue', 'lightcoral'], alpha=0.8,
                      edgecolor='black', linewidth=2, width=0.6)

        # Add jittered individual data points with connecting lines - MADE VERY SUBTLE
        np.random.seed(42)  # For reproducible jitter
        jitter_amount = 0.15  # Amount of horizontal jitter

        # Generate jittered x positions
        x_young_jitter = np.random.normal(0, jitter_amount, len(young_data))
        x_old_jitter = np.random.normal(1, jitter_amount, len(old_data))

        # Clip jitter to stay within reasonable bounds
        x_young_jitter = np.clip(x_young_jitter, -0.3, 0.3)
        x_old_jitter = np.clip(x_old_jitter, 0.7, 1.3)

        for i in range(len(spearman_significant)):
            y_young = young_data[i]
            y_old = old_data[i]
            x_young = x_young_jitter[i]
            x_old = x_old_jitter[i]

            # Plot connecting line - VERY SUBTLE
            ax.plot([x_young, x_old], [y_young, y_old], 'k-', alpha=0.08, linewidth=0.3, zorder=1)

            # Plot individual points - VERY SUBTLE
            ax.scatter([x_young], [y_young], color='darkblue', alpha=0.15, s=8, zorder=2,
                       edgecolors='none')
            ax.scatter([x_old], [y_old], color='darkred', alpha=0.15, s=8, zorder=2,
                       edgecolors='none')

        # Set fixed y-axis maximum and calculate positioning for significance line
        y_max_fixed = 50
        y_data_max = max(np.max(young_data), np.max(old_data))
        y_data_min = min(np.min(young_data), np.min(old_data))

        # Position the significance line strategically within the fixed range
        # Leave space for text above the line and below the y_max
        line_height = y_max_fixed - 8  # Position line 8 units below the top

        # Draw horizontal line between bar centers
        ax.plot([0, 1], [line_height, line_height], 'k-', linewidth=1, zorder=5)

        # Add vertical ticks at ends
        tick_height = 0.5  # Fixed tick height
        ax.plot([0, 0], [line_height - tick_height, line_height + tick_height], 'k-', linewidth=1, zorder=5)
        ax.plot([1, 1], [line_height - tick_height, line_height + tick_height], 'k-', linewidth=1, zorder=5)

        # Add mean difference text
        mid_x = 0.5
        text_height = line_height + 2  # Position text 2 units above the line
        ax.text(mid_x, text_height,
                f'Δ = {mean_difference:.2f} s, ' + r'$\it{p}$' + ' < .001',
                ha='center', va='bottom', fontsize=18, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='black',
                          alpha=0.95, linewidth=1))

        # Add mean value annotations on the bars
        for i, (mean_val, sem_val) in enumerate(zip(means, sems)):
            # Calculate annotation height, ensuring it doesn't exceed the line
            annotation_height = min(mean_val + sem_val + 1, line_height - 2)
            ax.text(x_positions[i], annotation_height,
                    f'{mean_val:.2f}±{sem_val:.2f}',
                    ha='center', va='bottom', fontsize=18, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        # Formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_labels, fontsize=18, fontweight='bold')
        ax.set_ylabel('Median State Duration (seconds)', fontsize=16, fontweight='bold')
        ax.tick_params(axis='y', which='major', labelsize=16)
        ax.set_title(
            f'Age Effects on Median State Duration\n({len(spearman_significant)} Significant Searchlights - Spearman, FDR corrected)',
            fontsize=18, fontweight='bold', pad=25)

        # Set fixed y-limits
        ax.set_ylim(0, y_max_fixed)

        # Make the plot frame more prominent
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(savedir + f'{analysis_name}_enhanced_age_effects_barplot.png', dpi=300, bbox_inches='tight')
        plt.savefig(savedir + f'{analysis_name}_enhanced_age_effects_barplot.pdf', bbox_inches='tight')

        # Print summary statistics
        print(f"\n=== ENHANCED AGE EFFECTS SUMMARY ===")
        print(f"Youngest group (0) - Mean: {young_mean:.3f} sec, SEM: {young_sem:.3f} sec")
        print(f"Oldest group (33) - Mean: {old_mean:.3f} sec, SEM: {old_sem:.3f} sec")
        print(f"Mean difference (Old - Young): {mean_difference:.3f} sec")
        print(f"Number of searchlights: {len(spearman_significant)}")

        # Statistical test on the difference
        from scipy.stats import ttest_rel

        t_stat, p_val = ttest_rel(old_data, young_data)
        print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.6f}")

                # Display of the plot
        plt.tight_layout()
        plt.show()

        # Save the enhanced data used for the plot
        enhanced_plot_data = {
            'young_data_seconds': young_data,
            'old_data_seconds': old_data,
            'young_mean_seconds': young_mean,
            'old_mean_seconds': old_mean,
            'young_sem_seconds': young_sem,
            'old_sem_seconds': old_sem,
            'mean_difference_seconds': mean_difference,
            'individual_differences_seconds': individual_differences,
            'positive_differences_count': positive_differences,
            'negative_differences_count': negative_differences,
            'zero_differences_count': zero_differences,
            'significant_searchlights': spearman_significant,
            't_stat': t_stat,
            'p_val': p_val,
            'TR_to_seconds_conversion': TR_to_seconds,
            'jitter_seed': 42,
            'x_young_jitter': x_young_jitter,
            'x_old_jitter': x_old_jitter
        }

        # Add correlation direction stats if available
        if 'correlation_direction_stats' in locals():
            enhanced_plot_data.update(correlation_direction_stats)

        np.savez(savedir + f'{analysis_name}_enhanced_age_effects_plot_data.npz', **enhanced_plot_data)
        print(f"\nEnhanced plot saved as: {analysis_name}_enhanced_age_effects_barplot.png and .pdf")
        print(f"Enhanced plot data saved as: {analysis_name}_enhanced_age_effects_plot_data.npz")

    else:
        print("No significant searchlights found for Spearman correlation - skipping visualization")

    print("Enhanced visualization complete!\n")

else:
    print("Enhanced age effects visualization is only available for MODE 1 (categorical age vs median duration)")

# =============================================================================
# AGE EFFECTS VISUALIZATION (MODE 2)
# =============================================================================
def plot_example_relationships(n_examples=6):
    """Plot examples of quadratic vs linear fits"""
    # Find searchlights where quadratic wins by a large margin
    aic_differences = results['linear']['aic'] - results['quadratic']['aic']
    strong_quadratic_indices = np.argsort(aic_differences)[-n_examples:]  # Largest differences
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Find global y-axis limits for consistent scaling
    all_y_data = [dependent_var[SL, :] for SL in strong_quadratic_indices]
    y_min = min(np.min(y_data) for y_data in all_y_data)
    y_max = max(np.max(y_data) for y_data in all_y_data)
    y_range = y_max - y_min
    y_margin = y_range * 0.05  # 5% margin
    global_ylim = (y_min - y_margin, y_max + y_margin)

    for i, SL in enumerate(strong_quadratic_indices):
        y_data = dependent_var[SL, :]

        # Fit both models
        popt_lin, _ = curve_fit(linear_func, age, y_data)
        popt_quad, _ = curve_fit(quadratic_func, age, y_data)

        # Generate smooth curves for plotting
        age_smooth = np.linspace(age.min(), age.max(), 100)
        y_lin = linear_func(age_smooth, *popt_lin)
        y_quad = quadratic_func(age_smooth, *popt_quad)

        # Plot
        axes[i].scatter(age, y_data, alpha=0.6, s=20)
        axes[i].plot(age_smooth, y_lin, 'r-', label=f'Linear (AIC={results["linear"]["aic"][SL]:.1f})')
        axes[i].plot(age_smooth, y_quad, 'b-', label=f'Quadratic (AIC={results["quadratic"]["aic"][SL]:.1f})')

        # Set title with larger font
        axes[i].set_title(f'SL {SL}: ΔAIC={aic_differences[SL]:.1f}', fontsize=16, fontweight='bold')

        # Set legend with larger font
        axes[i].legend(loc='lower right', fontsize=12)

        # Set consistent y-axis limits
        axes[i].set_ylim(global_ylim)

        # Set tick label sizes
        axes[i].tick_params(axis='both', which='major', labelsize=12)

        # Only add labels to the top-left subplot (index 0 in a 2x3 grid)
        if i == 0:  # Top-left subplot
            axes[i].set_xlabel('Age', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Boundary Strength', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(savedir + f'{analysis_name}_example_quadratic_fits.png', dpi=300)
    plt.show()


# Run the visualization
if MODE == 2:  # Only for continuous data
    plot_example_relationships()

print