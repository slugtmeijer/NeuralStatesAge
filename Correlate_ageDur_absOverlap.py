import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from scipy.stats import spearmanr

# Define the directory
basedir = '/home/sellug/wrkgrp/Selma/CamCAN_movie/'
vector1_dir = basedir + 'highpass_filtered_intercept2/34groups/analyses_results/'
vector2_dir = vector1_dir + 'event_boundaries_binary_1swindow/'

# Load the saved variables
age_dur = np.load(os.path.join(vector1_dir, "age_dur_vector.npy"))
age_iss = np.load(os.path.join(vector1_dir, "age_iss_vector.npy"))
abs_overlap = np.load(os.path.join(vector2_dir, "abs_overlap_vector.npy"))
age_strength = np.load(os.path.join(basedir, "highpass_filtered_intercept2/1groups/analyses_results/age_strength_vector.npy"))

# check distribution
# plt.figure(figsize=(12, 5))
#
# plt.subplot(1, 2, 1)
# plt.hist(age_iss, bins=20, color='skyblue', edgecolor='black')
# plt.title("Distribution of age_iss")
#
# plt.subplot(1, 2, 2)
# plt.hist(abs_overlap, bins=20, color='salmon', edgecolor='black')
# plt.title("Distribution of abs_overlap")
#
# plt.show()

# Spearman correlations
correlation, p_value = spearmanr(age_dur, abs_overlap)
print("Correlation age_dur x overlap:", correlation)
print("P-value:", p_value)

correlation, p_value = spearmanr(age_iss, age_dur)
print("Correlation age_iss x age_dur:", correlation)
print("P-value:", p_value)

correlation, p_value = spearmanr(age_dur, age_strength)
print("Correlation age_dur x age_strength:", correlation)
print("P-value:", p_value)

# Plot
slope, intercept, _, _, _ = linregress(age_dur, abs_overlap)

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(age_dur, abs_overlap, color='skyblue', edgecolor='black', alpha=0.7, label="Data points")

# Add regression line
plt.plot(age_dur, slope * age_dur + intercept, color='red', label=f"Regression line (slope={slope:.2f})")

# Annotate with correlation and p-value
plt.title(f"Correlation between age_dur and abs_overlap\nCorrelation: {correlation:.2f}, P-value: {p_value:.2e}")
plt.xlabel("age_dur")
plt.ylabel("abs_overlap")
plt.legend()

# Show the plot
plt.show()


# Get average overlap of the 4 searchlights that are also used in single subject GSBS
SL_ind = [692, 2463, 1874, 2466]
# Extract values from abs_overlap at indices specified in SL_ind
values_at_indices = abs_overlap[SL_ind]

print