#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Multivariate noise normalization covariance matrices
# =============================================================================
def main():
	"""MVNN covriance matrices.

	Parameters
	----------
	sub : int
	      Subject number (default is 1).
	mvnn_dim : str
	      Whether to compute the mvnn covariace matrices
		  for each time point["time"] or for each epoch["epochs"]
		  (default is "epochs").
	freq : int
	      Downsampling frequency (default is 200).
	run_where : str (optional)
	       To run the computations on "local" or "hpc" (default is "local").

	Returns
	-------
	The inverse of the covariance matrices, calculated for each time-point
	or epoch of each condition, and then averaged.

	"""

	import argparse
	import os
	import numpy as np
	from sklearn.discriminant_analysis import _cov
	import scipy


### Input arguments ###
	parser = argparse.ArgumentParser()

	parser.add_argument('--sub', default=1, type=int, help="subject_n")
	parser.add_argument('--mvnn_dim', default="time", type=str, \
					help="mvnn_dimension")
	parser.add_argument('--freq', default=200, type=int, \
					help="downsampling frequency")
	parser.add_argument("--run_where", type=str, default="local", \
						help="To run on 'hpc' or on 'local'")

	args = parser.parse_args()

	sub = args.sub
	mvnn_dim = args.mvnn_dim
	freq = args.freq
	run_where = args.run_where

	print("\n\n\n>>> Mvnn, %dhz, sub %s <<<" % (freq, sub))


### EEG data image conditions directories ###
	# Working directory
	if run_where == "local": # if running on local

		workDir = open("/home/ale/workDir_py.txt")
		workDir = workDir.read()
		workDir = workDir[:-1]

	else: # if running on hpc
		
		workDir = "/scratch/alegifford95/studies/"
	
	# Study directory
	studyDir = "buzz_study/collected_data/"

	# Subject directory
	subDir = "s" + str(sub) + "/eeg/preprocessed/"
	
	# File name
	fileDir = "preprocessed_" + str(freq) + "hz_sub_" + format(sub, "03") \
					+ ".npy"


### Loading the data ###
	eeg_data = np.load(os.path.join(workDir, studyDir, subDir, fileDir), \
						allow_pickle=True).item()

	time = eeg_data["time"]
	channels = eeg_data["channels"]
	img_category = eeg_data["img_category"]
	img_fragment = eeg_data["img_fragment"]
	img_filtering = eeg_data["img_filtering"]
	eeg_data = eeg_data["eeg_data"]


### Computing the covariance matrices ###
	n_conditions = len(np.unique(img_category)) * len(np.unique(img_fragment)) * len(np.unique(img_filtering))

	# n_conditions * n_sensors * n_sensors
	sigma_ = np.empty((n_conditions, len(channels), len(channels)))
	
	count = 0

	for c in range(len(np.unique(img_category))): # for image category

		for fr in range(len(np.unique(img_fragment))): # for image fragment

			for fi in range(len(np.unique(img_filtering))): # for image filtering

				# Selecting the data
				idx_cat = img_category == c+1
				idx_fra = img_fragment == fr+1
				idx_fil = img_filtering == fi+1
				idx = idx_cat & idx_fra & idx_fil
				
				cond_data = eeg_data[idx,:,:]

				# Performing the computation
				if mvnn_dim == "time": # if computing covariace matrices for each time point

					# Computing sigma for each time point, then averaging across time
					sigma_[count] = np.mean([_cov(cond_data[:,:,t], \
							shrinkage='auto') for t in range(len(time))], \
							axis=0)
					count += 1

				elif mvnn_dim == "epochs": # if computing covariace matrices for each time epoch

					# Computing sigma for each epoch, then averaging across epochs
					sigma_[count,:,:] = np.mean([_cov(np.transpose(cond_data[e,:,:]), \
							shrinkage='auto') for e in range(cond_data.shape[0])], \
							axis=0)
					count += 1


### Averaging sigma across conditions ###
	sigma = sigma_.mean(axis=0)


### Computing the inverse of sigma ###
	sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)


### Storing the data into a dictionary ###
	inv_sigma_dict = {
	"sigma_inv": sigma_inv,
	"time": time,
	"channels": channels
	}


### Saving the covariance matrices ###
	saveDir = "buzz_study/results/"
	subDir = "s" + str(sub) + "/mvnn/" + str(freq) + "hz/"
	fileDir = "mvnn_cov_mat_" + str(freq) + "hz_sub_" + format(sub, "03")

	# Creating the directory if not existing
	if os.path.isdir(os.path.join(workDir, saveDir, subDir)) == False: # if not a directory
		os.makedirs(os.path.join(workDir, saveDir, subDir))

	np.save(os.path.join(workDir, saveDir, subDir, fileDir), inv_sigma_dict)


### ###
if __name__ == "__main__":
	main()

