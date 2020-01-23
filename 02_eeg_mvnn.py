#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Multivariate noise normalization covariance matrix
# =============================================================================
def main():
	"""MVNN covriance matrix.

	Parameters
	----------
	prepr_eeg_dir : str
		  Preprocessed EEG data directory (default is "").
	mvnn_dim : str
	      Whether to compute the mvnn covariace matrices
		  for each time point["time"] or for each epoch["epochs"]
		  (default is "epochs").
	freq : int
	      Downsampling frequency (default is 1000).
	out_dir : str
		  MVNN covariance matrix output directory (default is "").

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

	parser.add_argument("--prepr_eeg_dir", default="",type=str, \
					help="preprocessed eeg data directory")
	parser.add_argument('--mvnn_dim', default="time", type=str, \
					help="MVNN dimension")
	parser.add_argument('--freq', default=1000, type=int, \
					help="downsampling frequency")
	parser.add_argument("--out_dir", default="",type=str, \
					help="MVNN matrix output directory")

	args = parser.parse_args()

	sub = args.sub
	mvnn_dim = args.mvnn_dim
	freq = args.freq
	run_where = args.run_where

	print("\n\n\n>>> Multivariate Noise Normalization Matrix <<<")


### Loading the data ###
	eeg_data = np.load(args.prepr_eeg_dir, allow_pickle=True).item()
	
	data = eeg_data["eeg_data"]
	events = eeg_data["events"]
	time = eeg_data["time"]
	channels = eeg_data["channels"]
	
	del eeg_data


### Computing the covariance matrices ###
	n_conditions = len(np.unique(events[:,2]))

	# n_conditions * n_sensors * n_sensors
	sigma_ = np.empty((n_conditions, len(channels), len(channels)))

	for c in range(len(np.unique(events[:,2]))): # for experimental conditions

		# Selecting the condition data
		idx = events[:,2] == c+1
		cond_data = data[idx,:,:]

		# Performing the computation
		if mvnn_dim == "time": # if computing covariace matrices for each time point

			# Computing sigma for each time point, then averaging across time
			sigma_[c] = np.mean([_cov(cond_data[:,:,t], \
					shrinkage='auto') for t in range(len(time))], \
					axis=0)

		elif mvnn_dim == "epochs": # if computing covariace matrices for each time epoch

			# Computing sigma for each epoch, then averaging across epochs
			sigma_[c,:,:] = np.mean([_cov(np.transpose(cond_data[e,:,:]), \
					shrinkage='auto') for e in range(cond_data.shape[0])], \
					axis=0)


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
	np.save(args.out_dir, inv_sigma_dict)


### ###
if __name__ == "__main__":
	main()

