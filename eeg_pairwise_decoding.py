#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Pairwise decoding of eeg data
# =============================================================================
def dec():
	"""Pairwise decoding of test data.

	Parameters
	----------
	prepr_eeg_dir : str
		  Preprocessed EEG data directory (default is "").
	mvnn_dir : str
		  MVNN covariance matrix directory (default is "").
	freq : int (optional)
	       Downsampling frequency (default is 1000).
	out_dir : str
		  Pairwise decoding output directory (default is "").

	Returns
	-------
	Pairwise decoding, across time, of the EEG data.

	"""

	import argparse
	import os
	import numpy as np
	from tqdm import tqdm
	from sklearn.model_selection import LeaveOneOut
	from sklearn.svm import SVC

	np.random.seed()


### Input arguments ###
	parser = argparse.ArgumentParser()

	parser.add_argument("--prepr_eeg_dir", default="",type=str, \
					help="preprocessed eeg data directory")
	parser.add_argument("--mvnn_dir", default="",type=str, \
					help="MVNN covariance matrix directory")
	parser.add_argument("--freq", default=1000,type=int, \
					help="downsampling frequency")
	parser.add_argument("--out_dir", default="",type=str, \
					help="Pairwise decoding output directory")

	args = parser.parse_args()


### EEG data image conditions directories ###
	eeg_data = np.load(args.prepr_eeg_dir, allow_pickle=True).item()

	data = eeg_data["eeg_data"]
	events = eeg_data["events"]
	time = eeg_data["time"]
	channels = eeg_data["channels"]
	
	del eeg_data


### Whitening using MVNN ###
	mvnn_dict = np.load(args.mvnn_dir, allow_pickle=True).item()

	sigma_inv = mvnn_dict["sigma_inv"]

	# Correcting the data with the inverse of sigma
	data = (data.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)


### Other parameters ###
	# Total conditions
	tot_cond = len(np.unique(events[:,2]))
	
	# Condition repetitions
	cond_rep = sum(events[:,2] == 1)



### Target vector ###
	target = np.zeros(cond_rep*2)
	target[cond_rep:cond_rep*2] = 1
	count = 1


### Performing the pairwise decoding ###
	diss = np.zeros((tot_cond, tot_cond, len(time)))

	msg = ("Pairwise decoding")

	for i1 in tqdm(range(tot_cond), desc=msg, position=0): # for total conditions

		for i2 in range(tot_cond): # for total conditions

			if i1 < i2:

				count += 1

				# Performing the analysis independently for each time-point
				for t in range(len(time)): # time

					# Condition 1
					idx_1 = events[:,2] == i1+1
					cond_1 = data[idx_1,:,t]

					# Condition 2
					idx_2 = events[:,2] == i2+1
					cond_2 = data[idx_2,:,t]

					# Appending the data
					pair_data = np.append(cond_1, cond_2, 0)

					# Establishing the partitioning scheme
					loo = LeaveOneOut()
					loo.get_n_splits(pair_data)

					score = np.zeros((loo.get_n_splits(pair_data)), dtype=bool)
					idx = 0

					for train_idx, test_idx in loo.split(pair_data): # folds

						# Defining the train/test partitions
						X_train, X_test = pair_data[train_idx], pair_data[test_idx]
						Y_train, Y_test = target[train_idx], target[test_idx]

						# Defining the classifier
						dec_svm = SVC(kernel="linear")

						# Training the classifier
						dec_svm.fit(X_train, Y_train)

						# Testing the classifier
						Y_pred = dec_svm.predict(X_test)
						score[idx] = Y_pred == Y_test;
						idx += 1

					# Storing the results
					acc = sum(score) / loo.get_n_splits(pair_data)
					diss[i1,i2,t] = acc
					diss[i2,i1,t] = acc


### Storing the results into a dictionary ###
	decoding = {
	"diss": diss,
	"channels": channels,
	"time": time
	}


### Saving the results ###
	np.save(args.out_dir, decoding)


### ###
if __name__ == "__main__":
	dec()


