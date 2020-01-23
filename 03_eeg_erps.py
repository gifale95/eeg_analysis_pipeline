#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Getting ERPs data
# =============================================================================
def getErps():
	"""ERPs of EEG data.
	
	Parameters
	----------
	prepr_eeg_dir : str
		  Preprocessed EEG data directory (default is "").
	freq : int
		  Downsampling frequency (default is 1000).
	out_dir : str
		  ERPs output directory (default is "").

	Returns
	-------
	ERPs of the selected data.

	"""

	import argparse
	import os
	import numpy as np


### Input arguments ###
	parser = argparse.ArgumentParser()

	parser.add_argument("--prepr_eeg_dir", default="",type=str, \
					help="preprocessed eeg data directory")
	parser.add_argument("--freq", default=1000,type=int, \
					help="downsampling frequency")
	parser.add_argument("--out_dir", default="",type=str, \
					help="ERPs output directory")

	args = parser.parse_args()

	print("\n\n\n>>> Getting ERPs <<<")


### Loading the data ###
	eeg_data = np.load(args.prepr_eeg_dir, allow_pickle=True).item()
	
	erps_data = eeg_data["eeg_data"]
	time = eeg_data["time"]
	channels = eeg_data["channels"]

	del eeg_data


### Collapsing across trials ###
	eeg_data = np.mean(eeg_data, axis=0)


### Putting the data into a dictionary ###
	ERPs_dict = {
	"erps_data": erps_data,
	"channels": channels,
	"time": time
	}


### Saving the results ###
	np.save(out_dir, ERPs_dict)


### ###
if __name__ == "__main__":
	getErps()

