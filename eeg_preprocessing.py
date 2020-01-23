#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Preprocessing EEG data
# =============================================================================
def preprEeg():
	"""Preprocessing of EEG data.

	Parameters
	----------
	eeg_data_dir : str
		  Raw EEG data directory (default is "").
	epoch_tmin : float
		  Epochs min time (default is -0.2).
	epoch_tmax : float
		  Epochs max time (default is 0.8).
	freq : int
		  Downsampling frequency (default is 1000).
	out_dir : str
		  Preprocessed EEG output directory (default is "").

	Returns
	-------
	Preprocessed EEG data.

	"""

	import argparse
	import mne
	import os
	import numpy as np
	from scipy import io


### Input arguments ###
	parser = argparse.ArgumentParser()

	parser.add_argument("--eeg_data_dir", default="",type=str, \
					help="eeg data directory")
	parser.add_argument("--epoch_tmin", default=-0.2,type=float, \
					help="epochs min time")
	parser.add_argument("--epoch_tmax", default=0.6,type=float, \
					help="epochs max time")
	parser.add_argument("--freq", default=1000,type=int, \
					help="downsampling frequency")
	parser.add_argument("--out_dir", default="",type=str, \
					help="preprocessed eeg output directory")

	args = parser.parse_args()

	print("\n\n\n>>> Preprocessing data <<<")


### Reading the data ###
	raw = mne.io.read_raw_brainvision(args.eeg_data_dir, preload=True)


### Finding the data events ###
	events, events_id = mne.events_from_annotations(raw)
	events = events[1:,:]


### Epoching the sequences & baseline correcting ###
	epochs = mne.Epochs(raw, events, tmin=args.epoch_tmin, tmax=args.epoch_tmax, \
								baseline=(None,0), preload=True)

	del raw


### Downsampling the sequence data ###
	epochs.resample(args.freq)


### Putting the sequence data into a dictionary ###
	eeg_data = {
			"eeg_data": epochs.get_data(),
			"events": events,
			"channels": epochs.ch_names,
			"time": epochs.times
	}

	del epochs


### Saving the sequence data ###
	np.save(out_dir, eeg_data)


### ###
if __name__ == "__main__":
	preprEeg()


