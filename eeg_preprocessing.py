#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Preprocessing EEG data
# =============================================================================
def preprEeg():
	"""Preprocessing of EEG_encoding_paradigm_2 data.

	Parameters
	----------
	sub : int
	      Subject number.
	freq : int
		  Downsampling frequency (default is 1000).
	run_where : str (optional)
	       To run the computations on "local" or "hpc" (default is "local").

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

	parser.add_argument("--sub", default=1, type=int, help="subject_n")
	parser.add_argument("--freq", default=1000,type=int, \
					help="downsampling frequency")
	parser.add_argument("--run_where", type=str, default="local", \
						help="To run on 'hpc' or on 'local'")

	args = parser.parse_args()

	sub = args.sub
	freq = args.freq
	run_where = args.run_where


### Data directories ###
	# Working directory
	if run_where == "local": # if running on local

		workDir = open("/home/ale/workDir_py.txt")
		workDir = workDir.read()
		workDir = workDir[:-1]

	else: # if running on hpc
		
		workDir = "/scratch/alegifford95/studies/"

	# Study directory
	studyDir = "buzz_study/collected_data/"

	# Behavioral data directory
	behavDir = "s" + str(sub) + "/behavioral/" + format(sub, "03") + ".mat"

	# EEG data directory
	eegDir = "s" + str(sub) + "/eeg/raw/Buzz_s" + format(sub, "04") + ".vhdr"


### Extracting beahavioral data ###
	# Loading the behavioral data
	behav = io.loadmat(os.path.join(workDir, studyDir, behavDir))
	behav = behav["data"]

	# Beahvioral image info
	img_cat = behav[0][0][1]["scene"][0]
	img_cat = np.asarray(img_cat, dtype=int)

	img_frag = behav[0][0][1]["fragment"][0]
	img_frag = np.asarray(img_frag, dtype=int)

	img_filt = behav[0][0][1]["filter"][0]
	img_filt = np.asarray(img_filt, dtype=int)


### Checking the accuracy ###
	# Extracting the data
	response = behav[0][0][1]["response"][0]
	response = np.asarray(response, dtype=int)
	correctness = behav[0][0][1]["correctness"][0]
	correctness = np.asarray(correctness, dtype=int)

	# Calculating the accuracy
	accuracy = sum(correctness) / len(correctness) * 100

	# Establishing the type of errors
	idx_incorrect = np.where(correctness == 0)
	false_positives = sum(response[idx_incorrect] == 1)
	false_negatives = sum(response[idx_incorrect] == 0)
	no_response = sum(response[idx_incorrect] == 2)

	# Printing the behavioral results
	print("\n\n\n>>> Preprocessing data %dhz, sub %s <<<" % (freq, sub))
	print("\nAccuracy: %f %%" % (accuracy))
	print("False positives: %d" % (false_positives))
	print("False negatives: %d" % (false_negatives))
	print("No response: %d\n\n\n" % (no_response))


### Reading the data ###
	raw = mne.io.read_raw_brainvision(os.path.join(workDir, studyDir, \
						eegDir), preload=True)


### Selecting only posterior channels ###
	sel_chan = "^O *|^P *"

	chan_idx = mne.pick_channels_regexp(raw.info["ch_names"], sel_chan)
	chan_idx = np.asarray(chan_idx)
	sel_chans = [raw.info["ch_names"][c] for c in chan_idx]

	raw.pick_channels(sel_chans)


### Finding the data events ###
	events, events_id = mne.events_from_annotations(raw)
	events = events[1:,:]


### Epoching the sequences & baseline correcting ###
	epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, \
								baseline=(None,0), preload=True)

	del raw


### Dowsampling the sequence data ###
	epochs.resample(freq)


### Removing the target trials ###
	targets = np.where(img_cat != 0)[0]

	epochs_data = epochs.get_data(); epochs_data = epochs_data[targets,:,:]
	img_cat = img_cat[targets]
	img_frag = img_frag[targets]
	img_filt = img_filt[targets]


### Putting the sequence data into a dictionary ###
	eeg_data = {
			"eeg_data": epochs_data,
			"img_category": img_cat,
			"img_fragment": img_frag,
			"img_filtering": img_filt,
			"channels": epochs.ch_names,
			"time": epochs.times
	}

	del epochs, epochs_data


### Saving the sequence data ###
	# Saving directories
	saveDir = "s" + str(sub) + "/eeg/preprocessed/"
	fileDir = "preprocessed_" + str(freq) + "hz_sub_" + format(sub, "03")

	# Creating the directory if not existing
	if os.path.isdir(os.path.join(workDir, studyDir, saveDir)) == False: # if not a directory
		os.makedirs(os.path.join(workDir, studyDir, saveDir))

	np.save(os.path.join(workDir, studyDir, saveDir, fileDir), eeg_data)


### ###
if __name__ == "__main__":
	preprEeg()

