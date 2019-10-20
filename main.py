import numpy as np
import matplotlib.pyplot as plt

def import_clean_data():
	return np.loadtxt('clean_dataset.txt')

def import_noisy_data():
	return np.loadtxt('noisy_dataset.txt')

def H(labels):
    ''' return H-function of labels'''
    total = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / total
    return -sum(probs * np.log2(probs))

def gain(sorted_data, left_data, right_data):
	''' return gain value of the feature according the split index'''
	len_all = len(sorted_data)
	len_left = len(left_data)
	len_right = len(right_data)

	remainder = (H(left_data)*len_left + H(right_data)*len_right) / len_all

	return H(sorted_data) - remainder

def feature_gain(feature, label):
	''' return (max gain, value) to split a feature 
		according to its gain value'''
	# iterate through each value in feature and compute gain
	best_feature_split = (-float('inf'), 0, [], []) # best feature gain, best value, left_data, right_data
	len_data = len(label)

	data = np.column_stack((feature,label)) # 2d np array of feature and label
	sorted_data = data[data[:,0].argsort()]

	for ind, (val, _) in enumerate(sorted_data):
		# skip index if it has the same value as the previous one
		if ind < len_data-1 and val == sorted_data[ind+1,0]:
			continue
		else:
			left_data = sorted_data[:ind]
			right_data = sorted_data[ind:]

			att_gain = gain(sorted_data, left_data, right_data)

			if att_gain > best_feature_split[0]:
				best_feature_split = (att_gain, val, left_data, right_data)

	return best_feature_split

def find_split(dataset):
	''' return: the best feature, value(continuous), and index 
		for the best split'''
	if len(dataset) == 0:
		return (0, 0, [], [])

	best_gain = -float('inf')
	best_split = (0, 0, [], []) # best feature, best value, left_data, right_data
	label = dataset[:,-1]

	for i in range(len(dataset[0])-1):
		gain, value, left_data, right_data = feature_gain(dataset[:,i], label)
		if gain > best_gain:
			best_gain = gain
			best_split = (i, value, left_data, right_data)

	return best_split

