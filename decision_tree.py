import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.lines as mlines

def import_clean_data():
	return np.loadtxt('./dataset/clean_dataset.txt')

def import_noisy_data():
	return np.loadtxt('./dataset/noisy_dataset.txt')

def H(labels):
	''' return H-function of labels'''
	total = len(labels)
	_, counts = np.unique(labels, return_counts=True)
	probs = counts / total
	
	return -sum(probs * np.log2(probs))

def gain(dataset, left_data, right_data):
	''' return gain value of a feature according to left and right data'''
	len_all = len(dataset)
	len_left = len(left_data)
	len_right = len(right_data)

	remainder = (H(left_data)*len_left + H(right_data)*len_right) / len_all

	return H(dataset) - remainder

def feature_gain(dataset, index):
	''' return: (max gain, value) to split a index-th feature
		according to its gain value'''
	# iterate through each value in feature and compute gain
	best_feature_split = (-float('inf'), 0, [], []) # best feature gain, best value, left_data, right_data
	len_data = len(dataset)

	sorted_data = dataset[dataset[:,index].argsort()] # sort data by feature
	feature_values = np.unique(dataset[:,index])

	for val in feature_values:
		left_data = sorted_data[np.where(sorted_data[:,index] <= val)]
		right_data = sorted_data[np.where(sorted_data[:,index] > val)]

		feature_gain = gain(sorted_data, left_data, right_data)

		if feature_gain > best_feature_split[0]:
			best_feature_split = (feature_gain, val, left_data, right_data)

	return best_feature_split

def find_split(dataset):
	''' return: best feature, value, left data and right data for the best split'''
	best_gain = -float('inf')
	best_split = (0, 0, [], []) # best feature, best value, left_data, right_data
	label = dataset[:,-1]

	for att_i in range(len(dataset[0])-1):
		gain, value, left_data, right_data = feature_gain(dataset, att_i)
		if gain > best_gain:
			best_gain = gain
			best_split = (att_i, value, left_data, right_data)

	return best_split

def decision_tree_learning(dataset, depth=0):
	''' return: root node of the decision tree '''
	# check if all values in the label are the same
	# do not split and return a leaf
	if len(np.unique(dataset[:,-1])) == 1:
		return {'attribute': 0, 'value': 0, 'left': None, 'right': None}, depth
	else:
		# find feature and value to split the dataset
		attribute, value, left_data, right_data = find_split(dataset)

		# recursively calls the fucntion on left and right node
		left_node, left_depth = decision_tree_learning(left_data, depth+1)
		right_node, right_depth = decision_tree_learning(right_data, depth+1)

		root = {'attribute': attribute, 'value': value, 'left': left_node, 'right': right_node}

		return root, max(left_depth, right_depth)

def nodes(decision_tree, depth , x , y , height , width):

	if decision_tree['attribute'] == 0 and decision_tree['value']==0 and decision_tree['left'] == None and decision_tree['right'] == None :
			plt.text(x, y, "X"+str(decision_tree['attribute'])+"<"+str(decision_tree['value']) , size=height,ha="center", va="center", bbox=dict(boxstyle='round', facecolor='white' ) , fontsize =  10 - np.minimum(depth , 5))

	plt.text(x, y, "X"+str(decision_tree['attribute'])+"<"+str(decision_tree['value']) , size=height,ha="center", va="center", bbox=dict(boxstyle='round', facecolor='white') , fontsize = 10 -  np.minimum(depth , 5))
	if decision_tree['left'] != None :
		x_left = x - (1/ (2**(depth+2)))
		y_left = y - height
		plt.plot([x , x_left] , [ y , y_left] , marker = 'o')
		nodes (decision_tree['left'] , depth+1  , x_left , y_left , height , width )

	if decision_tree['right'] != None :
		x_right = x + (1/(2**(depth+2)))
		y_right = y - height
		plt.plot([x , x_right] , [ y , y_right], marker = 'o')
		nodes (decision_tree['right'] , depth+1 , x_right , y_right, height , width  )

def visualization () :

	decision_tree , _ = decision_tree_learning(import_clean_data(),0)
	depth = 5

	fig = plt.figure(figsize=(12,6))
	axes = plt.gca()
	axes.set_xlim([0,1])
	axes.set_ylim([0,1])
	# plt.subplots_adjust(left=0.2, right=1.2, top=1.2, bottom=0.2)
	height = 1/depth
	width =  height
	nodes(decision_tree, 0 , 0.5 , 1 , height , width)
	plt.axis('off')
	plt.show()				

visualization()