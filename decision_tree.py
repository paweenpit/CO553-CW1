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

def visualisation(decision_tree,start_point):
# 	fig , ax = plt.subplots()
# 	#currentAxis = plt.gca()
# 	rectangle = {'rect' : ptch.Rectangle((start_point[0],start_point[1]),0.2,0.1,fill=None,alpha=1)}
# 	ax.add_artist (rectangle[1])
# 	ax.annotate(0.5,(0.2,0.2))
# 	plt.axis('off')
# 	plt.show()
	fig, ax = plt.subplots()
	rectangles = { str(decision_tree['attribute']): ptch.Rectangle((start_point[0],start_point[1]), 5, 2,fill = None , alpha = 1)}

	for r in rectangles:
		ax.add_artist(rectangles[r])
		rx, ry = rectangles[r].get_xy()
		cx = rx + rectangles[r].get_width()/2.0
		cy = ry + rectangles[r].get_height()/2.0
		ax.annotate(r, (cx, cy), color='b', weight='bold',fontsize=6, ha='center', va='center')

	ax.set_xlim((0, 15))
	ax.set_ylim((0, 15))
	plt.axis('off')
	ax.set_aspect('equal')
	plt.show()

start_point=[5,10]
decision_tree , _ = decision_tree_learning(import_clean_data(),0)
visualisation(decision_tree,start_point)
