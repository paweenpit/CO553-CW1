from decision_tree import *
from util import *

COLORS = ['b', 'g', 'r', 'c', 'm', 'y']

def plot_node(decision_tree, depth, x, y, height):
	''' recursive function to plot the decision tree'''
	# if the node is a leaf
	if 'label' in decision_tree:
		plt.text(
			x, y, 
			'{}'.format(int(decision_tree['label'])),
			size = height,
			ha = 'center', 
			va = 'center', 
			bbox = dict(boxstyle='round', facecolor='yellow'), 
			fontsize = 10 - np.minimum(depth, 5)
		)

	else:
		plt.text(
			x, y, 
			'X{dt[attribute]}<{dt[value]}'.format(dt=decision_tree),
			size = height,
			ha = 'center', 
			va = 'center', 
			bbox = dict(boxstyle='round', facecolor='white'), 
			fontsize = 10 - np.minimum(depth, 5)
		)

		color = COLORS[np.random.choice(len(COLORS))]
		# recursively call left and right leaves
		if decision_tree['left'] != None :
			x_left = x - 1/(2**(depth+2))
			y_left = y - height

			plt.plot([x, x_left], [y, y_left], color=color)
			plot_node(decision_tree['left'], depth+1, x_left, y_left, height)

		if decision_tree['right'] != None :
			x_right = x + 1/(2**(depth+2))
			y_right = y - height

			plt.plot([x, x_right], [y, y_right], color=color)
			plot_node(decision_tree['right'], depth+1, x_right, y_right, height)


def visualize(dataset, depth=5):
	''' visualize the tree'''
	decision_tree, _ = decision_tree_learning(dataset)

	plt.figure(figsize=(12,6))
	axes = plt.gca()
	axes.set_xlim([0,1])
	axes.set_ylim([0,1])
	height = 1/depth

	plot_node(decision_tree, 0, 0.5, 1, height)

	plt.axis('off')
	plt.show()