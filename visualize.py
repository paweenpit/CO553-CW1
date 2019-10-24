import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from decision_tree import decision_tree_learning, import_clean_data,\
        import_noisy_data

def nodes(decision_tree, depth , x , y , height ):

	if decision_tree['attribute'] == 0 and decision_tree['value']==0 and decision_tree['left'] == None and decision_tree['right'] == None :
			plt.text(x, y, "X"+str(decision_tree['attribute'])+"<"+str(decision_tree['value']) , size=height,ha="center", va="center", bbox=dict(boxstyle='round', facecolor='white' ) , fontsize =  10 - np.minimum(depth , 5))

	plt.text(x, y, "X"+str(decision_tree['attribute'])+"<"+str(decision_tree['value']) , size=height,ha="center", va="center", bbox=dict(boxstyle='round', facecolor='white') , fontsize = 10 -  np.minimum(depth , 5))
	if decision_tree['left'] != None :
		x_left = x - (1/ (2**(depth+2)))
		y_left = y - height
		plt.plot([x , x_left] , [ y , y_left] , marker = 'o')
		nodes (decision_tree['left'] , depth+1  , x_left , y_left , height  )

	if decision_tree['right'] != None :
		x_right = x + (1/(2**(depth+2)))
		y_right = y - height
		plt.plot([x , x_right] , [ y , y_right], marker = 'o')
		nodes (decision_tree['right'] , depth+1 , x_right , y_right, height  )

def visualization () :
	decision_tree , _ = decision_tree_learning(import_clean_data(),0)
	depth = 5

	plt.figure(figsize=(12,6))
	axes = plt.gca()
	axes.set_xlim([0,1])
	axes.set_ylim([0,1])
	height = 1/depth
	nodes(decision_tree, 0 , 0.5 , 1 , height )
	plt.axis('off')
	plt.show()				
