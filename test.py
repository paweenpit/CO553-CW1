from decision_tree import *

def test():
	dat = np.array([
	    [1,1,1,1,1,1],
	    [1,1,1,2,2,2],
	    [3,3,2,2,2,2],
	    [3,2,1,1,1,1],
	    [4,4,1,3,4,4],
	    [1,1,2,3,4,4],
	    [3,2,3,5,5,7],
	    [5,6,3,2,7,7]
	])

	print(decision_tree_learning(dat, 0))

if __name__ == __main__:
	test()

