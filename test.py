from decision_tree import *

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

def test(data):
	tree, depth = decision_tree_learning(data, 0)
	print('depth: {}'.format(depth))
	print('tree: {}'.format(tree))

if __name__ == '__main__':
	clean_data = import_clean_data()
	test(clean_data)
