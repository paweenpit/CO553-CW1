'''
10-fold validation of the decision tree
'''

from decision_tree import decision_tree_learning, import_clean_data,\
        import_noisy_data


def evaluate(test_data_set, trained_tree):

    current_node = trained_tree
    while True:
        if 'label' in current_node:
            return(current_node['label'])# whe know the prediced label so can assign it
            break
        elif test_data_set[current_node['attribute']] < current_node['value']:
            current_node = current_node['left']
        else:
            current_node = current_node['right']


def test(data):
    row = 1000
    test_data_set = data[row,:-1]
    correct_label = int(data[row,-1])
    tree, depth = decision_tree_learning(data, 0)
    predicted_label = evaluate(test_data_set, tree)
    print(predicted_label, correct_label)




if __name__ == '__main__':
	clean_data = import_clean_data()
	test(clean_data)
