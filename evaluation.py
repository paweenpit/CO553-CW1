'''
10-fold validation of the decision tree
'''
import numpy as np

from decision_tree import decision_tree_learning, import_clean_data,\
        import_noisy_data


def evaluate(test_data_set, trained_tree):
    confusion_matrix = np.zeros((4,4), dtype = int)

    for sample in test_data_set:
        current_node = trained_tree
        attribute_values = sample[0:-1]
        true_label = int(sample[-1])
        while True:
            if 'label' in current_node:
                predicted_label = current_node['label']
                confusion_matrix[true_label -1, predicted_label - 1] += 1
                break
            elif attribute_values[current_node['attribute']] < current_node['value']:
                current_node = current_node['left']
            else:
                current_node = current_node['right']

    average_recall, average_precision, F1, classification_rate = mertics(confusion_matrix)

    return confusion_matrix , [average_recall, average_precision, F1, classification_rate]


def mertics(confusion_matrix):
    recall = np.zeros((1, np.shape(confusion_matrix)[0]), dtype = int)
    precision = np.zeros((1, np.shape(confusion_matrix)[0]), dtype = int)
    for label in range(np.shape(confusion_matrix)[0]):
        recall[0:label] = confusion_matrix[label,label]/np.sum(confusion_matrix[label,:])
        precision[0:label] = confusion_matrix[label,label]/np.sum(confusion_matrix[:,label])

    average_recall = np.mean(recall)
    average_precision = np.mean(precision)
    F1 = 2*(average_precision*average_recall)/(average_precision + average_recall)
    classification_rate = np.sum(np.diagonal(confusion_matrix))/np.sum(confusion_matrix)
    classification_error = 1 - classification_rate

    return average_recall, average_precision, F1, classification_rate



def test(data):
    tree, depth = decision_tree_learning(data, 0)
    print(evaluate(data, tree))




if __name__ == '__main__':
	clean_data = import_clean_data()
	test(clean_data)
