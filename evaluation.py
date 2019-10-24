'''
10-fold validation of the decision tree
'''
import numpy as np

from decision_tree import decision_tree_learning, import_clean_data,\
        import_noisy_data

K = 10
test_size = 2000/K


def create_fold(data, nr_of_folds = 10):

    np.random.shuffle(data)
    folds = np.split(data, nr_of_folds)

    for fold in folds:
        training_data_set = folds



    print(folds[0])


def evaluate(test_dataset, trained_tree):
    ''' return: confusion matrix, [avg recall, avg precision, F1 score, classification rate]'''
    confusion_matrix = np.zeros((4,4), dtype = int)

    # pass each sample in the trained_tree to get a predicted label
    for sample in test_dataset:
        current_node = trained_tree
        attribute_values = sample[:-1]
        true_label = int(sample[-1])

        while True:
            # if we find the leaf node
            # update confusion matrix and exit
            if 'label' in current_node:
                predicted_label = current_node['label']
                confusion_matrix[true_label -1, predicted_label - 1] += 1
                break
            # iterate through left node if attribute value is less than the split node value
            elif attribute_values[current_node['attribute']] < current_node['value']:
                current_node = current_node['left']
            # iterate through right node if attribute value is not less than the split node value
            else:
                current_node = current_node['right']

    average_recall, average_precision, F1, classification_rate = metrics(confusion_matrix)

    return confusion_matrix , [average_recall, average_precision, F1, classification_rate]


def metrics(confusion_matrix):
    ''' return: avg recall, avg precision, F1, classification rate'''
    num_classes = confusion_matrix.shape[0]
    recall = np.zeros(num_classes)
    precision = np.zeros(num_classes)

    for label in range(num_classes):
        # recall = true positive / sum of all predicted positive
        recall[label] = confusion_matrix[label,label]/np.sum(confusion_matrix[label,:])
        # precision is true positive / sum of actual positive
        precision[label] = confusion_matrix[label,label]/np.sum(confusion_matrix[:,label])

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
    # create_fold(clean_data)
	test(clean_data)
