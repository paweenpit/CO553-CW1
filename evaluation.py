'''
10-fold validation of the decision tree
'''
import numpy as np

from decision_tree import decision_tree_learning, import_clean_data,\
        import_noisy_data

K = 10
test_size = 2000/K


def _10_fold_evaluation(data, nr_of_folds = 10):

    np.random.shuffle(data)
    folds = np.split(data, nr_of_folds)

    recall_matrix = []
    precision_matrix = []
    F1_matrix = []
    classification_rates = []
    confusion_tensor = []

    for index in range(nr_of_folds):
        test_data_set = folds[index]
        training_data_set = np.concatenate(folds[0:index] + folds[index + 1:])

        tree, _ = decision_tree_learning(training_data_set, 0)

        confusion_matrix , recall, precision, F1, classification_rate\
        = evaluate(test_data_set, tree)

        classification_rates.append(classification_rate)

        if index == 0:
            recall_matrix = recall
            precision_matrix = precision
            F1_matrix = F1
            confusion_tensor = confusion_matrix
        else:
            recall_matrix = np.vstack((recall_matrix, recall))
            precision_matrix = np.vstack((precision_matrix, precision))
            F1_matrix = np.vstack((F1_matrix, F1))
            confusion_tensor = np.dstack((confusion_tensor, confusion_matrix))

        print(confusion_tensor[1])

    average_recall = np.mean(recall_matrix, axis=0)
    average_precision = np.mean(precision_matrix, axis=0)
    average_F1 = np.mean(precision_matrix, axis=0)
    average_classification_rate = np.mean(classification_rates)



    # print(average_recall, average_precision, average_F1, average_classification_rate)

    return(average_recall,
            average_precision,
            average_F1,
            average_classification_rate)


def evaluate(test_dataset, trained_tree):
    ''' return: confusion matrix, recall, precision, F1 score, classification rate'''
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

    recall, precision, F1, classification_rate = metrics(confusion_matrix)

    return confusion_matrix, recall, precision, F1, classification_rate


def metrics(confusion_matrix):
    ''' return: avg recall, avg precision, F1, classification rate'''
    num_classes = confusion_matrix.shape[0]
    recall = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    F1 = np.zeros(num_classes)

    for label in range(num_classes):
        # recall = true positive / sum of all predicted positive
        recall[label] = confusion_matrix[label,label]/np.sum(confusion_matrix[label,:])
        # precision is true positive / sum of actual positive
        precision[label] = confusion_matrix[label,label]/np.sum(confusion_matrix[:,label])

        F1[label] = 2*(precision[label]*recall[label])/(precision[label] + recall[label])

    average_recall = np.mean(recall)
    average_precision = np.mean(precision)
    average_precision = np.mean(F1)

    classification_rate = np.sum(np.diagonal(confusion_matrix))/np.sum(confusion_matrix)
    classification_error = 1 - classification_rate

    return recall, precision, F1, classification_rate


def test(data):
    tree, depth = decision_tree_learning(data, 0)
    print(evaluate(data, tree))


if __name__ == '__main__':
    clean_data = import_clean_data()
    noisy_data = import_noisy_data()
    _10_fold_evaluation(clean_data)
    _10_fold_evaluation(noisy_data)
	# test(clean_data)
