'''
10-fold validation of the decision tree
'''
import numpy as np

from decision_tree import decision_tree_learning, import_clean_data,\
        import_noisy_data

def prune_tree(data,tree):
    lables = data[,7]
    original_tree = tree.copy()
    current_node = tree
    while( True ) :
        if ('label' in current_node['left']) and ('label' in current_node['right'] ):
            left_leaf = current_node['left']
            right_leaf = current_node['right']
            if left_leaf['is_checked'] == False :
                left_num = lables.count(left_leaf['label'])
                right_num = lables.count(right_leaf['label'])
                label = left_leaf['label']
                if left_num < right_num :
                    label = right_leaf['label']
                 
                current_node['attribute'] = None
                current_node['value'] = None
                current_node['left'] = None
                current_node['right'] = None

                current_node['lable'] = label
                current_node['is_cheked'] = False

                original_tree['left']['is_checked'] = True
                original_tree['right']['is_checked'] = True
                
                

def K_fold_pruning_evaluation(data, nr_of_folds = 10):
    np.random.shuffle(data)
    folds = np.split(data, nr_of_folds)
    test_data_set = folds[0]
    training_validation_data_set = np.concatenate(folds[1:])
    #train and evaluate unpruned trees
    (average_classification_rate,
    average_recall,
    average_precision,
    average_F1,
    average_confusion_matrix,
    trees) = \
    K_fold_evaluation(training_validation_data_set,\
            nr_of_folds = nr_of_folds - 1, shuffle = False)

    print(average_classification_rate,
           average_recall,
           average_precision,
           average_F1)

    folds = np.split(data, nr_of_folds)
    for tree in range(len(trees)):
        target_classification_rate = average_classification_rate
        target_recall = average_recall
        target_precision = average_precision
        target_F1 = average_F1
        target_confusion_matrix = average_confusion_matrix

        while True:
            recall_matrix = []
            precision_matrix = []
            F1_matrix = []
            classification_rates = []
            confusion_tensor = []
            #pruned_tree, flag = prune(trees[tree])
            # if flag:
            #     break
            for index in range(nr_of_folds):

                evaluation_data_set = folds[index]
                confusion_matrix, recall, precision, F1, classification_rate\
                = evaluate(test_data_set, trees[tree])
                print(classification_rate)

                confusion_matrix = np.reshape(confusion_matrix,(1, 4, 4))
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
                    confusion_tensor = np.vstack((confusion_tensor, confusion_matrix))

            #calculate mean of evaluation measures
            pruned_average_recall = np.mean(recall_matrix, axis=0)
            pruned_average_precision = np.mean(precision_matrix, axis=0)
            pruned_average_F1 = np.mean(precision_matrix, axis=0)
            pruned_average_classification_rate = np.mean(classification_rates)
            pruned_average_confusion_matrix = np.mean(confusion_tensor, axis =0)

            if pruned_average_classification_rate <= target_classification_rate:
                trees[tree] = pruned_tree
                target_classification_rate = pruned_average_classification_rate
                target_recall = pruned_average_recall
                target_precision = pruned_average_precision
                target_F1 = pruned_average_F1
                target_confusion_matrix = pruned_average_confusion_matrix


    return(trees)



   #TODO: compute average measures over classes
   # update/prune and run again and compare
   # save values for the best tree


def K_fold_evaluation(data, nr_of_folds = 10, shuffle = True):
    '''
    return: average confusion matrix, recall, precision, F1 score, classification rate
    across the K-folds.
    '''
    #shuffle the data so its not in labeled order
    if shuffle == True:
        np.random.shuffle(data)
    folds = np.split(data, nr_of_folds)
    #initiate arrays for storing evaluation measures
    recall_matrix = []
    precision_matrix = []
    F1_matrix = []
    classification_rates = []
    confusion_tensor = []
    trees = []

    for index in range(nr_of_folds):
        #pick out folds for training and testing
        test_data_set = folds[index]
        training_data_set = np.concatenate(folds[0:index] + folds[index + 1:])
        #train the tree
        tree, depth = decision_tree_learning(training_data_set, 0)
        trees.append(tree)
        #evaluate the tree
        confusion_matrix , recall, precision, F1, classification_rate\
        = evaluate(test_data_set, tree)

        #store evaluation measures
        confusion_matrix = np.reshape(confusion_matrix,(1, 4, 4))
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
            confusion_tensor = np.vstack((confusion_tensor, confusion_matrix))

    #calculate mean of evaluation measures
    average_recall = np.mean(recall_matrix, axis=0)
    average_precision = np.mean(precision_matrix, axis=0)
    average_F1 = np.mean(precision_matrix, axis=0)
    average_classification_rate = np.mean(classification_rates)
    average_confusion_matrix = np.mean(confusion_tensor, axis =0)

    return(average_classification_rate,
           average_recall,
           average_precision,
           average_F1,
           average_confusion_matrix,
           trees)


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



if __name__ == '__main__':
    clean_data = import_clean_data()
    noisy_data = import_noisy_data()
    # clean_classification_rate = K_fold_evaluation(clean_data)[0]
    # noisy_classification_rate = K_fold_evaluation(noisy_data)[0]
    # print('Classification rates: clean data:{0}, noisy data:{1}.'.format(clean_classification_rate, noisy_classification_rate))
    K_fold_pruning_evaluation(noisy_data)
