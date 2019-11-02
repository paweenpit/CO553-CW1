'''
10-fold validation of the decision tree
'''
from decision_tree import decision_tree_learning
from util import *

import copy

def prune_tree(labels, tree):
    ''' prune one parent of 2 leaves
        return: is_pruned, original tree, pruned tree'''
    original_tree_current_node = tree.copy()
    current_node = tree

    # if current_node is a leaf
    if 'label' in current_node:
        return False, original_tree_current_node, current_node

    # if current_node is a parent of 2 unchecked leaves
    if 'label' in current_node['left'] and 'label' in current_node['right'] and current_node['left']['is_checked'] == False:
        left_leaf = current_node['left']
        right_leaf = current_node['right']

        left_label = left_leaf['label']
        right_label = right_leaf['label']

        left_num = np.count_nonzero(labels == left_label)
        right_num = np.count_nonzero(labels == right_label)

        # change label according to training dataset
        label = left_label if left_num > right_num else right_label

        # change current node to a leaf
        current_node = {'label': label, 'is_checked': False}

        # set original tree's children is_checked to True
        original_tree_current_node['left']['is_checked'] = True
        original_tree_current_node['right']['is_checked'] = True

        return True, original_tree_current_node, current_node

    # a parent with at least one child checked
    else:
        # prune left tree
        is_modified_left, original_tree_left, modified_tree_left = prune_tree( labels , current_node['left'] )
        # if pruning is successful, return
        if is_modified_left == True:
            original_tree_current_node['left'] = original_tree_left
            current_node['left'] = modified_tree_left
            return True, original_tree_current_node, current_node

        # prune right tree
        is_modified_right, original_tree_right, modified_tree_right = prune_tree(labels, current_node['right'] )
        if is_modified_right == True :
            original_tree_current_node['right'] = original_tree_right
            current_node['right'] = modified_tree_right
            return True, original_tree_current_node, current_node

    # last case: if is a parent with 2 children checked
    return False, original_tree_current_node, current_node


def K_fold_pruning_evaluation(data, nr_of_folds = 10):
    np.random.shuffle(data)
    folds1 = np.split(data, nr_of_folds)
    for i in range(nr_of_folds):
        #print status###########################################
        print ('#'*70)
        print ('#'*70)
        print('USING FOLD {} AS THE TEST DATA'.format(i + 1))
        print ('-'*70)
        ########################################################
        test_data_set = folds1[i]
        training_validation_data_set_folds =\
        [index for index in range(nr_of_folds) if index != i]
        training_validation_data_set = np.concatenate(folds1[0:i] + folds1[i + 1:])

        #initiate arrays to store results
        recall_matrix = []
        precision_matrix = []
        F1_matrix = []
        classification_rates = []
        confusion_tensor = [] # patent pending by Olle

        pruned_recall_matrix = []
        pruned_precision_matrix = []
        pruned_F1_matrix = []
        pruned_classification_rates = []
        pruned_confusion_tensor = []

        #split up dataset
        folds = np.split(training_validation_data_set, nr_of_folds - 1)

        for index in range(nr_of_folds - 1):
            training_data_set_folds = training_validation_data_set_folds
            [i for i in training_validation_data_set_folds if i != index]

            print('WITH FOLD {} AS THE VALIDATION DATA'.format(training_validation_data_set_folds[index]  + 1))
            print('USING FOLDS {} AS THE TRAINING DATA'.format(training_data_set_folds))

            evaluation_data_set = folds[index]
            training_data_set = np.concatenate(folds[0:index] + folds[index + 1:])
            labels = training_data_set[:,-1]
            # train and evaluate the unpurned tree
            original_tree, _ = decision_tree_learning(training_data_set, 0)
            confusion_matrix , recall, precision, F1, classification_rate\
            = evaluate(evaluation_data_set, original_tree)
            print('The classicfication for original tree on evaluation data: {}'.format(classification_rate))
            current_tree = copy.deepcopy(original_tree)
            #prune
            while True:
                flag, current_tree, pruned_tree = prune_tree(labels, current_tree)
                #break if all nodes have been pruned
                if not flag:
                    break
                #evaluate pruned tree
                pruned_confusion_matrix , pruned_recall, pruned_precision, pruned_F1,\
                pruned_classification_rate = evaluate(evaluation_data_set, pruned_tree)

                #check if better
                if pruned_classification_rate > classification_rate:
                    current_tree = pruned_tree
                    classification_rate = pruned_classification_rate

            print(current_tree)
            #evaluate unpruned and best pruned tree on test dataset
            confusion_matrix , recall, precision, F1, classification_rate\
            = evaluate(test_data_set, original_tree)
            print('The classicfication for original tree on test data: {}'.format(classification_rate))
            pruned_confusion_matrix , pruned_recall, pruned_precision, pruned_F1,\
            pruned_classification_rate = evaluate(evaluation_data_set, current_tree)
            print('The classicfication for pruned tree on evaluation data: {}'.format(pruned_classification_rate))
            pruned_confusion_matrix , pruned_recall, pruned_precision, pruned_F1,\
            pruned_classification_rate = evaluate(test_data_set, current_tree)
            print('The classicfication for pruned tree on test data: {}'.format(pruned_classification_rate))
            #store measures
            classification_rates.append(classification_rate)
            pruned_classification_rates.append(pruned_classification_rate)
            if index == 0:
                recall_matrix = recall
                precision_matrix = precision
                F1_matrix = F1
                confusion_tensor = confusion_matrix

                pruned_recall_matrix = pruned_recall
                pruned_precision_matrix = pruned_precision
                pruned_F1_matrix = pruned_F1
                pruned_confusion_tensor = pruned_confusion_matrix
            else:
                recall_matrix = np.vstack((recall_matrix, recall))
                precision_matrix = np.vstack((precision_matrix, precision))
                F1_matrix = np.vstack((F1_matrix, F1))
                confusion_tensor = np.vstack((confusion_tensor, confusion_matrix))

                pruned_recall_matrix =\
                np.vstack((pruned_recall_matrix, pruned_recall))
                pruned_precision_matrix =\
                np.vstack((pruned_precision_matrix, pruned_precision))
                pruned_F1_matrix =\
                np.vstack((pruned_F1_matrix, pruned_F1))
                pruned_confusion_tensor =\
                np.vstack((pruned_confusion_tensor, pruned_confusion_matrix))


        #calculate mean of evaluation measures
        average_recall = np.mean(recall_matrix, axis=0)
        average_precision = np.mean(precision_matrix, axis=0)
        average_F1 = np.mean(precision_matrix, axis=0)
        average_classification_rate = np.mean(classification_rates)
        average_confusion_matrix = np.mean(confusion_tensor, axis =0)

        pruned_average_recall = np.mean(pruned_recall_matrix, axis=0)
        pruned_average_precision = np.mean(pruned_precision_matrix, axis=0)
        pruned_average_F1 = np.mean(pruned_precision_matrix, axis=0)
        pruned_average_classification_rate = np.mean(pruned_classification_rates)
        pruned_average_confusion_matrix = np.mean(pruned_confusion_tensor, axis =0)

        unpruned_measures =\
        [average_recall, average_precision, average_F1, average_classification_rate]
        pruned_measures =\
        [pruned_average_recall, pruned_average_precision, pruned_average_F1,\
        pruned_average_classification_rate]

    return unpruned_measures, pruned_measures


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
    unpruned_measures, pruned_measures = K_fold_pruning_evaluation(noisy_data)
    print(unpruned_measures[3], pruned_measures[3])
