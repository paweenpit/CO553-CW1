'''
10-fold validation of the decision tree
'''
import numpy as np

from decision_tree import decision_tree_learning, import_clean_data,\
        import_noisy_data

<<<<<<< HEAD
def prune_tree(tree, labels):
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

=======

# output : boolean , original tree , modified tree
def prune_tree( data , tree ) : 
    lables = data[:,7]
    original_tree_current_node = tree.copy()
    current_node = tree

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

            original_tree_current_node['left']['is_checked'] = True
            original_tree_current_node['right']['is_checked'] = True

            return True , original_tree_current_node , current_node 
            
    else : 

        is_modified_left, original_tree_left , modified_tree_left = prune_tree( date , current_node['left'] )
        if is_modified_left == True :
            original_tree_current_node['left'] = original_tree_left
            current_node['left'] = modified_tree_left
            return True , original_tree_current_node , current_node

        is_modified_right, original_tree_right , modified_tree_right = prune_tree( data , current_node['right'] )
        if is_modified_right == True :
            original_tree_current_node['right'] = original_tree_right
            current_node['right'] = modified_tree_right
            return True , original_tree_current_node , current_node

        return False , None , None
>>>>>>> 6c0f40791d8e1eb95146ea9b9c6fda926aa61e5f

def K_fold_pruning_evaluation(data, nr_of_folds = 10):
    np.random.shuffle(data)
    folds = np.split(data, nr_of_folds)
    test_data_set = folds[0]
    training_validation_data_set = np.concatenate(folds[1:])
    #initiate arrays to store results
    recall_matrix = []
    precision_matrix = []
    F1_matrix = []
    classification_rates = []
    confusion_tensor = [] #patent pending

    pruned_recall_matrix = []
    pruned_precision_matrix = []
    pruned_F1_matrix = []
    pruned_classification_rates = []
    pruned_confusion_tensor = []

    for index in range(nr_of_folds):
        #split up dataset
        folds = np.split(training_validation_data_set, nr_of_folds)
        evaluation_data_set = folds[index]
        training_data_set = np.concatenate(folds[0:index] + folds[index + 1:])
        labels = training_data_set[:,-1]
        # train and evaluate the unpurned tree
        original_tree, _ = decision_tree_learning(training_data_set, 0)
        confusion_matrix , recall, precision, F1, classification_rate\
        = evaluate(evaluation_data_set, original_tree)
        current_tree = original_tree.copy()
        #prune
        while True:
            pruned_tree, flag = prune_tree(current_tree, labels)
            #break if all nodes have been pruned
            if flag:
                break
            #evaluate pruned tree
            pruned_confusion_matrix , pruned_recall, pruned_precision, pruned_F1,\
            pruned_classification_rate = evaluate(evaluation_data_set, pruned_tree)
            #check if better
            if pruned_classification_rate >= classification_rate:
                current_tree = pruned_tree
                classification_rate = pruned_classification_rate


        #evaluate unpruned and best pruned trees on test dataset
        confusion_matrix , recall, precision, F1, classification_rate\
        = evaluate(test_data_set, original_tree)
        pruned_confusion_matrix , pruned_recall, pruned_precision, pruned_F1,\
        pruned_classification_rate = evaluate(test_data_set, current_tree)
        #store measures
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

    pruned_average_recall = np.mean(recall_matrix, axis=0)
    pruned_average_precision = np.mean(precision_matrix, axis=0)
    pruned_average_F1 = np.mean(precision_matrix, axis=0)
    pruned_average_classification_rate = np.mean(classification_rates)
    pruned_average_confusion_matrix = np.mean(confusion_tensor, axis =0)

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
    K_fold_pruning_evaluation(noisy_data)
