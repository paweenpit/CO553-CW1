ef K_fold_pruning_evaluation(data, nr_of_folds = 10):
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
