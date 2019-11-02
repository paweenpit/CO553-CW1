from evaluation import *



if __name__ == '__main__':
    STEP3()


def STEP3():
    """
    This is running the code for the STEP 3 fo the corsework spec which is
    evaluation of the the accuracy using 10-fold cross validation on both
    the clean and noisy datasets.
    :retuns:
    """
    clean_data = import_clean_data()
    noisy_data = import_noisy_data()
    clean_measures = K_fold_evaluation(clean_data)
    clean_measures = K_fold_evaluation(noisy_data)
    print('Classification rates: clean data:{0}, noisy data:{1}.'.format(clean_classification_rate, noisy_classification_rate))
