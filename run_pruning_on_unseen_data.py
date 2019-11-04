from evaluation import *
from util import import_new_data

#move the file with the data you want to run our code on to the same folder as this file
# replace unseendata.txt below with the actual filname
#call python3 run_on_unseen_data.py from teriminal

new_data__file_name = 'unseendata.txt'
new_data = import_new_data(new_data__file_name)


# this block will run the 10-fold crossvalidation with pruning on
#the data imported and print all results as it as it goes and average results at the end
print ('#'*70)
print('PRUNING AND 10-FOLD CROSSVALIDATION ON UNSEEN DATA')
print ('#'*70)
K_fold_pruning_evaluation(new_data)
