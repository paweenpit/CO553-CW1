import numpy as np
from evaluation import evaluate
from util import import_new_data

#move the file with the data you want to run our code on to the same folder as this file
# replace unseendata.txt below with the actual filename
#call python3 run_on_unseen_data.py from teriminal
new_data_file_name = 'unseendata_placeholder.txt'
new_data = import_new_data(new_data_file_name)

# Load 'best' tree
decision_tree = np.load('desicion_tree.npy', allow_pickle = True).item()

#run classification of the dataset
confusion_matrix , recall, precision, F1, classification_rate =\
evaluate(new_data, decision_tree)

print('Confusion matrix:')
print(np.round(confusion_matrix, 0))
print('Recall per class: {}'\
.format(np.round(recall, 3)))
print('Precision per class: {}'\
.format(np.round(precision, 3)))
print('F1-score per class: {}'\
.format(np.round(F1, 3)))
print('Classification rate: {}'\
.format(np.round(classification_rate, 3)))
