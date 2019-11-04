This file explains how to run the code submitted for the CO553 Intro to Machine Learning coursework 1: desicion trees from the group comprising of Niusha Alavi, Olle Nilsson, Paween Pitimanaaree and Alfred Tingey.


Which file contains what?

desicion_tree.py contains the implementation of the algorirm for training desicion trees.

evaluation.py contains the implementation of crossvalidation and pruning algoritms.

visulize.py contains the implementation of decision tree visulisation.

util.py is for utilites such as functions for importing data.


All other files(with .py extention) are setup to run the code for the various questions in the assingment.


How to run the code?
After unpacking the zip file navigate to the unzipped folder in the directory:

To run the code that we used to calculate the result that are in the report call:

python3 main.py 

in the terminal. This will first run the 10-fold crossvalidation on both clean and noisy datasets and then run the 10-fold crossvalidation with pruning on the clean and noisy datasets. Results will print in screen while it calculates and averages displayed at the end. Note that the dataset is shuffled in a random order each time so results may not mach exacly to whats in the report.


To run our crossvalidation code on a different dataset:

move the file containg the dataset to the same folder as our .py files

open the run_crossvalidation_on_unseen_data.py file in an editor. On line 8 where it says 'unseendata.txt' replace this with the name of the file for the dataset you want to run. Then call:

run_crossvalidation_on_unseen_data.py 

in the terminal. This will first run the 10-fold crossvalidation on your new dataset. And results printed at the end.


To run our crossvalidation code with pruning on a different dataset:

move the file containg the dataset to the same folder as our .py files

open the run_pruning_on_unseen_data.py file in an editor. On line 8 where it says 'unseendata.txt' replace this with the name of the file for the dataset you want to run. Then call:

run_pruning_on_unseen_data.py 

in the terminal. This will first run the 10-fold crossvalidation on your new dataset. And results printed at the end.