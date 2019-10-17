import numpy as np
import matplotlib.pyplot as plt

def import_clean_data():
	return np.loadtxt('clean_dataset.txt')

def import_noisy_data():
	return np.loadtxt('noisy_dataset.txt')

def decision_tree_learning(training_dataset, depth):
	
