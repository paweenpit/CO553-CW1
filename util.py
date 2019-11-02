import numpy as np
import matplotlib.pyplot as plt

def import_clean_data():
	return np.loadtxt('./dataset/clean_dataset.txt')

def import_noisy_data():
	return np.loadtxt('./dataset/noisy_dataset.txt')

def get_depth( tree ):
	if 'label' in tree : 
		return 0
	return (max ( get_depth( tree['right'] ) , get_depth(tree['left']) ) + 1)

''' credited to: https://stackoverflow.com/questions/47048366/implementing-a-copy-deepcopy-clone-function'''
def deepcopy(data):
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = my_deepcopy(value)

        assert id(result) != id(data)

    elif isinstance(data, list):
        result = []
        for item in data:
            result.append(my_deepcopy(item))

        assert id(result) != id(data)

    elif isinstance(data, tuple):
        aux = []
        for item in data:
            aux.append(my_deepcopy(item))
        result = tuple(aux)

        assert id(result) != id(data)

    elif isinstance(data, (int, float, type(None), str, bool)):
        result = data
    else:
        raise ValueError("unexpected type")

    return result