import numpy as np
import sys

def load_dataset(filepath):
    data = np.loadtxt(filepath)
    return data

def decision_tree_learning(dataset: np.ndarray, depth: int):
    print(dataset)
    
if __name__ == "__main__":
    if sys.argv[1] == None:
        print("dataset filepath needed")
    else:
        dataset = load_dataset(sys.argv[1])
        decision_tree_learning(dataset, 1)