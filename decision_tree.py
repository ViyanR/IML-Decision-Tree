import numpy as np
import sys

LABEL_COLUMN = 7

def load_dataset(filepath):
    data = np.loadtxt(filepath)
    return data

def decision_tree_learning(dataset, depth):
    labels = dataset[:, 7]
    print(labels)
    
    if np.all(labels[0] == labels):
        pass
        # Base Case
        # Return leaf node with this value, depth
        return {"class": labels[0]}
    else:
        # Recursive case
        # Find the optimum split
        root_node = find_split(dataset)
        (left_data, right_data) = split_data(root_node["attribute"], root_node["value"])
        root_node["left"], l_depth = decision_tree_learning(left_data, depth+1)
        root_node["right"], r_depth = decision_tree_learning(right_data, depth+1)
        return (root_node, max(l_depth, r_depth))

def find_split(dataset):
    node = {}
    maxIg = 0
    maxIgAttribute = None
    maxIgValue = None
    
    for attribute in range(7):
        (value, ig) = find_split_value(dataset, attribute)
        if ig > maxIg:
            maxIgValue = value
            maxIgAttribute = attribute

    node["attribute"] = maxIgAttribute
    node["value"] = maxIgValue

    return node

def find_split_value(dataset, attribute):
        attributeValues = dataset[:, attribute]
        sortedAttributeValues = np.sort(attributeValues)
        for i in range(len(sortedAttributeValues)-1):
            split_value = (sortedAttributeValues[i] + sortedAttributeValues[i+1])/2
            
        print(labels)
        return(5, 2)

def split_data(attribute, value):
    left_data = data[data[:, attribute] < value]
    right_data = data[data[:, attribute] >= value]
    return (left_data, right_data)
    
if __name__ == "__main__":
    if sys.argv[1] == None:
        print("dataset filepath needed")
    else:
        dataset = load_dataset(sys.argv[1])
        decision_tree_learning(dataset, 1)