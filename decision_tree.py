import numpy as np
import math
import sys

LABEL_COLUMN = 7

def load_dataset(filepath):
    data = np.loadtxt(filepath)
    return data

def decision_tree_learning(dataset, depth):
    labels = dataset[:, 7]
    
    if np.all(labels[0] == labels):
        # Base Case
        # Return leaf node with this value, depth
        return ({"class": labels[0]}, depth+1)
    else:
        # Recursive case
        # Find the optimum split
        root_node = find_split(dataset)
        (left_data, right_data) = split_data(dataset, root_node["attribute"], root_node["value"])
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
        maxIg = 0
        maxIgValue = None
        
        for i in range(len(sortedAttributeValues)-1):
            split_value = (sortedAttributeValues[i] + sortedAttributeValues[i+1])/2
            (left_data, right_data) = split_data(dataset, attribute, split_value)
            ig = entropy(dataset) - remainder(left_data, right_data)
            if ig > maxIg:
                maxIg = ig
                maxIgValue = split_value
            
        return(maxIgValue, maxIg)

def entropy(dataset):
    ans = 0
    labels = dataset[:, 7]
    label_counts = {}
    for label in labels:
        label = str(int(label))
        label_counts[label] = label_counts.get(label, 0) + 1
    for i in range(1, 7):
        label_freq = label_counts.get(str(i), 0)
        if label_freq != 0:
            probability = label_counts.get(str(i), 0)/len(labels)
            ans += probability * math.log(probability, 2)
    ans *= -1
    return ans

def remainder(left_data, right_data):
    left_size, _ = left_data.shape
    right_size, _ = right_data.shape
    
    left_remainder = left_size/(left_size + right_size) * entropy(left_data)
    right_remainder = right_size/(left_size+right_size) * entropy(right_data)
    
    return left_remainder + right_remainder
    

def split_data(data, attribute, value):
    left_data = data[data[:, attribute] < value]
    right_data = data[data[:, attribute] >= value]
    return (left_data, right_data)
    
if __name__ == "__main__":
    if sys.argv[1] == None:
        print("dataset filepath needed")
    else:
        dataset = load_dataset(sys.argv[1])
        node = decision_tree_learning(dataset, 1)
        print(node)