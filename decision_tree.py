import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from visualise import visualise_tree

LABEL_COLUMN = 7
NUMBER_OF_FOLDS = 10
NUMBER_OF_ROOMS = 4

# Convert the txt dataset to a numpy ndarray
def load_dataset(filepath):
    data = np.loadtxt(filepath)
    return data

# Recursive function to build a decision tree to classify the data
def decision_tree_learning(dataset, depth):
    labels = dataset[:, LABEL_COLUMN]
    
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

# Go through each attribute, and find which attribute gives the optimal information gain. Return
# the attribute and split value as a node
def find_split(dataset):
    node = {}
    maxIg = 0
    maxIgAttribute = None
    maxIgValue = None
    
    for attribute in range(LABEL_COLUMN):
        (value, ig) = find_split_value(dataset, attribute)
        if ig > maxIg:
            maxIgValue = value
            maxIgAttribute = attribute

    node["attribute"] = maxIgAttribute
    node["value"] = maxIgValue

    return node

# Take in an attribute and dataset, then find the split value which gives the optimal information
# gain. Return the split value and the information gain.
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

# Calculate the entropy of a dataset
def entropy(dataset):
    ans = 0
    labels = dataset[:, LABEL_COLUMN]
    label_counts = {}
    for label in labels:
        room = str(int(label))
        label_counts[room] = label_counts.get(room, 0) + 1
    for i in range(1, 5):
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

# Split the dataset into two by comparing an attribute with a value
def split_data(data, attribute, value):
    left_data = data[data[:, attribute] < value]
    right_data = data[data[:, attribute] >= value]
    return (left_data, right_data)

# 10-fold cross validation of data
def cross_validate(data):
    k = NUMBER_OF_FOLDS
    np.random.shuffle(data)
    split = np.split(data, k)

    avg_confusion_matrix = np.zeros((NUMBER_OF_ROOMS,NUMBER_OF_ROOMS))
    avg_precision = np.zeros(NUMBER_OF_ROOMS)
    avg_recall = np.zeros(NUMBER_OF_ROOMS)
    avg_f1 = np.zeros(NUMBER_OF_ROOMS)

    for i in range(NUMBER_OF_FOLDS):
        data_copy = split.copy()
        test_data = data_copy.pop(i)
        training_data = np.concatenate(data_copy)
        (trained_tree, depth) = decision_tree_learning(training_data, 1)
        (confusion_matrix, precision, recall, f1) = evaluate(test_data, trained_tree)
        avg_confusion_matrix += confusion_matrix
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1
    
    avg_confusion_matrix /= NUMBER_OF_FOLDS
    avg_precision /= NUMBER_OF_FOLDS
    avg_recall /= NUMBER_OF_FOLDS
    avg_f1 /= NUMBER_OF_FOLDS
    
    return (avg_confusion_matrix, avg_precision, avg_recall, avg_f1)

# Find the confusion matrix, precision, recall and f1 of a given trained decision tree
def evaluate(test_db, trained_tree):
    confusion_matrix = np.zeros((NUMBER_OF_ROOMS,NUMBER_OF_ROOMS))

    for sample in test_db:
        room = traverse_tree(sample, trained_tree)
        confusion_matrix[int(sample[LABEL_COLUMN]) - 1, int(room) - 1] += 1

    correct = np.trace(confusion_matrix)
    all_elements = np.sum(confusion_matrix)
    accuracy = correct / all_elements

    precision_arr = np.zeros(NUMBER_OF_ROOMS)
    recall_arr = np.zeros(NUMBER_OF_ROOMS)
    f1_arr = np.zeros(NUMBER_OF_ROOMS)
    for room in range(NUMBER_OF_ROOMS):
        precision = confusion_matrix[room, room] / np.sum(confusion_matrix[:, room])
        precision_arr[room] = precision
        recall = confusion_matrix[room,room] / np.sum(confusion_matrix[room,:])
        recall_arr[room] = recall
        f1_arr[room] = (2 * precision * recall)/(precision + recall)

    return (confusion_matrix, precision_arr, recall_arr, f1_arr)

# Traverse a trained decision tree to find which class a given sample is predicted as
def traverse_tree(sample, trained_tree):
    if "class" in trained_tree.keys():
        return trained_tree["class"]
    elif sample[trained_tree["attribute"]] < trained_tree["value"]:
        return traverse_tree(sample, trained_tree["left"])
    else:
        return traverse_tree(sample, trained_tree["right"])


if __name__ == "__main__":
    if sys.argv[1] == None:
        print("dataset filepath needed")
    else:
        dataset = load_dataset(sys.argv[1])
        (trained_tree, depth) = decision_tree_learning(dataset, 1)
        plt.figure(figsize=(17,8))
        visualise_tree(trained_tree)
        plt.axis('off')
        plt.show()
        # print(node)
        # (avg_confusion_matrix, avg_precision, avg_recall, avg_f1) = cross_validate(dataset)
        # print(avg_confusion_matrix)
        # print(avg_precision)
        # print(avg_recall)
        # print(avg_f1)

    