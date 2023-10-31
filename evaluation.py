import numpy as np

def cross_validate(data):
    k = 10
    np.random.shuffle(data)

    split = np.split(data, k)

    final_confusion_matrix = np.zeros(4,4)
    final_precision = 0
    final_recall = 0
    final_f1 = 0

    for i in range(10):
        data_copy = data
        test_data = data_copy.pop(i)
        training_data = np.concatenate(data_copy)
        (trained_tree, depth) = decision_tree_learning(training_data, 1)
        (confusion_matrix, precision, recall, f1) = evaluate_tree(test_data, trained_tree)
        final_confusion_matrix += confusion_matrix
        final_precision += precision
        final_recall += recall
        final_f1 += f1
    
    return (final_confusion_matrix, final_precision, final_recall, final_f1)


def evaluate_tree(test_db, trained_tree):
    confusion_matrix = np.zeros(4,4)

    for row in test_db:
        room = traverse_tree(row, trained_tree)
        confusion_matrix[row[7] - 1, room - 1] += 1

    correct = np.trace(confusion_matrix)
    all_elements = np.sum(confusion_matrix)
    accuracy = correct / all_elements

    precision = []
    recall = []
    for room in range(4):
        precision[room] += confusion_matrix[room, room] / np.sum(confusion_matrix[:, room])
        recall[room] += confusion_matrix[room,room] / np.sum(confusion_matrix[room,:])

    f1 = (2 * precision * recall)/(precision + recall)

    return (confusion_matrix, precision, recall, f1)

    

def traverse_tree(sample, trained_tree):
    if "class" in trained_tree.keys():
        return trained_tree["class"]
    elif sample[trained_tree["attribute"]] < trained_tree["value"]:
        traverse_tree(sample, trained_tree["left"])
    else:
        traverse_tree(sample, trained_tree["right"])