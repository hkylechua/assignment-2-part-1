# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
import sklearn
from sklearn.tree import export_graphviz
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle

import labels


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'data/all_labeled_data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,2], data[i,3], data[i,4]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:2],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

data = np.nan_to_num(data)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 100 Hz (sensor logger app); you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,1] - data[0,1])
sampling_rate = n_samples / time_elapsed_seconds

print("Sampling Rate: " + str(sampling_rate))

# TODO: list the class labels that you collected data for in the order of label_index (defined in labels.py)
class_names = labels.activity_labels

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []
feature_names = []
for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:-1]
    # print("window = ")
    # print(window)
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# TODO: split data into train and test datasets using 10-fold cross validation
cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

accuracies = []
precisions = []
recalls = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    tree.fit(X_train, Y_train)
    Y_pred = tree.predict(X_test)
    conf = confusion_matrix(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test, Y_pred, average='macro')
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    print(f"Fold {fold+1}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:")
    print(conf)
    print()

# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
mean_accuracy = np.mean(accuracies)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

print(f"Average Accuracy: {mean_accuracy}")
print(f"Average Precision: {mean_precision}")
print(f"Average Recall: {mean_recall}")


# TODO: train the decision tree classifier on entire dataset
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
tree.fit(X, Y)

# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
# export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)
export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
# print("saving classifier model...")
# with open('classifier.pickle', 'wb') as f:
#     pickle.dump(tree, f)
print("saving classifier model...")
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)