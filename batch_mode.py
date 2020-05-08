import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pickle
import csv
import random
from sklearn.svm import SVC, LinearSVC  # Have inbuilt decision function
from sklearn.neighbors import KNeighborsClassifier  # Can use predicted probabilities
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
from sklearn import metrics

# Initialise the classifier
# Classifiers that have decision function available in sklearn
# clf = SVC()
# clf = LinearSVC()
# clf = SGDClassifier()
# clf = Perceptron()

# Classifiers that don't have decision function available in sklearn
# clf = KNeighborsClassifier()
# clf = BernoulliNB()
clf = RandomForestClassifier()

train_data = pickle.load(open('Outfiles/train_outfile', 'rb'))
test_data = pickle.load(open('Outfiles/test_outfile', 'rb'))
label_data = pickle.load(open('Outfiles/labeled_outfile', 'rb'))

print("\nLength of training data : " + str(len(train_data)))
print(train_data[0])
print("\nLength of testing data : " + str(len(test_data)))
print(test_data[0])
print("\nLength of labeled data : " + str(len(label_data)))
print(label_data[0])

train_count = 0
for i in range(0, len(train_data)):
    if (train_data[i]["Label"] == 1):
        train_count += 1

print("\nTotal hate speech occurrences in training data : " + str(train_count))
p_train = float(train_count) / len(train_data) * 100
print("Percentage of hate speech in training data : " + str(p_train))

test_count = 0
for i in range(0, len(test_data)):
    if (test_data[i]["Label"] == 1):
        test_count += 1

print("\nTrain hate speech occurrences in testing data : " + str(test_count))
p_test = float(test_count) / len(test_data) * 100
print("Percentage of hate speech in testing data : " + str(p_test))

train_X = []
train_Y = []
test_X = []
test_Y = []
label_X = []
label_Y = []

for i in range(0, len(train_data)):
    train_X.append(train_data[i]["Word Vector"])
    train_Y.append(train_data[i]["Label"])

for i in range(0, len(test_data)):
    test_X.append(test_data[i]["Word Vector"])
    test_Y.append(test_data[i]["Label"])

for i in range(0, len(label_data)):
    label_X.append(label_data[i]["Word Vector"])
    label_Y.append(label_data[i]["Actual Label"])

print("\nInitial length of training data : " + str(len(train_X)))
print(len(train_Y))
print("\nInitial length of testing data : " + str(len(test_X)))
print(len(test_Y))
print("\nInitial length of the labeled data : " + str(len(label_X)))
print(len(label_Y))

train_X = np.array(train_X)
train_Y = np.array(train_Y)
test_X = np.array(test_X)
test_Y = np.array(test_Y)

# Initialise the number of questions for confidence score/random selection
num_questions = 5

# Fit the classifier on the data
clf.fit(train_X, train_Y)

with open('Results_labeled.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows([["Text", "Actual Label", "Predicted Label"]])

    # Use the trained model to predict the labels for the labeled dataset
    for i in range(0, len(label_data)):
        label_sentence = label_data[i]["Actual sentence"]
        label_word_vector = np.array(label_data[i]["Word Vector"]).reshape(1, -1)
        # print(label_word_vector)

        actual_label = label_data[i]["Actual Label"]
        pred_label = clf.predict(label_word_vector)
        label_data[i]["Predicted Label"] = pred_label

        writer.writerows([[label_sentence, actual_label, pred_label]])

# Evaluate the performance on the labeled dataset
actual_label = []
predicted_label = []
for i in range(0, len(label_data)):
    actual_label.append(label_data[i]["Actual Label"])
    predicted_label.append(label_data[i]["Predicted Label"])

accuracy_label = metrics.accuracy_score(actual_label, predicted_label)
precision_label = metrics.precision_score(actual_label, predicted_label)
fscore_label = metrics.f1_score(actual_label, predicted_label)
print("\nInitial Accuracy on the labeled set : " + str(accuracy_label * 100))
print("Initial Precision on the labeled set : " + str(precision_label * 100))
print("Initial Fscore on the labeled set : " + str(fscore_label * 100))

# Evaluate the performance on the testing set
predict_Y = clf.predict(test_X)

accuracy_test = metrics.accuracy_score(test_Y, predict_Y)
precision_test = metrics.precision_score(test_Y, predict_Y)
fscore_test = metrics.f1_score(test_Y, predict_Y)
print("\nInitial Accuracy on the labeled set : " + str(accuracy_test * 100))
print("Initial Precision on the labeled set : " + str(precision_test * 100))
print("Initial Fscore on the labeled set : " + str(fscore_test * 100))


# Loop for active learning in Batch Mode
l = 0
while l < 10:

    # ---------- Include this part for those classifiers that have a decision function

    # # Get the confidences for labeled data
    # confidences = np.abs(clf.decision_function(label_X))
    # # print(confidences)

    # # Sort the confidence values
    # sorted_confidences = np.argsort(confidences)
    # # print(sorted_confidences)

    # question_samples = []

    # #select top k low confidence labeled samples
    # low_confidence_samples = sorted_confidences[0:num_questions]

    # #select top k high confidence labeled samples
    # high_confidence_samples = sorted_confidences[-num_questions:]

    # question_samples.extend(low_confidence_samples.tolist())
    # question_samples.extend(high_confidence_samples.tolist())

    # ----------

    # ---------- Include this part for those classifiers that don't have decision function

    # Select random indices in the case of certain algorithms that don't have decision function
    question_samples = random.sample(range(len(label_data)), num_questions * 2)

    # ----------

    print(
        "\n--------------------------------------------------------------------------------------------------------------------------------------------")
    print("\nQuestion Samples : " + str(question_samples))
    print("\nPlease provide one common label -> 1 for hate and 0 for no hate for the following texts : ")

    for i in question_samples:
        print(str(i) + " - " + label_data[i]["Actual sentence"])

    label = input("\nPlease enter your label : ")

    for i in question_samples:
        word_vector = np.array(label_data[i]["Word Vector"])
        train_X = np.vstack([train_X, word_vector])
        train_Y = np.hstack([train_Y, int(label)])

    label_data = [i for i in label_data if label_data.index(i) not in question_samples]
    label_X = [i for i in label_X if label_X.index(i) not in question_samples]

    print("\nNew shape of training data : " + str(train_X.shape))
    print("New length of labeled data : " + str(len(label_data)))

    # Fit the classifier on the data
    clf.fit(train_X, train_Y)

    # Evaluate on the testing set
    predict_Y = clf.predict(test_X)
    # print(predict_Y)

    accuracy = metrics.accuracy_score(test_Y, predict_Y)
    precision = metrics.precision_score(test_Y, predict_Y)
    fscore = metrics.f1_score(test_Y, predict_Y)

    print("\nAccuracy : " + str(accuracy * 100))
    print("Precision : " + str(precision * 100))
    print("Fscore : " + str(fscore * 100))

    l += 1