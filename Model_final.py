import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pickle
import csv
import random
from sklearn.svm import SVC, LinearSVC # Have inbuilt decision function
from sklearn.neighbors import KNeighborsClassifier # Can use predicted probabilities
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
from sklearn import metrics

def get_data(filename):
    data = pickle.load(open(filename,  'rb'))
    return data

def count_hate_speech(data):
    count = 0
    for i in range(0,len(data)):
        if (data[i]["Label"] == 1):
            count += 1

    return count


def initialise_training_array(data):
    X = []
    Y = []

    for i in range(0,len(data)):
        X.append(data[i]["Word Vector"])
        Y.append(data[i]["Label"])

    return {'X':X, 'Y':Y}


def initialise_unlabel_array(unlabel_data):
    unlabel_X = []

    for i in range(0,len(unlabel_data)):
        unlabel_X.append(unlabel_data[i]["Word Vector"])

    return unlabel_X

def apply_training (unlabel_data, train_X, train_Y, test_X, test_Y, clf, filename):
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    # Fit the classifier on the data
    if clf=="random_forest":
        clf=RandomForestClassifier(n_estimators=100,oob_score=True)
        clf.fit(train_X,train_Y)
    elif clf=="knn":
        clf=KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_X,train_Y)
    elif clf == "nb":
        clf = GaussianNB()
        clf.fit(train_X, train_Y)

    #clf.fit(train_X, train_Y)

    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        #writer.writerows([["Text", "Label"]])
        writer.writerows([["index","Label","proba_0","proba_1","probability","Text"]])
        #tambahan

        # Use the trained model to predict the labels for the unlabeled dataset
        for i in range(0,len(unlabel_data)):
            unlabel_index=unlabel_data[i]["index"]
            unlabel_sentence = unlabel_data[i]["Actual sentence"]
            unlabel_word_vector = np.array(unlabel_data[i]["Word Vector"]).reshape(1,-1)
            # print(unlabel_word_vector)
            pred_label = clf.predict(unlabel_word_vector)
            predict_proba_0=clf.predict_proba(unlabel_word_vector)[:,0]
            predict_proba_1=clf.predict_proba(unlabel_word_vector)[:,1]
            predict_proba=clf.predict_proba(unlabel_word_vector)[:,1]
            if pred_label==[0]:
                probability=predict_proba_0
            elif pred_label==[1]:
                probability=predict_proba_1
            #writer.writerows([[unlabel_sentence , pred_label]])
            writer.writerows([[unlabel_index,pred_label,predict_proba_0,predict_proba_1,
                               probability,unlabel_sentence]])

    # Evaluate on the testing set
    predict_Y = clf.predict(test_X)
    # print(predict_Y)

    # Get the metrics
    accuracy = metrics.accuracy_score(test_Y, predict_Y)
    return {'predict_Y': predict_Y, 'accuracy': accuracy }

# Initialise the number of questions for confidence score/random selection
def get_random_questions( num_questions, clf, decision_function, unlabel_X, unlabel_data):
    question_samples = []
    if decision_function:
        # ---------- Include this part for those classifiers that have a decision function
        # Get the confidences for unlabeled data
        confidences = np.abs(clf.decision_function(unlabel_X))
        # print(confidences)
        # Sort the confidence values
        sorted_confidences = np.argsort(confidences)
        # print(sorted_confidences)

        #select top k low confidence unlabeled samples
        low_confidence_samples = sorted_confidences[0:num_questions*10]

        #select top k high confidence unlabeled samples
        high_confidence_samples = sorted_confidences[-num_questions*10:]

        question_samples.extend(low_confidence_samples.tolist())
        question_samples.extend(high_confidence_samples.tolist())

        # ----------
    else:
        # ---------- Include this part for those classifiers that don't have decision function
        # Select random indices in the case of certain algorithms that don't have decision function
        question_samples.extend(random.sample(range(len(unlabel_data)), num_questions * 20))

    return question_samples


def evaluate(question_samples, unlabel_data, train_X, train_Y, unlabel_X):
    #print("\nQuestion Samples : " + str(question_samples))
    #print("\nPlease provide the label 1 for hate and 0 for no hate for the following texts : ")
    for i in question_samples:
        word_vector = np.array(unlabel_data[i['index']]["Word Vector"])
        train_X = np.vstack([train_X,word_vector])
        train_Y = np.hstack([train_Y,int(i['label'])])

    question_indexes = [ i['index'] for i in question_samples]

    unlabel_data = [i for i in unlabel_data if unlabel_data.index(i) not in question_indexes]
    unlabel_X = [i for i in unlabel_X if unlabel_X.index(i) not in question_indexes]

    return {'train_X': train_X, 'train_Y': train_Y, 'unlabel_data': unlabel_data, 'unlabel_X': unlabel_X}

# this code taken from question_batch_mode
# {{list_question.append(question_ids[order])}}