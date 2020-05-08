import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pickle
import sys
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from gensim.models import Word2Vec
import gensim
import gensim.models.keyedvectors as word2vec
from gensim.models.wrappers import FastText

def extract_sentences (filename):
    """
    Loads a CSV file and reads the sentences from it

    :param: filename: A string that contains the path to the CSV file
    """
    try:
        file_data = pd.read_csv(filename)
    except:
        print("Unable to open file:")
        sys.exit(1)
    data_frame = pd.DataFrame(file_data)

    sentences = []

    for i in range(0,len(data_frame)):    # Extract from the train dataframe
        t = str(data_frame.iloc[i]["text"])
        t = nltk.tokenize.word_tokenize(t)
        if(len(t) >= 3):
            sentences.append(t)

    return {'data_frame':data_frame, 'sentences':sentences}

def generate_model (sentences):
    """
    Using all the sentences from the CSV file, get the training model

    :param: sentences: The full list of sentences received from all the three files
    """
    # train model using the sentence list as the corpus
    model = Word2Vec(sentences, min_count=1, size=1)
    #model=gensim.models.FastText(sentences,min_count=1,size=1)

    #model=word2vec.KeyedVectors.load_word2vec_format('../../model/GoogleNews-vectors-negative300.bin.gz',binary=True)
    #model=gensim.models.Word2Vec(sentences,window=10, min_count=2, workers=10)

    #model=FastText.load_fasttext_format('../../model/cc.en.300.bin')
    # save the model
    model.save('model.bin')
    return model

def w2v_sentence(sentence, train_data, model, output_filename):
    """
    Creates word to vector representation of the sentences training data

    :param: sentence: The sentences to which to apply the transformation
    :param: train_data: Training data, received from the extract_sentences function
    :param: output_filename: The path of the filename to which the output is written
    """
    sentence_l_d = []

    for i in range(0,len(sentence)):
        w2v = []
        for j in range(0,len(sentence[i])):
            w2v.append(model[sentence[i][j]])
        try:
            sentence_l_d.append({
                "index" :train_data.iloc[i]["index"],
                "Actual sentence" : train_data.iloc[i]["text"],
                "Tokenized sentence" : sentence[i],
                "Word Vector" : [float(sum(w2v))/len(w2v)],
                "Label" : train_data.iloc[i]["label"]
            })
        except:
            pass

    with open(output_filename, 'wb') as fp:
        pickle.dump(sentence_l_d, fp)

    return sentence_l_d

def w2v_unlabel(unlabel_sentence, df_unlabel, model, output_filename):
    unlabel_l_d = []

    for i in range(0,len(unlabel_sentence)):
        w2v = []
        for j in range(0,len(unlabel_sentence[i])):
            w2v.append(model[unlabel_sentence[i][j]])
        try:
            unlabel_l_d.append({
                "index": df_unlabel.iloc[i]["index"],
                "Actual sentence" : df_unlabel.iloc[i]["text"],
                "Tokenized sentence" : unlabel_sentence[i],
                "Word Vector" : [float(sum(w2v))/len(w2v)]
            })
        except:
            pass

    with open(output_filename, 'wb') as fp:
        pickle.dump(unlabel_l_d, fp)

    return unlabel_l_d