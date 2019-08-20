"""
Author: Prabhakar Gadupudi
Creation Date: Feb 17, 2019
USAGE: 
    python nltk_chatbot.py train (to train the model)
    python nltk_chatbot.py
Description: Users can modify the intents.json (training) data and we can train the model dynamically.
"""
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from datetime import datetime
class TRAIN_DYNAMIC_MODEL:
    def __init__(self):
        self.training_deep_level = 1000

    def read_intents(self, intents_file):
        """Read intents.json file and convert into JSON format"""
        with open(intents_file) as file:
            data = json.load(file)
        return data
    
    def make_data_picke_file(self, data_pickle_file, words_collection, labels_tags, training, output):
        with open(data_pickle_file, "wb") as f:
            pickle.dump((words_collection, labels_tags, training, output), f)

    def train(self, intents_file, data_pickle_file, model_tflearn_file):    
        print ("#"*100)
        st_time = datetime.now()
        print ("Started training the model")
        print ("#"*100)

        words_collection = []
        labels_tags = []
        docs_axis_A = []
        docs_axis_B = []
        data = self.read_intents(intents_file)
        # with open(intents_file) as file:
        #     data = json.load(file)

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                pattern_questions = nltk.word_tokenize(pattern)
                words_collection.extend(pattern_questions)
                docs_axis_A.append(pattern_questions)
                docs_axis_B.append(intent["tag"])

            if intent["tag"] not in labels_tags:
                labels_tags.append(intent["tag"])

        words_collection = [stemmer.stem(w.lower()) for w in words_collection if w != "?"]
        words_collection = sorted(list(set(words_collection)))

        labels_tags = sorted(labels_tags)

        training = []
        output = []

        output_empty = [0 for _ in range(len(labels_tags))]

        for index_val, doc in enumerate(docs_axis_A):
            bag_of_words = []
            pattern_questions = [stemmer.stem(w.lower()) for w in doc]
            for w in words_collection:
                if w in pattern_questions:
                    bag_of_words.append(1)
                else:
                    bag_of_words.append(0)
            output_row = output_empty[:]
            output_row[labels_tags.index(docs_axis_B[index_val])] = 1
            training.append(bag_of_words)
            output.append(output_row)

        training = numpy.array(training)
        output = numpy.array(output)
        self.make_data_picke_file(data_pickle_file, words_collection, labels_tags, training, output)
        # with open(data_pickle_file, "wb") as f:
        #     pickle.dump((words_collection, labels_tags, training, output), f)
        #Train and save the model
        tensorflow.reset_default_graph()
        neurals = tflearn.input_data(shape=[None, len(training[0])])
        neurals = tflearn.fully_connected(neurals, 8)
        neurals = tflearn.fully_connected(neurals, 8)
        neurals = tflearn.fully_connected(neurals, len(output[0]), activation="softmax")
        neurals = tflearn.regression(neurals)
        model = tflearn.DNN(neurals)
        model.fit(training, output, n_epoch=self.training_deep_level, batch_size=8, show_metric=True)
        model.save(model_tflearn_file)
        ed_time = datetime.now()    
        return [words_collection, labels_tags, training, output, st_time, ed_time]
