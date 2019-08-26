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
from datetime import datetime
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os, sys

class NLTK_DEEPLEARNING_CHATBOT:
    def __init__(self):
        self.training_deep_level = 1000
        self.intents_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data", "intents.json")        
        self.logger_file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),"chatbot.log")
        self.data_pickle_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"model", "data.pickle")
        self.model_tflearn_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"model","model.train")
        self.data = {}
        self.words = []
        self.labels = []
        self.training = []
        self.output = []
        self.model = None
        self.pre_checks()

    def logger(self, msg=''):
        """Log entries"""
        with open(self.logger_file_name, 'a') as fd:
            fd.write("\n"+str(str(datetime.now()))+": "+str(msg))
    def pre_checks(self):
        self.logger("Pre validation checks")
        
        for i in [self.logger_file_name, self.data_pickle_file, self.model_tflearn_file]:
            if not os.path.isdir(os.path.dirname(i)):
                self.logger("Creating folder -"+str(os.path.dirname(i)))
                os.makedirs(os.path.dirname(i))

        if str(sys.version_info).find("major=3") > -1 and str(sys.version_info).find("minor=6") > -1:
            self.logger("Validation checks PASSED")
            return True
        else:
            print ("Python 3.6 version is recommended to run this program")
            self.logger("Validation checks FAILED")
            self.logger("Python 3.6 version is recommended to run this program")
            return False

    def initialize(self):
        if self.pre_checks():
            self.logger("Initializing Chatbot")
            self.read_intents()
            self.load_data_pickle()
            self.load_neurals()
            self.start_chat()

    def read_intents(self):
        """Read intents.json file and convert into JSON format"""
        self.logger("Started Reading Intents")
        with open(self.intents_file) as file:
            self.data = json.load(file)

    def load_data_pickle(self):
        if os.path.isfile(self.data_pickle_file):
            self.logger("Loading Pickle data")
            with open(self.data_pickle_file, "rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
        else:
            self.logger("Data pickle not found. Initializing training")
            from train_model import TRAIN_DYNAMIC_MODEL
            self.words, self.labels, self.training, self.output, st_time, ed_time = TRAIN_DYNAMIC_MODEL().train(self.intents_file, self.data_pickle_file, self.model_tflearn_file)
            print ("*"*100)
            print ("Started the Training at -",st_time)
            print ("Completed the training  -",ed_time)
            print ("*"*100)
            self.logger("Started the Training at -"+str(st_time))
            self.logger("Completed the training  -"+str(ed_time))
    def load_neurals(self):
        self.logger("Loading neurals")
        tensorflow.reset_default_graph()

        neurals = tflearn.input_data(shape=[None, len(self.training[0])])
        neurals = tflearn.fully_connected(neurals, 8)
        neurals = tflearn.fully_connected(neurals, 8)
        neurals = tflearn.fully_connected(neurals, len(self.output[0]), activation="softmax")
        neurals = tflearn.regression(neurals)
        self.model = tflearn.DNN(neurals)
        try:
            self.model.load(self.model_tflearn_file)
            self.logger("Loaded existing Model")
        except:
            self.logger("Saving new model after Training")
            self.model.fit(self.training, self.output, n_epoch=self.training_deep_level, batch_size=8, show_metric=True)
            self.model.save(self.model_tflearn_file)

    def get_bag_of_words(self, input_val, words):
        self.logger("Fetching bag of words")
        bag_of_words = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(input_val)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag_of_words[i] = 1
        self.logger("Bag of words :"+str(bag_of_words))
        return numpy.array(bag_of_words)


    def start_chat(self):
        print("Start chatting with the chatbot (type quit/ exit to stop)!")
        self.logger("All labels :"+str(self.labels))
        while True:
            # Get Input from users
            inp = input("You: ")
            if inp.lower() in  ["quit","exit"]:
                print ("..... Thank you!!")
                self.logger("..... Thank you!!")
                break
            self.logger("USER INPUT :"+str(inp))
            # Get bag of words for given inputs and predict from model
            results = self.model.predict([self.get_bag_of_words(inp, self.words)])
            results_index = numpy.argmax(results)
            self.logger("results :"+str(results))
            self.logger("results_index :"+str(results_index))
            # Fetch nearest possible tag/key from intents.json
            tag = self.labels[results_index]
            self.logger("tag :"+str(tag))            
            for tg in self.data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            # Fetch response from intents.json based on the prediction
            print(random.choice(responses))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if str(sys.argv[1]).strip().lower().find('train') > -1:
            obj = NLTK_DEEPLEARNING_CHATBOT()
            from train_model import TRAIN_DYNAMIC_MODEL
            words, labels, training, output, st_time, ed_time = TRAIN_DYNAMIC_MODEL().train(obj.intents_file, obj.data_pickle_file, obj.model_tflearn_file)
            print ("*"*100)
            print ("Started the Training at         -",st_time)
            print ("Completed the training          -",ed_time)
            print ("Total training time in minutes  -",round((ed_time-st_time).total_seconds()/60.0,2))
            print ("*"*100)
            obj.logger("Started the Training at -"+str(st_time))
            obj.logger("Completed the training  -"+str(ed_time))
            obj.logger("Total training time in minutes  :"+str(round((ed_time-st_time).total_seconds()/60.0,2)))
    else:
        NLTK_DEEPLEARNING_CHATBOT().initialize()
