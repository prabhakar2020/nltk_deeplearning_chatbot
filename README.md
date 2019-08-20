# NLTK DeepLearning Chatbot
The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language

Installation Steps
-----
 * > git clone git@github.com:prabhakar2020/nltk_deeplearning_chatbot.git
 * > cd nltk_deeplearning_chatbot
 * > pip install -r requirements.txt 
(if you face any permission related errors. Please use **pip install -r requirements.txt --user**)

Execution steps
-----
 * For first run, it is expected to train the model.  Use this command to train the model using commandline argument **train**
 > **python nltk_chatbot.py train** 
 
 * You can modify the training data on **intents.json** file and train the model as many as you required.
 * Once training is completed, use this command to test/run the chatting application. 
 > **python nltk_chatbot.py**
 
 
Info
-----
* User can find the input file on ** intents.json** on **data** folder and can fill the questions pattern and possible responses.
* Once training is completed, it will generate the **model.tflearn** file(s) on model directory
* Train  If we re-run the program with **train** mode, it will overwrite the existing model.
* If we re run the program with chatting mode, then it will load the existing model.tflearn file and will start the chatting application.
*  For debugging, please refer **chatbot.log** file on your working directory
 
 
**Note:**
----------
```
 This application is developed and designed for Python 3.6 ONLY. 
 It is highly recommended to run this program on python 3.6 version only.

 ```
