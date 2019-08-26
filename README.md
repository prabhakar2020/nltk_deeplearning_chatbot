# NLTK DeepLearning Chatbot
The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language

What it is?
-----
 > It is NLTK based simple chatbot and it will be more suitable for FAQ based question & Answers 

How do I enhance this chatbot for my work?
-----
 > Edit intent.json file and train the model as described below
 > and use it

Where do i get the reference tutorial for enhancements?
-----
 * > https://www.nltk.org/
 * > https://www.nltk.org/api/nltk.chat.html
 
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
 
Workflow
-----
![alt text](https://github.com/prabhakar2020/nltk_deeplearning_chatbot/blob/master/nltk_chatbot.png)

Info
-----
* User can find the input file on ** intents.json** on **data** folder and can fill the questions pattern and possible responses.
* Once training is completed, it will generate the **model.tflearn** file(s) on model directory
* Train  If we re-run the program with **train** mode, it will overwrite the existing model.
* If we re run the program with chatting mode, then it will load the existing model.tflearn file and will start the chatting application.
* For debugging, please refer **chatbot.log** file on your working directory

Usage
-----
 * > **You: hello**
 * > Hi there, how can I help?
 * > **You: i want to stop my cheque payment**
 * > To make a Stop Payment request on an issued Cheque, kindly follow the steps mentioned below: 1. Log in to your NetBanking account 2. Click
on the Stop Payment of Cheque option available under Request menu on the left hand side of the web page Select the Account Number (if more
than 1 account) Enter first/Start cheque number to the last/End cheque number, with a reason for stopping the payment. Click on Continue. On the next web page, you can verify and click on Confirm tab to confirm the stop payment request.
 * > **You: how do i open joint account**
 * > Yes, parents, spouse, child or sibling can be a joint applicant to an account. The joint applicant will need to submit a valid photo ID and address proof.
 * > **You: I want to reset my credit card password**
 * > To report lost/stolen credit card and reissue through website, follow these simple steps: Step 1 Log in to your account on sbicard.com Step 2 Click on â€˜Report Lost/Stolen Cardâ€™ under the â€˜Requestsâ€™ tab on the left
 * > **You: I want to add nominee in my account**
 * > When a card is lost or stolen or damaged, the same should be blocked immediately and reissued.  Request to block & reissue of card can be made through following channels:-Website sbicard.com
 * > **You: I lost my credit card what should i do**
 * > When a card is lost or stolen or damaged, the same should be blocked immediately and reissued.  Request to block & reissue of card can be made through following channels:-Website sbicard.com
 * > **You: I need home loan**
 * > For applying loan, please contact your nearest branch for more details
 * > **You: exit**
 * > ..... Thank you!!
 
**Note:**
----------
```
 This application is developed and designed for Python 3.6 ONLY. 
 It is highly recommended to run this program on python 3.6 version only.

 ```
