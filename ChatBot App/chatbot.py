# # Read in dataset
# import nltk
# import random
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity      
# import streamlit as st
# import warnings
# warnings.filterwarnings('ignore')

# # Download required NLTK data
# # nltk.download('stopwords')
# # nltk.download('punkt')
# # nltk.download('wordnet')

# with open("chatbot dataset copy.txt", 'r', encoding='utf8', errors='ignore') as file:
#     dataset = file.read()

# import pandas as pd
  
# # read the file with two delimiters separated by tabs
# # df = pd.read_csv(r'C:\Users\BIOLA\Desktop\gomycode\Projects\StreamLit Dev\ChatBot App\full_dialogs.txt', sep='\t', header=None, names=['question', 'answer', 'extra'])

# # drop the extra column         
# # df.drop('extra', axis=1, inplace=True)

# # # fill missing values with null
# # df.fillna(value=pd.np.nan, inplace=True)

# # Here, the Next step is tokenize our text dataset.
# # There are two types of tokenization:
#     # 1. Word Tokenization: This is  the process of breaking down a text or document into individual words or tokens.
#     # 2. Sent Tokenization: This is to break down the text data into individual sentences so that each sentence can be processed separately.
# # Lemmatization: The goal of lemmatization is to reduce a word to its canonical form so that variations of the same word can be treated as the same token
# # For example, the word "jumped" may be lemmatized to "jump", and the word "walking" may be lemmatized to "walk".
# # By reducing words to their base forms, lemmatization can help to simplify text data and reduce the number of unique tokens that need to be analyzed or processed.

# # We Therefore Preprocess our text dataset by TOKENIZING and Lemmatizing them
# sent_tokens = nltk.sent_tokenize(dataset)
# word_tokens = nltk.word_tokenize(dataset)
# lemmatizer = nltk.stem.WordNetLemmatizer()


# def preprocess(sentence):
#     return [lemmatizer.lemmatize(sentence.lower()) for sentence in sentence if sentence.isalnum()]
# # The code above does the following:
#     # 1. Turns the word to a lower case (.lower())
#     # 2. Checks if it's all alphanumeric (.isalnum())
#     # 3. Lemmatizes the word if the word is turned is alphanumeric and lowercase.


# corpus = [" ".join(preprocess(nltk.word_tokenize(sentence))) for sentence in sent_tokens]
# # The code above does the following:
#     # 1. Runs the preprocess function created above on the sent_tokens list we created before.
#     # 2. Then joins all this words with a space


# # Vectorize corpus
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# # TDIDF is a numerical statistic used to evaluate how important a word is to a document in a collection or corpus. 
# # The TfidfVectorizer calculates the Tfidf values for each word in the corpus and uses them to create a matrix where each row represents a document and each column represents a word. 
# # The cell values in the matrix correspond to the importance of each word in each document.

# # Define chatbot function
# def chatbot_response(user_input):
#     # Preprocess user input
#     user_input = " ".join(preprocess(nltk.word_tokenize(user_input)))

#     # Vectorize user input
#     user_vector = vectorizer.transform([user_input])

#     # Calculate cosine similarity between user input and corpus
#     similarities = cosine_similarity(user_vector, X)

#     # Get index of most similar sentence
#     idx = similarities.argmax()

#     # Return corresponding sentence from corpus
#     return sent_tokens[idx]

# # Run chatbot
# print("Welcome to the chatbot! How can I help you today?")

# while True:
#     user_input = input("> ")
#     if user_input.lower() == 'quit':
#         break
#     response = chatbot_response(user_input)
#     print(response)


# # ---------------------------------- STREAMLIT IMPLEMENTATION --------------------------------------
# import streamlit as st


# st.title("CHATBOT MACHINE.")
# st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

# quit_sentences = ['quit', 'bye', 'Goodbye', 'exit']

# history = []

# st.markdown('<h3>Quit Words are: Quit, Bye, Goodbye, Exit</h3>', unsafe_allow_html = True)

# # Get the user's question    
# user_input = st.text_input(f'Input your response')
# if user_input not in quit_sentences:
#     if st.button("Submit"):
#         # Call the chatbot function with the question and display the response
#         response = chatbot_response(user_input)
#         st.write("Chatbot: " + response)

#         # Create a history for the chat
#         history.append(('User: ', user_input))
#         history.append(('Bot: ', chatbot_response(user_input)))
# else:
#     st.write('Bye')

# st.markdown('<hr><hr>', unsafe_allow_html= True)
# st.subheader('Chat History')

# chat_history_str = '\n'.join([f'{sender}: {message}' for sender, message in history])

# st.text_area('Conversation', value=chat_history_str, height=300)



















# ---------------------------  CHATBOT WITH TRANSFER LEARNING  ------------------------------------------

# !pip install transformers --q
# pip3 install torch torchvision torchaudio
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\BIOLA\Desktop\gomycode\Projects\StreamLit Dev\ChatBot App\microsoft_model")
model = AutoModelForCausalLM.from_pretrained(r"C:\Users\BIOLA\Desktop\gomycode\Projects\StreamLit Dev\ChatBot App\microsoft_model")
# Save the model and tokenizer
# tokenizer.save_pretrained()
# model.save_pretrained()
# OR

# Get the current working directory
# import os
# cwd = os.getcwd()

# # Define the paths to save the model and tokenizer
# model_path = os.path.join(cwd, r"C:\Users\BIOLA\Desktop\gomycode\Projects\StreamLit Dev\ChatBot App\ChatBot")
# tokenizer_path = os.path.join(cwd, r"C:\Users\BIOLA\Desktop\gomycode\Projects\StreamLit Dev\ChatBot App\ChatBot")
# tokenizer.save_pretrained(tokenizer_path) #---------------Save the tokenizer to the specified path
# model.save_pretrained(model_path) # ---------------------- Save the model to the specified path

import numpy as np
import time 
import os
import torch

class ChatBot():
    # initialize
    def __init__(self):
        # once chat starts, the history will be stored for chat continuity
        self.chat_history_ids = None
        # make input ids global to use them anywhere within the object
        self.bot_input_ids = None
        # a flag to check whether to end the conversation
        self.end_chat = False
        # greet while starting
        self.welcome()
        
    def welcome(self):
        print("Initializing ChatBot ...")
        print('Type "bye" or "quit" or "exit" to end chat \n')

        # Greet and introduce
        greeting = np.random.choice([
            "Welcome, I am ChatBot, here for your kind service",
            "Hey, Great day! I am your virtual assistant",
            "Hello, it's my pleasure meeting you",
            "Hi, I am a ChatBot. Let's chat!"
        ])
        print("ChatBot >>  " + greeting)
        
    def user_input(self):
        # receive input from user
        text = input("User    >> ")
        # end conversation if user wishes so
        if text.lower().strip() in ['bye', 'quit', 'exit', 'goodbye', 'talk to you later']:
            # turn flag on 
            self.end_chat=True
            # a closing comment
            print('ChatBot >>  See you soon! Bye!')
            time.sleep(1)
            print('\nQuitting ChatBot ...')
        else:
            # continue chat, preprocess input text
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            self.new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, \
                                                       return_tensors='pt')

    def bot_response(self):
        # append the new user input tokens to the chat history
        # if chat has already begun
        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1) 
        else:
            # if first entry, initialize bot_input_ids
            self.bot_input_ids = self.new_user_input_ids
        
        # define the new chat_history_ids based on the preceding chats
        # generated a response while limiting the total chat history to 1000 tokens, 
        self.chat_history_ids = model.generate(self.bot_input_ids, max_length=1000, \
                                               pad_token_id=tokenizer.eos_token_id)
            
        # last ouput tokens from bot
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0], \
                               skip_special_tokens=True)
        # in case, bot fails to answer
        if response == "":
            response = self.random_response()
        # print bot response
        print('ChatBot >>  '+ response)
        
    # in case there is no response from model
    def random_response(self):
        i = -1
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], \
                               skip_special_tokens=True)
        # iterate over history backwards to find the last token
        while response == '':
            i = i-1
            response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], \
                               skip_special_tokens=True)
        # if it is a question, answer suitably
        if response.strip() == '?':
            reply = np.random.choice(["I don't know", 
                                     "I am not sure"])
        # not a question? answer suitably
        else:
            reply = np.random.choice(["Great", 
                                      "Fine. What's up?", 
                                      "Okay"
                                     ])
        return reply

# build a ChatBot object
bot = ChatBot()
# start chatting
while True:
# receive user input
    bot.user_input()
    # check whether to end chat
    if bot.end_chat:
        break
    # output bot response
    bot.bot_response()


#     # ----------------STREAMLIT IMPLEMENTATION --------------
    
# st.title("CHATBOT MACHINE.")
# st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

# quit_sentences = ['bye', 'quit', 'exit', 'goodbye', 'talk to you later']

# history = []

# st.markdown('<h3>Quit Words are: Quit, Bye, Goodbye, Exit</h3>', unsafe_allow_html = True)

# # Get the user's question    
# user_input = st.text_input(f'Input your response')
# if user_input not in quit_sentences:
#     if st.button("Submit Your Response"):
#         # Call the chatbot function with the question and display the response
#         response = bot.bot_response(user_input)
#         st.write("Chatbot: " + response)
