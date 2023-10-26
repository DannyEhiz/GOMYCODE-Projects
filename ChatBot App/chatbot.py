# ---------------------------  CHATBOT WITH TRANSFER LEARNING  ------------------------------------------
import numpy as np
import time 
import os
import torch
!pip install transformers --q
pip3 install torch torchvision torchaudio
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\BIOLA\Desktop\gomycode\Projects\StreamLit Dev\ChatBot App\microsoft_model")
model = AutoModelForCausalLM.from_pretrained(r"C:\Users\BIOLA\Desktop\gomycode\Projects\StreamLit Dev\ChatBot App\microsoft_model")
# Save the model and tokenizer
tokenizer.save_pretrained()
model.save_pretrained()
# OR

# Get the current working directory
import os
cwd = os.getcwd()

# # Define the paths to save the model and tokenizer
model_path = os.path.join(cwd, r"C:\Users\BIOLA\Desktop\gomycode\Projects\StreamLit Dev\ChatBot App\ChatBot")
tokenizer_path = os.path.join(cwd, r"C:\Users\BIOLA\Desktop\gomycode\Projects\StreamLit Dev\ChatBot App\ChatBot")
tokenizer.save_pretrained(tokenizer_path) #---------------Save the tokenizer to the specified path
model.save_pretrained(model_path) # ---------------------- Save the model to the specified path

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


    # ----------------STREAMLIT IMPLEMENTATION --------------
    
st.title("CHATBOT MACHINE.")
st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

quit_sentences = ['bye', 'quit', 'exit', 'goodbye', 'talk to you later']

history = []

st.markdown('<h3>Quit Words are: Quit, Bye, Goodbye, Exit</h3>', unsafe_allow_html = True)

# Get the user's question    
user_input = st.text_input(f'Input your response')
if user_input not in quit_sentences:
    if st.button("Submit Your Response"):
        # Call the chatbot function with the question and display the response
        response = bot.bot_response(user_input)
        st.write("Chatbot: " + response)
