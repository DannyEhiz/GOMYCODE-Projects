import streamlit as st
import speech_recognition as sr
import pyttsx3
import time


# ask for username
# ask for what means of the reply the user wants to be answered
# ask for a what user wants
# a model that replies what the user asked
# a chatbot or talkback to reply

st.markdown("<h1 style = 'bottom_margin: 0rem; text-align: centre; color: #B2A4FF'>WEATHER APP</h1> ", unsafe_allow_html = True)
st.markdown("<p style = 'bottom_margin: 5rem; text-align: centre; color: #FFB4B4'>Speech Recognition, Chatbot, and TalkBack Powered Weather App</p>", unsafe_allow_html = True)
st.markdown('<br>', unsafe_allow_html= True)

st.image('pngwing.com (28).png', width = 400)
# st.write('Usage Direction: ')


username = st.text_input('Enter Your Name: ')
if st.button('submit'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")

side_img1 = st.sidebar.image('pngwing.com (29).png', caption = username, width = 200)

st.sidebar.write(f"Welcome {username}. Pls choose your input type")
select_reply = st.sidebar.selectbox('How Do You Want Your Reply? ', ['Talkback', 'Chat'])

if select_reply == 'Talkback':
    st.success('You have chosen Talkback Mode')
else: st.success('You have chosen chatbot mode')


st.markdown('<hr>', unsafe_allow_html= True)

# Create a function for Speech Trancription
def transcribe_speech():
    # Initialize recognizer class
    r = sr.Recognizer()

    # Reading Microphone as source
    with sr.Microphone() as source:

        # create a streamlit spinner that shows progress
        with st.spinner(text='Silence pls, Caliberating background noise.....'):
            time.sleep(3)

        r.adjust_for_ambient_noise(source, duration = 1) # ..... Adjust the sorround noise
        st.info("Speak now...")

        audio_text = r.listen(source) #........................ listen for speech and store in audio_text variable
        with st.spinner(text='Transcribing your voice to text'):
            time.sleep(2)

        try:
            # using Google speech recognition to recognise the audio
            text = r.recognize_google(audio_text)
            # print(f' Did you say {text} ?')
            return text
        except:
            return "Sorry, I did not get that."

# Create a function for text Talkback
def Text_Speaker(your_command):
    speaker_engine = pyttsx3.init() #................ initiate the talkback engine
    speaker_engine.say(your_command) #............... Speak the command
    speaker_engine.runAndWait() #.................... Run the engine




def main():
    global your_words_in_text
    # add a button to trigger speech recognition
    if st.button("Start Recording"):
        your_words_in_text = transcribe_speech()
        st.write("Transcription: ", your_words_in_text)
if __name__ == "__main__":
    main()

Text_Speaker(your_words_in_text)