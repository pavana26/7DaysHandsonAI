# Import necessary libraries
# pip install SpeechRecognition
# pip install pyttsx3
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os

# Initialize the speech engine for text-to-speech
engine = pyttsx3.init()

# Function to make the assistant speak
def speak(text):
    engine.say(text)
    engine.runAndWait() 

# Function to take  voice commands from the user
def take_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio, language='en-in')
        print(f"User said: {command}\n")
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that. Please say that again.")
        return "None"
    except sr.RequestError:
        print("Sorry, Network error.")
        return "None"
    return command.lower()

# Function to respond to different commands
def respond(command):
    if 'hello' in command or 'hi' in command:
        speak("Hello! How can I assist you today?")
    elif 'time' in command:
        now = datetime.datetime.now().strftime("%H:%M")
        speak(f"The current time is {now}")
    elif 'open google' in command:
        webbrowser.open("https://www.google.com")
        speak("Opening Google")
    elif 'open youtube' in command:
        webbrowser.open("https://www.youtube.com")
        speak("Opening YouTube")
    elif 'play music' in command:
        music_dir = "C:\\Users\\PavanaBhat\\Music"  # Change to your music directory
        songs = os.listdir(music_dir)
        if songs:
            os.startfile(os.path.join(music_dir, songs[0]))
            speak("Playing music")
        else:
            speak("No music files found in the directory.")
    elif 'bye' in command or 'exit' in command or 'quit' in command:
        speak("Goodbye! Have a great day!")
        exit()
    else:
        speak("Sorry, I didn't understand that command.")

# Main function to run the assistant
def run_assistant():
    speak("Hello! I am your voice assistant. How can I help you?")
    while True:
        command = take_command()
        if command != "None":
            respond(command)
# start the assistant
run_assistant()