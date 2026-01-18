# Import the regular expression module to handle pattern matching
import re

# A dictionary that map keywords to prefined responses
respsonses = {
    "hello": "Hi there! How can I assist you today?",
    "hi": "Hello! How can I help you today?",
    "how are you": "I'm just a bot, but I'm doing great! How about you?",
    "what is your name": "I'm ChatBot, your virtual assistant.",
    "help": "Sure! I'm here to help. What do you need assistance with?",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're welcome! I'm here to help.",
    "default": "I'm sorry, I didn't understand that. Can you please rephrase?"
}
# Function to find the appropriate response based on the user's input
def chatbot_response(user_input):
    # Convert the input to lowercase for case-insensitive matching
    user_input = user_input.lower()
    
    # Check for each keyword in the user's input
    for keyword in respsonses.keys():
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_input):
            return respsonses[keyword]
    
    # Return the default response if no keywords matched
    return respsonses["default"]

# Main function to run the chatbot
def chatbot():
    print("Welcome to ChatBot! I am here to assis you.Type 'bye' to end the conversation.")
    while True:
        user_input = input("You: ")
        # If user types 'bye', exit the chatbot
        if user_input.lower() == 'bye':
            print("ChatBot: Goodbye! Have a great day!")
            break
        # Get chatbot's response based on user input
        response = chatbot_response(user_input)
        print(f"ChatBot: {response}")

# Run the chatbot
chatbot()
