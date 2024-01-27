from flask import Flask, render_template, request, jsonify
from itertools import zip_longest
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

app = Flask(__name__)

# Initialize the ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo"
)

# Global variable to store conversation history
conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')  # Render your HTML file

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_input = data['message']
    custom_prompt = data.get('prompt', "")  # Get the custom prompt if provided
    response = generate_response(user_input, custom_prompt)
    return jsonify({'response': response})

def generate_response(user_input, custom_prompt):
    # Use the custom prompt if provided, else use a default prompt
    prompt_message = custom_prompt if custom_prompt else "You are a helpful AI assistant talking with a human. If you do not know an answer, just say 'I don't know', do not make up an answer."
    # Add user input to conversation history
    conversation_history.append(HumanMessage(content=user_input))

    # Assuming only the last user input and AI response are relevant
    zipped_messages = [
        SystemMessage(content=prompt_message)
    ]
    zipped_messages.extend(conversation_history)

    # Generate response using the chat model
    ai_response = chat(zipped_messages)

    # Add AI response to conversation history
    conversation_history.append(AIMessage(content=ai_response.content))

    return ai_response.content

if __name__ == '__main__':
    app.run(debug=True)
