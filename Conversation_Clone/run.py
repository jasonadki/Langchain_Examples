from itertools import zip_longest
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv(find_dotenv())


# Set streamlit page configuration
st.set_page_config(page_title="Simple Chat Bot")
st.title("Simple Chat Bot")

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

# Initialize the ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo"
)

def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content="You are a helpful AI assistant talking with a human. If you do not know an answer, just say 'I don't know', do not make up an answer.")]

    # Add the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(content=human_msg))
        if ai_msg is not None:
            zipped_messages.append(AIMessage(content=ai_msg))

    return zipped_messages

def generate_response():
    """
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages = build_message_list()

    # Generate response using the chat model
    ai_response = chat(zipped_messages)

    return ai_response.content

# Input for new message using chat_input
if prompt := st.chat_input("Your question:"):
    # Update session state for user query
    st.session_state.past.append(prompt)

    # Generate and update session state for AI response
    output = generate_response()
    st.session_state.generated.append(output)

# Display messages using the new chat interface
for idx in range(len(st.session_state['past'])):
    # Display user message
    if idx < len(st.session_state['past']):
        with st.chat_message("user"):
            st.markdown(st.session_state['past'][idx])

    # Display AI response
    if idx < len(st.session_state['generated']):
        with st.chat_message("assistant"):
            st.markdown(st.session_state['generated'][idx])
