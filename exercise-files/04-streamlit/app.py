import streamlit as st
import openai
import os
from dotenv import load_dotenv
from query import query


# Load environment variables from .env file
load_dotenv()


# Get OpenAI API key from environment
API_KEY = os.getenv("OPENAI_API_KEY")
# Set OpenAI API key for API usage
openai.api_key = API_KEY
# Set the model engine to use for OpenAI
MODEL_ENGINE = "gpt-3.5-turbo"


# Set the Streamlit app title
st.title("🤖 Chatbot App")
# Create an empty placeholder for chat messages
chat_placeholder = st.empty()



# Function to initialize chat history in Streamlit session state
def init_chat_history():
    """Initialize chat history with a system message."""
    # If messages are not already in session state, add a system message
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "message": "You are a helpful assistant. Ask me anything!"}
        ]



# Function to start and manage the chatbot conversation
def start_chat():
    """Start the chatbot conversation."""
    # Display chat messages from history on app rerun
    with chat_placeholder.container():
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                # Use 'message' for system, 'content' for user/assistant
                if msg["role"] == "system":
                    continue
                else:
                    st.markdown(msg["content"])

    # Wait for user input in the chat input box
    if prompt := st.chat_input("What is up?"):
        # Add the user's message to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display the user's message in a chat bubble
        with st.chat_message("user"):
            st.markdown(prompt)

        response = query(prompt)
        
        # Placeholder for assistant's response (currently empty)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        # Add an empty user message to the chat history (likely a placeholder for future logic)
        st.session_state.messages.append(
            {"role": "assistant", "content": response["answer"]}
        )

        # Placeholder for updating assistant message (not implemented)



# Main entry point for the Streamlit app
if __name__ == "__main__":
    # Initialize chat history
    init_chat_history()
    # Start the chat interface
    start_chat()