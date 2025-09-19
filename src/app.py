import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os

# Set page configuration
st.set_page_config(
    page_title="Mistral Chatbot",
    page_icon="🤖",
    layout="wide"
)


def initialize_chatbot():
    """Initialize the chatbot with Mistral model and conversation chain"""

    # Initialize the Mistral model
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.7,
        max_tokens=500
    )

    # Create conversation memory
    memory = ConversationBufferMemory(return_messages=True)

    # Define a custom prompt template
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""You are a helpful AI assistant. Have a natural conversation with the user.

Previous conversation:
{history}

Current input: {input}
Assistant:"""
    )

    # Create the conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )

    return conversation


def main():
    st.title("🤖 Mistral AI Chatbot")
    st.markdown("### Chat with Mistral AI using LangChain")

    # Sidebar for configuration
    st.sidebar.title("Configuration")

    # API Key input
    api_key = st.sidebar.text_input(
        "Enter your Mistral API Key:",
        type="password",
        help="Get your API key from https://console.mistral.ai/"
    )

    # Model selection
    model_options = [
        "mistral-small-latest",
        "mistral-medium-latest",
        "mistral-large-latest"
    ]
    selected_model = st.sidebar.selectbox("Select Model:", model_options)

    # Temperature slider
    temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)

    # Max tokens slider
    max_tokens = st.sidebar.slider("Max Tokens:", 100, 1000, 500, 50)

    # Clear conversation button
    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    # Check if API key is provided
    if not api_key:
        st.warning(
            "Please enter your Mistral API key in the sidebar to start chatting.")
        return

    # Set the API key in environment variable
    os.environ["MISTRAL_API_KEY"] = api_key

    # Initialize the chatbot if not already done
    if "conversation" not in st.session_state:
        try:
            # Update model parameters
            llm = ChatMistralAI(
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            memory = ConversationBufferMemory(return_messages=True)

            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template="""You are a helpful AI assistant. Have a natural conversation with the user.

Previous conversation:
{history}

Current input: {input}
Assistant:"""
            )

            st.session_state.conversation = ConversationChain(
                llm=llm,
                memory=memory,
                prompt=prompt,
                verbose=False
            )

        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            return

    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation.predict(
                        input=prompt)
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})

                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
