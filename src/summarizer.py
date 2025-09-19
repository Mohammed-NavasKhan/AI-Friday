import streamlit as st
import tempfile
import os
from typing import List, Optional
from io import BytesIO

# LangChain imports
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Configure Streamlit page
st.set_page_config(
    page_title="Document Summarizer Agent",
    page_icon="📄",
    layout="wide"
)


class DocumentSummarizerAgent:
    def __init__(self, mistral_api_key: str):
        self.mistral_api_key = mistral_api_key
        self.llm = ChatMistralAI(
            api_key=mistral_api_key,
            model="mistral-large-latest",
            temperature=0.1
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len
        )

    def load_document(self, uploaded_file) -> List[Document]:
        """Load and process uploaded document"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Load document based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == 'txt':
                loader = TextLoader(tmp_file_path)
            elif file_extension in ['docx', 'doc']:
                loader = Docx2txtLoader(tmp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()

            # Clean up temporary file
            os.unlink(tmp_file_path)

            return documents

        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return []

    def summarize_document(self, documents: List[Document], summary_type: str = "stuff") -> str:
        """Summarize the loaded documents"""
        if not documents:
            return "No documents to summarize."

        # Split documents into chunks if they're too long
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)

        # Create summarization chain
        if summary_type == "map_reduce" and len(all_chunks) > 1:
            # For longer documents, use map-reduce approach
            map_template = """
            Write a concise summary of the following text:
            {text}
            CONCISE SUMMARY:
            """
            map_prompt = PromptTemplate(
                template=map_template, input_variables=["text"])

            combine_template = """
            Write a comprehensive summary of the following summaries:
            {text}
            COMPREHENSIVE SUMMARY:
            """
            combine_prompt = PromptTemplate(
                template=combine_template, input_variables=["text"])

            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=False
            )
        else:
            # For shorter documents, use stuff approach
            template = """
            Please provide a comprehensive summary of the following document:
            
            {text}
            
            Summary should include:
            - Main topics and key points
            - Important details and findings
            - Conclusions or recommendations if any
            
            SUMMARY:
            """
            prompt = PromptTemplate(
                template=template, input_variables=["text"])

            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="stuff",
                prompt=prompt,
                verbose=False
            )

        try:
            summary = chain.run(all_chunks)
            return summary
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def extract_key_insights(self, documents: List[Document]) -> str:
        """Extract key insights from documents"""
        if not documents:
            return "No documents to analyze."

        # Combine all document content
        full_text = "\n".join([doc.page_content for doc in documents])

        # Truncate if too long
        if len(full_text) > 10000:
            full_text = full_text[:10000] + "..."

        prompt = f"""
        Analyze the following document and extract key insights:
        
        {full_text}
        
        Please provide:
        1. Top 5 key insights or findings
        2. Main themes or patterns
        3. Notable statistics or data points
        4. Important conclusions
        
        Key Insights:
        """

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error extracting insights: {str(e)}"

    def create_agent_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        def summarize_tool(query: str) -> str:
            """Tool for summarizing documents"""
            if 'current_documents' not in st.session_state or not st.session_state.current_documents:
                return "No documents loaded. Please upload a document first."

            return self.summarize_document(st.session_state.current_documents)

        def insights_tool(query: str) -> str:
            """Tool for extracting key insights"""
            if 'current_documents' not in st.session_state or not st.session_state.current_documents:
                return "No documents loaded. Please upload a document first."

            return self.extract_key_insights(st.session_state.current_documents)

        tools = [
            Tool(
                name="document_summarizer",
                description="Summarizes the uploaded document comprehensively",
                func=summarize_tool
            ),
            Tool(
                name="key_insights_extractor",
                description="Extracts key insights, themes, and important findings from the document",
                func=insights_tool
            )
        ]

        return tools

    def create_agent(self) -> AgentExecutor:
        """Create the document analysis agent"""
        tools = self.create_agent_tools()

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful document analysis assistant. You can:
            1. Summarize documents comprehensively
            2. Extract key insights and findings
            
            When users ask questions about documents, use the appropriate tools to provide accurate and helpful responses.
            Always be clear about what analysis you're performing and provide structured, easy-to-read responses."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        return agent_executor


def main():
    st.title("📄 Document Summarizer Agent")
    st.write("Upload a document and let the AI agent analyze it for you!")

    # Sidebar for API key
    st.sidebar.header("Configuration")
    mistral_api_key = st.sidebar.text_input("Mistral API Key", type="password")

    if not mistral_api_key:
        st.warning(
            "Please enter your Mistral API key in the sidebar to continue.")
        st.info("Get your API key from: https://console.mistral.ai/")
        return

    # Initialize the agent
    if 'agent' not in st.session_state:
        st.session_state.agent = DocumentSummarizerAgent(mistral_api_key)
        st.session_state.agent_executor = st.session_state.agent.create_agent()

    # File upload
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'docx'],
        help="Supported formats: PDF, TXT, DOCX"
    )

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        # Load document
        with st.spinner("Loading document..."):
            documents = st.session_state.agent.load_document(uploaded_file)
            st.session_state.current_documents = documents

        if documents:
            st.success(
                f"Document loaded successfully! ({len(documents)} pages)")

            # Quick actions
            st.header("🚀 Quick Actions")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📝 Generate Summary", use_container_width=True):
                    with st.spinner("Generating summary..."):
                        response = st.session_state.agent_executor.invoke({
                            "input": "Please provide a comprehensive summary of the uploaded document."
                        })
                        st.subheader("📋 Document Summary")
                        st.write(response['output'])

            with col2:
                if st.button("💡 Extract Key Insights", use_container_width=True):
                    with st.spinner("Extracting insights..."):
                        response = st.session_state.agent_executor.invoke({
                            "input": "Please extract the key insights and important findings from the document."
                        })
                        st.subheader("🔍 Key Insights")
                        st.write(response['output'])

            # Chat interface
            st.header("💬 Chat with Your Document")

            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # Chat input
            user_input = st.chat_input("Ask questions about your document...")

            if user_input:
                # Add user message to chat history
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input})

                with st.chat_message("user"):
                    st.write(user_input)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.agent_executor.invoke(
                                {"input": user_input})
                            assistant_response = response['output']
                            st.write(assistant_response)

                            # Add assistant response to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": assistant_response
                            })
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg
                            })

    # Usage instructions
    with st.expander("ℹ️ How to Use"):
        st.write("""
        1. **Enter your Mistral API Key** in the sidebar
        2. **Upload a document** (PDF, TXT, or DOCX)
        3. **Use Quick Actions** for instant summaries or insights
        4. **Chat with your document** by asking specific questions
        
        **Example questions:**
        - "What are the main conclusions of this document?"
        - "Can you highlight the key statistics mentioned?"
        - "What recommendations are provided?"
        - "Summarize the methodology used in this research"
        """)


if __name__ == "__main__":
    main()
