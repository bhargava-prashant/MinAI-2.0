import streamlit as st
from PyPDF3 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

 
st.set_page_config(page_title="MinAI", page_icon="Untitled design (4).png")

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize chat history if not present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfFileReader(pdf)
        for page in pdf_reader.pages:
            text += page.extractText()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get the question-answer chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input
def user_input():
    user_question = st.session_state['user_question_input']
    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load FAISS index with dangerous deserialization enabled
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Search for relevant documents
        docs = new_db.similarity_search(user_question)

        # Get the conversational chain
        chain = get_conversational_chain()

        # Get the response from the chain
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        # Append question and response to chat history
        st.session_state['chat_history'].append({"question": user_question, "response": response["output_text"]})

        # Clear input field after submission
        st.session_state['user_question_input'] = ""

# Main function to run the Streamlit app
def main():
    st.header("Chat with Multiple PDF using MinAI")

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")

        # "New Chat" Button to Reset Chat History
        if st.button("New Chat"):
            st.session_state['chat_history'] = []  # Clear chat history
            st.success("New chat started! Chat history has been reset.")

        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")
            else:
                st.error("Please upload at least one PDF file.")

    # Display chat history
    st.subheader("Conversation")
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**MinAI:** {chat['response']}")
            st.markdown("---")

    # Input field for user question (on "Enter" submission)
    st.text_input("Ask a Question", key="user_question_input", on_change=user_input)

    # Add creator's name at the bottom-right corner
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            text-color: white;
            bottom: 0;
            right: 0;
            background-color: default;
            color: grey;
            padding: 10px;
            font-size: 14px;
        }
        </style>
        <div class="footer">
            Created by Prashant Bhargava
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
