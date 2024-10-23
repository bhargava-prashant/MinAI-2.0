import streamlit as st
from PyPDF3 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader
from pymongo import MongoClient
import gridfs
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io

# Initialize Streamlit app
st.set_page_config(page_title="MinAI", page_icon="Untitled design (4).png")

# Load environment variables
load_dotenv()

# Google Generative AI API setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["pdfs"]  # Correct database name 'pdfs'
pdf_collection = db["multiple_pdf"]  # Collection for storing PDFs
fs = gridfs.GridFS(db)  # Use GridFS for storing large PDFs

# Initialize chat history if not present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Initialize vector store in session state
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None  # Store the vector store in session state

# Function to extract text from PDFs
def get_pdf_text(pdf_data):
    text = ""
    pdf_reader = PdfReader(pdf_data)
    for page in pdf_reader.pages:
        text += page.extract_text()
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
    return vector_store

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

# Function to upload PDF to MongoDB using GridFS
def upload_pdf_to_db(pdf_file):
    pdf_binary = pdf_file.read()
    pdf_id = fs.put(pdf_binary, filename=pdf_file.name)  # Store the PDF in GridFS
    pdf_document = {
        "filename": pdf_file.name,
        "pdf_id": pdf_id,  # Store GridFS ID for later retrieval
    }
    pdf_collection.insert_one(pdf_document)
    st.success(f"Uploaded '{pdf_file.name}' successfully to the database!")

# Function to process and create a vector store from uploaded PDFs
def process_uploaded_pdfs(pdf_docs):
    all_chunks = []
    for pdf in pdf_docs:
        pdf_data = fs.get(pdf['pdf_id']).read()  # Retrieve the PDF data from GridFS
        pdf_text = get_pdf_text(io.BytesIO(pdf_data))
        chunks = get_text_chunks(pdf_text)
        all_chunks.extend(chunks)
    
    # Create a vector store from all chunks
    st.session_state['vector_store'] = get_vector_store(all_chunks)

# Function to retrieve PDFs from the database
def retrieve_pdfs_from_db():
    return list(pdf_collection.find())  # Find all PDF documents in the collection

# Function to process user input
def user_input():
    user_question = st.session_state['user_question_input']
    if user_question and st.session_state['vector_store']:
        docs = st.session_state['vector_store'].similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.session_state['chat_history'].append({"question": user_question, "response": response["output_text"]})
        st.session_state['user_question_input'] = ""  # Clear the input field

# Main function to run the Streamlit app
def main():
    st.header("Chat with Multiple PDFs using MinAI 2.0")

    with st.sidebar:
        st.title("Menu:")
        if st.button("New Chat"):
            st.session_state['chat_history'] = []  # Clear chat history
            st.session_state['vector_store'] = None  # Clear vector store
            st.success("New chat started! Chat history has been reset.")
        
        existing_pdfs = retrieve_pdfs_from_db()
        if existing_pdfs and not st.session_state['vector_store']:
            with st.spinner("Processing PDFs from database..."):
                process_uploaded_pdfs(existing_pdfs)
                st.success("Existing PDFs from the database have been processed!")

        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type="pdf")
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    for pdf in pdf_docs:
                        upload_pdf_to_db(pdf)
                    process_uploaded_pdfs(retrieve_pdfs_from_db())
                    st.success("All PDFs processed and stored!")
            else:
                st.error("Please upload at least one PDF file.")

    # Display chat history
    st.subheader("Conversation")
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**MinAI 2.0:** {chat['response']}")
            st.markdown("---")

    # Input field for user question
    st.text_input("Ask a Question", key="user_question_input", on_change=user_input)

if __name__ == "__main__":
    main()
