import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pymongo import MongoClient
import gridfs
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io
import hashlib  # For password hashing



# Initialize Streamlit app
st.set_page_config(page_title="MinAI", page_icon="Untitled design (4).png")

# Load environment variables
load_dotenv()

# Google Generative AI API setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["pdfs"]  # Main database
user_collection = db["users"]  # Collection for user data

# Initialize session state variables if not present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Helper function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Sign up function
def signup(username, email, password):
    if user_collection.find_one({"username": username}):
        st.error("Username already exists. Please choose a different username.")
        return False
    hashed_password = hash_password(password)
    user_data = {"username": username, "email": email, "password": hashed_password}
    user_collection.insert_one(user_data)
    st.success("Successfully registered! You can now log in.")
    return True

# Login function
def login(identifier, password):
    user = user_collection.find_one({"$or": [{"username": identifier}, {"email": identifier}]} )
    if user and user["password"] == hash_password(password):
        st.session_state['user'] = user  # Store user data in session state
        return True
    st.error("Invalid username/email or password.")
    return False

# Logout function
def logout():
    if st.session_state['user']:  # Check if the user is logged in
        st.session_state['user'] = None
        st.session_state['chat_history'] = []
        st.session_state['vector_store'] = None
        st.success("Logged out successfully.")
    else:
        st.warning("No user is currently logged in.")

# Function to show login/signup interface
def auth_interface():
    auth_choice = st.sidebar.radio("Select Login or Sign-Up", ["Login", "Sign-Up"], key="auth_choice")
    
    if auth_choice == "Sign-Up":
        st.subheader("Create a New Account")
        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Register"):
            signup(username, email, password)
    
    elif auth_choice == "Login":
        st.subheader("Login to Your Account")
        identifier = st.text_input("Username or Email", key="login_identifier")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login(identifier, password):
                st.success("Login successful! Welcome to the chat application.")

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

def get_vector_store(text_chunks):
    if not text_chunks:  # Check if text_chunks is empty
        st.error("No text chunks found. Please ensure the PDFs have been processed correctly.")
        return None

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
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to upload PDF to MongoDB under user-specific collection
def upload_pdf_to_user_db(pdf_file):
    username = st.session_state['user']['username']
    user_db = client[username]  # Switch to user-specific database
    fs = gridfs.GridFS(user_db)
    pdf_binary = pdf_file.read()
    pdf_id = fs.put(pdf_binary, filename=pdf_file.name)
    pdf_collection = user_db["pdf_files"]
    pdf_document = {
        "filename": pdf_file.name,
        "pdf_id": pdf_id,
    }
    pdf_collection.insert_one(pdf_document)
    st.success(f"Uploaded '{pdf_file.name}' successfully to {username}'s database!")

# Function to process and create vector store from user's PDFs
def process_user_pdfs():
    # Check if user is logged in
    if 'user' not in st.session_state or st.session_state['user'] is None:
        st.error("User not logged in. Please log in to access your PDFs.")
        return

    username = st.session_state['user']['username']
    user_db = client[username]
    fs = gridfs.GridFS(user_db)
    pdf_collection = user_db["pdf_files"]
    
    all_chunks = []
    for pdf_doc in pdf_collection.find():
        pdf_data = fs.get(pdf_doc['pdf_id']).read()
        pdf_text = get_pdf_text(io.BytesIO(pdf_data))
        chunks = get_text_chunks(pdf_text)
        
        # Check if chunks were created
        if not chunks:
            st.warning(f"No text chunks found in {pdf_doc['filename']}.")
            continue
        
        all_chunks.extend(chunks)
    
    if all_chunks:  # Check if any chunks were collected
        st.session_state['vector_store'] = get_vector_store(all_chunks)
    else:
        st.warning("No text chunks found across all PDFs. Please check your files.")

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

    # Authentication required before accessing the app
    if not st.session_state['user']:
        auth_interface()
    else:
        with st.sidebar:
            st.title("Menu:")
            if st.button("Logout"):
                logout()
            
            if st.button("New Chat"):
                st.session_state['chat_history'] = []
                st.session_state['vector_store'] = None
                st.success("New chat started! Chat history has been reset.")
            
            # Load user PDFs only if vector store is not already loaded
            if not st.session_state['vector_store']:
                with st.spinner("Processing PDFs from user database..."):
                    process_user_pdfs()
                    st.success("Your PDFs have been processed!")

            pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type="pdf")
            if st.button("Submit & Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        for pdf in pdf_docs:
                            upload_pdf_to_user_db(pdf)
                        process_user_pdfs()
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
