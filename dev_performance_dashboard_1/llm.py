import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.globals import set_verbose  # New import for verbosity
from dotenv import load_dotenv
import os
import pandas as pd
import json

# Set verbosity (if needed)
set_verbose(True)  # or False, depending on your needs

# Load environment variables
load_dotenv()

# Model selection
model_type = "ollama"

# Initializing LLaMA
if model_type == "ollama":
    model = Ollama(
        model="llama3.1",  # Updated to use the llama3.1 model
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

# Vector Database
persist_directory = 'db'  # Persist directory path
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define the text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)

# Initialize vectordb
vectordb = None

# Global variable for the exploded DataFrame
df_exploded = None

# Process CSV file
csv_file_path = r"C:\\Users\\Anbarasan\\Downloads\\files\\ollama\\data.csv"  # Path to the CSV file

def process_csv(csv_file_path):
    global vectordb, df_exploded
    if os.path.exists(csv_file_path):
        # Load CSV
        data = pd.read_csv(csv_file_path)
        df = pd.DataFrame(data)

        # Convert specified columns to dictionaries
        def convert_to_dict(column):
            try:
                return column.apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            except json.JSONDecodeError:
                return column

        df['commits'] = convert_to_dict(df['commits'])
        df['pull_requests'] = convert_to_dict(df['pull_requests'])
        df['issues'] = convert_to_dict(df['issues'])
        df['reviews'] = convert_to_dict(df['reviews'])

        # Explode the DataFrame
        df_exploded = df.explode('commits').explode('pull_requests').explode('issues').explode('reviews')
        df_exploded1 = df_exploded.head(1000)

        # Create a list of strings from DataFrame for the vector database
        data = df_exploded1.to_dict(orient='records')
        text_data = [str(record) for record in data]

        # Vector Database Creation
        vectordb = Chroma.from_texts(text_data, embeddings, persist_directory=persist_directory)
        vectordb.persist()

process_csv(csv_file_path)

# Initialize query chain only if vectordb is available
query_chain = None
if vectordb:
    query_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vectordb.as_retriever()
    )

def handle_specific_query(query):
    global df_exploded
    if "total number of commits" in query.lower():
        df_unique_commits = df_exploded.drop_duplicates(subset=['commits'])
        total_commits = df_unique_commits['commits'].count()
        return f"Total number of commits: {total_commits}"
    
    elif "filter commits by user" in query.lower():
        username = query.split("user")[1].strip()
        df_unique_commits = df_exploded.drop_duplicates(subset=['commits'])
        filtered_commits = df_unique_commits[df_unique_commits['user_name'] == username]
        if filtered_commits.empty:
            return f"No commits found for user {username}"
        
        result = f"Commits by {username}:\n"
        for commits in filtered_commits['commits']:
            if isinstance(commits, list):
                for commit in commits:
                    result += f"{commit}\n"
            else:
                result += f"{commits}\n"
        return result
        
    elif "number of open and closed pull requests for" in query.lower():
        repo_name = query.split("for")[1].strip()
        repo_data = df_exploded[df_exploded['repo_name'] == repo_name]
        repo_data = df_exploded.drop_duplicates(subset=['pull_requests'])
        
        open_pull_requests = repo_data[repo_data['pull_requests'].apply(lambda x: isinstance(x, list) and x[1] == "open")]
        closed_pull_requests = repo_data[repo_data['pull_requests'].apply(lambda x: isinstance(x, list) and x[1] == "closed")]
        
        num_open = open_pull_requests.shape[0]
        num_closed = closed_pull_requests.shape[0]
        
        return f"Number of open pull requests for '{repo_name}': {num_open}\nNumber of closed pull requests for '{repo_name}': {num_closed}"
    
    elif "number of open and closed issues for" in query.lower():
        repo_name = query.split("for")[1].strip()
        repo_data = df_exploded[df_exploded['repo_name'] == repo_name]
        repo_data = df_exploded.drop_duplicates(subset=['issues'])
        
        open_issues = repo_data[repo_data['issues'].apply(lambda x: isinstance(x, list) and x[0] == "open")]
        closed_issues = repo_data[repo_data['issues'].apply(lambda x: isinstance(x, list) and x[0] == "closed")]
        
        num_open = open_issues.shape[0]
        num_closed = closed_issues.shape[0]
        
        return f"Number of open issues for '{repo_name}': {num_open}\nNumber of closed issues for '{repo_name}': {num_closed}"
    
    elif "number of reviews for" in query.lower():
        repo_name = query.split("for")[1].strip()
        repo_data = df_exploded[df_exploded['repo_name'] == repo_name]
        repo_data = df_exploded.drop_duplicates(subset=['reviews'])
        
        num_reviews = repo_data['reviews'].dropna().shape[0]
        
        return f"Number of reviews for '{repo_name}': {num_reviews}"

# Streamlit App
st.title("Data Query Interface")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query:
        specific_answer = handle_specific_query(query)
        if specific_answer:
            response = specific_answer
        else:
            if query_chain:
                with st.spinner('💡 Thinking...'):
                    response = query_chain({"query": query})['result']
            else:
                response = "Vector DB is not initialized."
        st.write("Assistant:", response)
    else:
        st.write("Please enter a query.")
