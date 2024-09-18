"""
referred links
     1. https://docs.streamlit.io/develop/api-reference/text
     2. used chat-gpt to customize frontend
     3. https://github.com/dataprofessor/streamlit_freecodecamp
"""



import pandas as pd
import streamlit as st

from data_collection.github_api import get_github_data, get_overall_data
from data_collection.data_storage import store_data_in_csv, store_overall_data

from metrics.definitions import commit_frequency, issue_resolution, pull_requests_ratio, cal_stars, cal_fork, cal_contributors
from visualization.dashboard import github_metrics_dashboard


import plotly.graph_objects as go
import json
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


st.set_page_config(page_title="GitHub Metrics Dashboard", layout="centered")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overall Metrics", "Individual Developer Metrics", "Natural Language Query"])

def set_page_bg_image(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{url}");
            background-size: cover;
            background-blend-mode: lighten;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.6);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
def clear_page_bg():
    st.markdown(
        """
        <style>
        .stApp {{
            background: none;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def create_gauge_chart(value, title, max_value=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge=dict(
            axis=dict(range=[None, max_value]),
            bar=dict(color="black"),
            bgcolor="white",
            steps=[dict(range=[0, max_value * 0.5], color="lightgray"),
                   dict(range=[max_value * 0.5, max_value], color="gray")],
        ),
        title={'text': title}
    ))

    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

if page == "Overall Metrics":
    #set_page_bg_image("https://b2316719.smushcdn.com/2316719/wp-content/uploads/2022/03/pattern_02-768x591.jpg?lossy=1&strip=1&webp=1")
    
    st.title("Overall Repository Metrics")
    st.sidebar.header("Input Options")
    uploaded_file = st.sidebar.file_uploader("Upload your GitHub Metrics CSV", type=["csv"])
    
    st.sidebar.header("Filter Options")
    review_filter = st.sidebar.multiselect("Select Review Status", options=['open', 'closed'], default=['open', 'closed'])
    issue_filter = st.sidebar.multiselect("Select Issue Types", options=['bug', 'enhancement', 'documentation'], default=['bug', 'enhancement', 'documentation'])
    pull_filter = st.sidebar.multiselect("Select Pulls State", options=['merged', 'open', 'closed'], default=['merged', 'open', 'closed'])
    github_metrics_dashboard(uploaded_file, review_filter, issue_filter, pull_filter)










    
elif page == "Individual Developer Metrics":
    clear_page_bg()
    set_page_bg_image("https://www.bleepstatic.com/content/hl-images/2021/05/10/GitHub-headpic.jpg")
    
    st.title("Individual Developer Metrics")
    
    repo_url = st.text_input("Enter the GitHub repository URL for developer metrics", placeholder="https://github.com/user/repo")
    
    if repo_url:
        try:
            data = get_github_data(repo_url)
            overall_data = get_overall_data(repo_url)

            if data:
                user_name = data['user_name']
                repo_name = data['repo_name']

                file_name = "github_data.csv"
                store_data_in_csv(data, file_name)

                overall_fnme = "overall_metric_data.csv"
                store_overall_data(overall_data, overall_fnme)
                st.success(f"Data saved to {file_name}")

                st.markdown(
                    f"""
                    <div style="background-color:#8e44ad; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                        <h1 style="color:white;">{user_name}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                no_of_stars = cal_stars(data)
                no_of_forks = cal_fork(data)
                contributors = cal_contributors(data)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.plotly_chart(create_gauge_chart(no_of_stars, "Stars", max_value=100))
                
                with col2:
                    st.plotly_chart(create_gauge_chart(no_of_forks, "Forks", max_value=100))
                
                with col3:
                    st.plotly_chart(create_gauge_chart(contributors, "Contributors", max_value=50))

                avg_commits_per_day = commit_frequency(file_name)
                avg_resolution_time = issue_resolution(file_name)
                pull_request_ratio = pull_requests_ratio(file_name)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        f"""
                        <div style="background-color:#2ecc71; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3 style="color:white;">Commit Frequency</h3>
                            <h1 style="color:white;">{avg_commits_per_day}</h1>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        f"""
                        <div style="background-color:#3498db; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3 style="color:white;">Issue Resolution Time</h3>
                            <h1 style="color:white;">{avg_resolution_time}</h1>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        f"""
                        <div style="background-color:#e74c3c; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3 style="color:white;">Pull Request Ratio</h3>
                            <h1 style="color:white;">{pull_request_ratio}</h1>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"Error (Individual Developer Metrics): {e}")



elif page == "Natural Language Query":
    import streamlit as st
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.llms import Ollama
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.callbacks.manager import CallbackManager
    from langchain.chains import RetrievalQA
    from dotenv import load_dotenv
    import os
    import pandas as pd
    import json

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
    if 'df_exploded' not in st.session_state:
        st.session_state.df_exploded = None

    # Initialize the result history
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Streamlit app
    st.title("Data Processing and Query Interface")

    # Upload CSV file
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    def process_csv(file):
        global vectordb
        if file:
            # Load CSV
            data = pd.read_csv(file)
            df = pd.DataFrame(data)

            # Display data types
            st.write("Data types in DataFrame:")
            st.write(df.dtypes)

            # Convert all columns to string to avoid dtype issues with Arrow
            df = df.applymap(str)

            # Convert specified columns to dictionaries (handles JSON decoding)
            def convert_to_dict(column):
                try:
                    return column.apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                except json.JSONDecodeError:
                    st.error("Error decoding JSON for column data")
                    return column

            df['commits'] = convert_to_dict(df['commits'])
            df['pull_requests'] = convert_to_dict(df['pull_requests'])
            df['issues'] = convert_to_dict(df['issues'])
            df['reviews'] = convert_to_dict(df['reviews'])

            # Flatten nested structures by converting complex types to JSON strings
            for col in ['commits', 'pull_requests', 'issues', 'reviews']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))

            # Explode the DataFrame for all relevant columns
            st.session_state.df_exploded = df.explode('commits').explode('pull_requests').explode('issues').explode('reviews')

            # Select only the first 1000 rows
            df_exploded1 = st.session_state.df_exploded.head(1000)

            # Convert complex columns to strings to avoid Arrow serialization issues
            for col in ['commits', 'pull_requests', 'issues', 'reviews']:
                if col in df_exploded1.columns:
                    df_exploded1[col] = df_exploded1[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))

            # Create a list of strings from DataFrame for the vector database
            data = df_exploded1.to_dict(orient='records')
            text_data = [str(record) for record in data]

            # Vector Database Creation
            vectordb = Chroma.from_texts(text_data, embeddings, persist_directory=persist_directory)
            vectordb.persist()

            st.success("Vector DB Created and Persisted")

            # Store vectordb in session state
            st.session_state.vectordb = vectordb

            # Display the DataFrame without serialization issues
            st.write(df_exploded1)

            # Display a sample of the DataFrame to verify
            st.write("Sample of exploded DataFrame:")
            st.write(df_exploded1.head())

    if csv_file:
        process_csv(csv_file)

    # Initialize query chain only if vectordb is available
    query_chain = None
    if 'vectordb' in st.session_state and st.session_state.vectordb:
        query_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=st.session_state.vectordb.as_retriever()
        )

    def handle_specific_query(query):
        if "total number of commits" in query.lower():
            df_unique_commits = st.session_state.df_exploded.drop_duplicates(subset=['commits'])
            total_commits = df_unique_commits['commits'].count()
            return f"Total number of commits: {total_commits}"
        
        elif "filter commits by user" in query.lower():
            username = query.split("user")[1].strip()
            df_unique_commits = st.session_state.df_exploded.drop_duplicates(subset=['commits'])
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
            repo_data = st.session_state.df_exploded[st.session_state.df_exploded['repo_name'] == repo_name]
            repo_data = st.session_state.df_exploded.drop_duplicates(subset=['pull_requests'])
            
            open_pull_requests = repo_data[repo_data['pull_requests'].apply(lambda x: isinstance(x, list) and x[1] == "open")]
            closed_pull_requests = repo_data[repo_data['pull_requests'].apply(lambda x: isinstance(x, list) and x[1] == "closed")]
            
            num_open = open_pull_requests.shape[0]
            num_closed = closed_pull_requests.shape[0]
            
            return f"Number of open pull requests for '{repo_name}': {num_open}\nNumber of closed pull requests for '{repo_name}': {num_closed}"
        
        elif "number of open and closed issues for" in query.lower():
            repo_name = query.split("for")[1].strip()
            repo_data = st.session_state.df_exploded[st.session_state.df_exploded['repo_name'] == repo_name]
            repo_data = st.session_state.df_exploded.drop_duplicates(subset=['issues'])
            
            open_issues = repo_data[repo_data['issues'].apply(lambda x: isinstance(x, list) and x[0] == "open")]
            closed_issues = repo_data[repo_data['issues'].apply(lambda x: isinstance(x, list) and x[0] == "closed")]
            
            num_open = open_issues.shape[0]
            num_closed = closed_issues.shape[0]
            
            return f"Number of open issues for '{repo_name}': {num_open}\nNumber of closed issues for '{repo_name}': {num_closed}"
        
        elif "number of reviews for" in query.lower():
            repo_name = query.split("for")[1].strip()
            repo_data = st.session_state.df_exploded[st.session_state.df_exploded['repo_name'] == repo_name]
            repo_data = st.session_state.df_exploded.drop_duplicates(subset=['reviews'])
            
            num_reviews = repo_data['reviews'].dropna().shape[0]
            
            return f"Number of reviews for '{repo_name}': {num_reviews}"

    if query_chain:
        query = st.text_input("Enter your query:")
        if query:
            st.write("💡Thinking...")
            specific_answer = handle_specific_query(query)
            if specific_answer:
                response = specific_answer
            else:
                response = query_chain({"query": query})['result']
            
            # Append the query and response to the history
            st.session_state.query_history.append((query, response))
            
            st.write("Assistant:", response)

    # Display the query history
    if st.session_state.query_history:
        st.write("### Query History")
        for q, r in st.session_state.query_history:
            st.write(f"**Query:** {q}")
            st.write(f"**Response:** {r}")

    else:
        st.write("Vector DB is not initialized.")
