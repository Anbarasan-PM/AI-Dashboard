# AI-Powered Developer Performance Analytics Dashboard

![image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkJAH_Tug5SJu--Vvhn1xsAqSnfqUGhL6eSg&s)

### GitHub Fine-Grained Personal Access Token Guide

This guide explains how to create a fine-grained personal access token for accessing GitHub resources securely. Follow these steps to generate your token:

#### Steps to Generate a Fine-Grained Personal Access Token:

1. **Verify Your Email Address**  
   If you haven't verified your email address yet, make sure to do so before generating a token.

2. **Go to GitHub Settings**  
   - In the upper-right corner of any GitHub page, click your profile photo.
   - From the dropdown, select **Settings**.

3. **Access Developer Settings**  
   - In the left sidebar, click **Developer settings**.

4. **Navigate to Fine-Grained Tokens**  
   - Under **Personal access tokens**, click **Fine-grained tokens**.

5. **Generate a New Token**  
   - Click **Generate new token**.

6. **Configure Your Token**  
   - **Token Name**: Enter a descriptive name for your token.
   - **Expiration**: Select an expiration date for your token (optional but recommended).
   - **Description**: Optionally, add a note describing the purpose of the token.

7. **Select Resource Owner**  
   - Choose the **Resource Owner** for the token. The token will only have access to resources owned by the selected resource owner.
   - Note: Organizations may not appear unless they have opted in to fine-grained tokens.

8. **Provide Justification (Optional)**  
   - If the resource owner is an organization that requires approval for fine-grained tokens, enter a **justification** for the request.

9. **Define Repository Access**  
   - Under **Repository access**, select which repositories you want the token to access. Choose the minimal repository access that meets your needs. 
   - Public repositories will always have read-only access.

10. **Select Specific Repositories (If Applicable)**  
   - If you chose **Only select repositories**, use the **Selected repositories** dropdown to select the repositories you want the token to access.

11. **Assign Permissions**  
   - Under **Permissions**, specify the permissions to grant the token. Permissions include:
     - **Repository**: Access to repositories.
     - **Organization**: Access to organization settings.
     - **Account**: Access to your GitHub account.
   - You should always select the minimal permissions necessary.

12. **Save and Use Your Token**  
   After configuring your token, click **Generate token**. Copy the token and store it securely, as it will not be shown again.

#### Usage Notes

- **Minimal Access**: Always select the minimal access and permissions necessary to avoid overexposing sensitive data.
- **Expiration**: Tokens with expiration dates improve security, as they reduce long-term risks.
- **Security**: Store your token securely and never share it publicly. GitHub will only display your token once.

For more information on GitHub fine-grained personal access tokens, visit the official [GitHub documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

## Installation

- pip install -r requirements.txt

- `Project Structure` is mentioned at end of the file

This simple command allows users to install all the required Python packages listed in the `requirements.txt` file for the project.

## Dev Performance Dashboard

### data collection
 -  It is used to store and collect the repo infromation which is given by the user is csv file in an automated way using `GitHub Access Tokens`.

 - * `github_api.py` uses to fetch information, preprocess the fetched information and store the values in the new dataframe.
 - * `data_storage.py` uses to store the dataframe created by `github_api.py` after preprocessing.

### metrics
  - * This project calculates various GitHub metrics such as commit frequency, pull request ratio, issue resolution time, repository stars, forks and contributors to provide insights into repository performance.

  - * `calculator.py` is used to create definitions of the metrics where `definitions.py` is used to calculate the metrics by call calculator.py file.


### visualization and query interface
  
  - * `charts.py` visualization functions are defined and `dashboard.py` calls the functions.
  - * `nlp_processsor.py` creates chatbot simply by calling the bot() in `app.py`.

### .env
Refer to [google api key](https://cloud.google.com/docs/authentication/api-keys) for instructions on how to generate a Google API key for using LLM's.


### app.py - Streamlit Pages

The application offers three main sections for analyzing GitHub repository data:

1. **Overall Metrics**  
   Provides an overview of repository-wide statistics such as commit frequency, pull request ratios, issue resolution time, stars, forks, and contributors.

2. **Individual Developer Statistics**  
   Focuses on metrics for individual contributors, highlighting their commit patterns, pull request contributions, and issue resolution activities.

3. **Natural Language Query Interface**  
   Allows users to interact with the data using natural language queries, making it easy to retrieve specific information about the repository and developers without needing technical expertise.

`streamlit run app.py` run this command

## Images

**Individual Contributor Statistics**

<img src="https://github.com/Anbarasan-PM/AI-Dashboard/blob/main/dev_performance_dashboard_1/Individual_Contributor_Statistics.png" alt="Individual Contributor Statistics" width="400"/>


**Overall Metrics**

<img src="https://github.com/Anbarasan-PM/AI-Dashboard/blob/main/dev_performance_dashboard_1/Overall_Metrics.png" alt="Overall Metrics" width="400"/>

**Natural Language Query Interface - Chat Bot**

<img src="Chat_Bot.png" alt="Chat Bot" width="400"/>




## Project Structure

```plaintext
dev_performance_dashboard/
├── data_collection/
│   ├── github_api.py
│   └── data_storage.py
├── metrics/
│   ├── calculator.py
│   └── definitions.py
├── visualization/
│   ├── charts.py
│   └── dashboard.py
├── query_interface/
│   ├── nlp_processor.py
│   └── response_generator.py
├── .env
├── app.py
├── requirements.txt
└── README.md






