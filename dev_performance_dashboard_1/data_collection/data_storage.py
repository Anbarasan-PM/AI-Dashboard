"""
           https://docs.github.com/en/rest/pulls?apiVersion=2022-11-28
           https://stackoverflow.com/questions/17423598/how-can-i-get-a-list-of-all-pull-requests-for-a-repo-through-the-github-api
"""


import pandas as pd
import os
import json

# to store individual records
def store_data_in_csv(data, file_name):
    try:
        user_name = data['user_name']
        repo_name = data['repo_name']
        no_of_stars = data['no_of_stars']
        no_of_forks = data['no_of_forks']
        contributors = data['contributors']
        commits = data['commit_msg']
        pull_req = data["pull_requests"]
        issues = data['issues']
        reviews = data['reviews']

        cmt_msg_json = json.dumps(commits)
        pull_req_json = json.dumps(pull_req)
        issues_json = json.dumps(issues)
        reviews_json = json.dumps(reviews)

        new_data = pd.DataFrame({
            'user_name': [user_name],
            'repo_name': [repo_name],
            'no_of_stars': [no_of_stars],
            'no_of_forks': [no_of_forks],
            'contributors': [contributors],
            'commits': [cmt_msg_json],
            'pull_requests': [pull_req_json],
            'issues': [issues_json],
            'reviews': [reviews_json]
        })


        
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

    try :
        if os.path.exists(file_name):
            new_data.to_csv(file_name, mode='a', header=False, index=False)
        else:
            new_data.to_csv(file_name, mode='w', header=True, index=False)
    except Exception as e:
        print(f"Error (data_storage) {e}")


# to store overall data
def store_overall_data(overall_data,file_name):

  
    avg_commit_counts=overall_data['avg_commit_counts']
    pulls_state= overall_data['pulls_state']
    issue_types=overall_data['issue_types']
    avg_review_time=overall_data['avg_review_time']
    reviews= overall_data['reviews']

    pull_state_json =json.dumps(pulls_state)
    issue_types_json=json.dumps(issue_types)
    reviews_json= json.dumps(reviews)

    new_data = pd.DataFrame({
        'average_commint_count': [avg_commit_counts],
        'average_review_time': [avg_review_time],
        'pulls_state': [pull_state_json],
        'issue_types':[issue_types_json],
        'reviews': [reviews_json]
    })

    try :
        if os.path.exists(file_name):
            new_data.to_csv(file_name, mode='a', header=False, index=False)
        else:
            new_data.to_csv(file_name, mode='w', header=True, index=False)
    except Exception as e:
        print(f"Error (data_storage) {e}")

