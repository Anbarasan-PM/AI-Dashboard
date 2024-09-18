"""
      referred from 
      1. https://pygithub.readthedocs.io/en/latest/examples.html
      2. https://stackoverflow.com/questions/17423598/how-can-i-get-a-list-of-all-pull-requests-for-a-repo-through-the-github-api
      3. https://docs.python.org/3/library/exceptions.html
      4. https://regexr.com/
"""


from github import Github, Auth
from collections import defaultdict
import re
import datetime
import numpy as np

# collect data from repo
def get_github_data(repo_url):
    # access token
    auth = Auth.Token("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")
    g = Github(auth=auth)

    # matches user name and repo name 
    git_hub = {"github": "https://github.com/([^/]+)/([^/]+)"}
    match = re.match(git_hub["github"], repo_url)

    try:
        user = match.group(1)
        repo_name = match.group(2)
        repo = g.get_repo(f"{user}/{repo_name}")

        # star
        no_of_stars=repo.stargazers_count
        # fork
        no_of_forks=repo.forks_count
        # contributions
        contributors=repo.get_contributors().totalCount

        # commit msg details for 200 days
        delta = 200
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=delta)
        commits = repo.get_commits(since=start_date, until=end_date)
        cmt_msg=[]
        for commit in commits:
            commit_date = commit.commit.committer.date.date()
            commit_msg=commit.commit.message
            cmt_msg.append([commit_date.strftime("%Y-%m-%d"),commit_msg])

        # pull requests
        pull_req=[]
        for pr in repo.get_pulls(state="all"):
            pull_req.append([pr.title,pr.state,pr.is_merged()])
            

        # isssue
        issues=[]
        issue = repo.get_issues(state='all')

        for iu in issue:
            created_at=str(iu.created_at)
            closed_at=str(iu.closed_at)
            issues.append([iu.state,iu.title,created_at,closed_at,str(iu.pull_request)])


        # reviews
        reviews = []
        for pr in repo.get_pulls(state="all"):
            for review in pr.get_reviews():
                reviews.append([review.body, review.state])
    

        # Return all data
        return {
            "user_name":user,
            "repo_name":repo_name,
            "no_of_stars":no_of_stars,
            "no_of_forks":no_of_forks,
            "contributors":contributors,
            "commit_msg":cmt_msg,
            "pull_requests":pull_req,
            "issues":issues,
            "reviews":reviews
        }

    except Exception as e:
        raise Exception(f"Error (github_api): {e}")



def get_overall_data(repo_url):
    # access token
    auth = Auth.Token("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")
    g = Github(auth=auth)

    # matches user name and repo name
    git_hub = {"github": "https://github.com/([^/]+)/([^/]+)"}
    match = re.match(git_hub["github"], repo_url)

    try:
        user = match.group(1)
        repo_name = match.group(2)
        repo = g.get_repo(f"{user}/{repo_name}")
        
        
        # avg_commit for 30 days
        delta = 30
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=delta)
        commits = repo.get_commits(since=start_date, until=end_date)
        commit_counts = defaultdict(int)
        for commit in commits:
            commit_date = commit.commit.committer.date.date()
            commit_counts[commit_date] += 1
        
        avg_commit=np.average(list(commit_counts.values()))

        # pull requests state
        pulls_state={"merged":0,"open":0,"closed":0}
        for pr in repo.get_pulls(state="all"):
            if pr.state == "open":
                pulls_state["open"] += 1
            elif pr.state == "closed":
                if pr.is_merged():
                    pulls_state["merged"] += 1
                pulls_state["closed"] += 1

        # issues type
        issues_types={'bug':0,'enhancement':0,'documentation':0}
        issues_types['bug'] = repo.get_issues(labels=['bug']).totalCount
        issues_types['enhancement'] = repo.get_issues(labels=['enhancement']).totalCount
        issues_types['documentation'] = repo.get_issues(labels=['documentation']).totalCount


        # reviewstate
        reviews = defaultdict(int)
        for pr in repo.get_pulls(state="all"):
            for review in pr.get_reviews():
                reviews[pr.state]+=1

        # averagereviewtime
        review_times = []
        for pr in repo.get_pulls(state='closed'):
             for review in pr.get_reviews():
                  review_time = (review.submitted_at - pr.created_at).total_seconds()
                  review_times.append(review_time)
        avg_reviewt=np.average(review_times)
        return {
            "avg_commit_counts":avg_commit,
            "pulls_state": pulls_state,
            'issue_types':issues_types,
            'avg_review_time':avg_reviewt,
            "reviews": reviews
        }

    except Exception as e:
        raise Exception(f"Error: {e}")


