from collections import defaultdict
import datetime
import numpy as np



# calculates frequency of commit message on each day
def calculate_commit_f(commits):
    commit_counts=defaultdict(int)
    for commit in commits:
        commit_counts[commit]+=1
    
    commits_per_day=sum(commit_counts.values())

    return f"{commits_per_day}"

# calculate pull requests ratio 

def calculate_pull_requests_ratio(pull_requests):
    pulls_state = {"merged": 0, "open": 0, "closed": 0}

    for pull_request in pull_requests:
        if len(pull_request) >= 3:
            if pull_request[1] == "open":
                pulls_state["open"] += 1
            elif pull_request[1] == "closed":
                if pull_request[2]==True:  
                    pulls_state["merged"] += 1
                pulls_state["closed"] += 1
        else:
            print(f"Invalid pull request data: {pull_request}")

    total_pulls = pulls_state["open"] + pulls_state["closed"]
    if total_pulls > 0:
        pulls_req_ratio = pulls_state["merged"] / total_pulls
        return f"{pulls_req_ratio:.2f}"
    else:
        return "No pull requests available to calculate ratio"



# calculate resolution time in seconds

def calculate_resolution_time(issues):
    avg_res = 0
    tot_res = 0
    issue_cnt = 0

    for issue in issues:
        try:
            created_at = issue[2]  
            closed_at = issue[3]   
            
            if issue[4] == 'None':
                closed_at = created_at  
            
            
            created_dt = datetime.datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S%z")
            closed_dt = datetime.datetime.strptime(closed_at, "%Y-%m-%d %H:%M:%S%z")
            
            res_time = (closed_dt - created_dt).total_seconds()
            
            
            tot_res += res_time
            issue_cnt += 1
        except ValueError as e:
            print(f"Date parsing error: {e}")

    if issue_cnt > 0:
        avg_res = tot_res / issue_cnt / (24 * 60 * 60) 

    return np.round(avg_res,1)

def cal_stars(data):
    return data["no_of_stars"]
def cal_fork(data):
    return data["no_of_forks"]
def cal_contributors(data):
    return data["contributors"]
    