import pandas as pd
import json
import datetime

from metrics.calculator import calculate_commit_f
from metrics.calculator import calculate_resolution_time
from metrics.calculator import calculate_pull_requests_ratio
from metrics.calculator import cal_stars
from metrics.calculator import cal_fork
from metrics.calculator import cal_contributors

def read_data_from_csv(file_name):
    df = pd.read_csv(file_name)
    
    # Deserialize JSON strings to lists
    df['commits'] = df['commits'].apply(json.loads)
    df["pull_requests"]=df["pull_requests"].apply(json.loads)
    df['issues'] = df['issues'].apply(json.loads)
    df['reviews'] = df['reviews'].apply(json.loads)
    
    return df



def commit_frequency(file_name):
    df=read_data_from_csv(file_name).copy()
    lst=[]
    for commit in df["commits"].iloc[-1]:
        lst.append(commit[0])
    return calculate_commit_f(lst)

def pull_requests_ratio(file_name):
    df=read_data_from_csv(file_name).copy()
    lst=[]
    for pull_requet in df["pull_requests"].iloc[-1]:
        lst.append(pull_requet)
    return calculate_pull_requests_ratio(lst)



def issue_resolution(file_name):
    df=read_data_from_csv(file_name).copy()
    lst=[]

    for issues in df["issues"].iloc[-1]:
        if issues[0]=="closed":
            lst.append(issues)
    return calculate_resolution_time(lst)

def stars(file_name):
    df=read_data_from_csv(file_name).copy()

    return cal_stars(df)
def fork(file_name):
    df=read_data_from_csv(file_name).copy()

    return cal_fork(df)
def contributors(file_name):
    df=read_data_from_csv(file_name).copy()

    return cal_contributors(df)


    