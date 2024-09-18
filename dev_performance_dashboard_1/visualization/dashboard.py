"""
referred links
  1. https://www.w3schools.com/colors/colors_groups.asp
  2. https://docs.streamlit.io/develop/concepts/design/animate
  3. used chat-gpt to customize charts and dashboard
  4. https://plotly.com/python/
      
"""




import streamlit as st
import pandas as pd
import numpy as np
import json
from visualization.charts import generate_histogram,generate_distplot,generate_bar_chart,generate_scatter_plot,generate_pie_chart

def github_metrics_dashboard(uploaded_file, review_filter, issue_filter, pull_filter):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['pulls_state'] = df['pulls_state'].apply(json.loads)
        df['issue_types'] = df['issue_types'].apply(json.loads)
        df['reviews'] = df['reviews'].apply(json.loads)
    else:
        st.error("Please upload a CSV file to proceed.")
        return

    # Filtering Function
    def filter_df(df, review_filter, issue_filter, pull_filter):
        pulls_df = pd.json_normalize(df['pulls_state'])
        issues_df = pd.json_normalize(df['issue_types'])
        reviews_df = pd.json_normalize(df['reviews'])

        # Apply filters
        pulls_df = pulls_df.loc[:, pulls_df.columns.intersection(pull_filter)]
        issues_df = issues_df.loc[:, issues_df.columns.intersection(issue_filter)]
        reviews_df = reviews_df.loc[:, reviews_df.columns.intersection(review_filter)]

        return pulls_df, issues_df, reviews_df

    if not df.empty:
        pulls_df, issues_df, reviews_df = filter_df(df, review_filter, issue_filter, pull_filter)

        # Structured layout with consistent sizes
        col1, col2 = st.columns(2)
        
        with col1:
            if "average_review_time" in df.columns:
                slider_col1, slider_col2 = st.columns([2,2])
                with slider_col1:
                    mean = st.slider("Mean", min_value=float(df["average_review_time"].min()), max_value=float(df["average_review_time"].max()), value=float(df["average_review_time"].mean()), label_visibility='visible')
                with slider_col2:
                    std = st.slider("Std Dev", min_value=0.1, max_value=5.0, value=float(df["average_review_time"].std()), label_visibility='visible')
                
                data = np.random.normal(mean, std, size=100)
                fig2 = generate_histogram(data, [df["average_review_time"].min(), df["average_review_time"].max()])
                st.plotly_chart(fig2, use_container_width=True)

            else:
                st.warning("Column 'average_review_time' not found in data.")

        with col2:
            if not reviews_df.empty:
                reviews_sum = reviews_df.sum()
                fig5 = generate_bar_chart(reviews_sum)
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.warning("No data available for 'reviews'.")

        col3, col4 = st.columns(2)

        with col3:
            if not pulls_df.empty:
                pulls_counts = pulls_df.sum()
                fig3 = generate_pie_chart(pulls_counts.index, pulls_counts.values, "Pulls State Distribution")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("No data available for 'pulls_state'.")

        with col4:
            if not issues_df.empty:
                issues_counts = issues_df.sum()
                fig4 = generate_pie_chart(issues_counts.index, issues_counts.values, "Issue Types Distribution")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("No data available for 'issue_types'.")

        col5, col6 = st.columns(2)

        with col5:
            if "average_commit_count" in df.columns:
                hist_data = [df['average_commit_count'].dropna()]
                group_labels = ['Average Commit Count']
                fig1 = generate_distplot(hist_data, group_labels)
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("Column 'average_commit_count' not found in data.")
        
        with col6:
            if "average_commit_count" in df.columns and "average_review_time" in df.columns:
                df['frequency'] = df['average_commit_count'].map(df['average_commit_count'].value_counts())
                fig = generate_scatter_plot(df, 'average_commit_count', 'average_review_time', 'frequency')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Columns 'average_commit_count' or 'average_review_time' not found in data.")
