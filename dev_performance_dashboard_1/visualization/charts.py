# charts.py
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

def generate_histogram(data, x_range):
    fig = px.histogram(data, range_x=x_range)
    fig.update_layout(title="Histogram of Average Review Time", title_x=0.25, height=300, paper_bgcolor='#6495ED', plot_bgcolor='#8FBC8F', showlegend=False, margin=dict(l=0, r=20, t=50, b=100))
    return fig

def generate_bar_chart(reviews_sum):
    fig = go.Figure()

    # Assign colors
    colors = ['#2F4F4F' if status == 'closed' else '#FFA07A' for status in reviews_sum.index]

    fig.add_trace(go.Bar(
        x=reviews_sum.values, 
        y=reviews_sum.index, 
        orientation='h', 
        name='Reviews',
        marker=dict(color=colors) 
    ))

    fig.update_layout(
        title="Total Reviews by Status", 
        title_x=0.25, 
        height=400, 
        paper_bgcolor='#6495ED', 
        plot_bgcolor='#8FBC8F',
        legend=dict(orientation='h',yanchor='bottom',y=-0.3,xanchor='center',x=0.5),
        showlegend=True  
    )

    return fig




def generate_pie_chart(labels, values, title):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title=title, 
        title_x=0.25, 
        height=400, 
        paper_bgcolor='#6495ED', 
        plot_bgcolor='#8FBC8F',
        legend=dict(orientation='h',yanchor='bottom',  y=-0.3,  xanchor='center',x=0.5)
    )
    fig.update_traces(textinfo='label+percent', pull=[0.1] * len(labels))
    return fig


def generate_scatter_plot(df, x, y, size):
    fig = px.scatter(df, x=x, y=y, size=size)
    fig.update_layout(title='Avg Commit(f) vs Avg Review(ts)', title_x=0.25, paper_bgcolor='#6495ED', height=400, plot_bgcolor='#8FBC8F', margin=dict(l=0, r=20, t=50, b=100), xaxis_title='Average Commit Count', yaxis_title='Average Review Time')
    fig.update_yaxes(title_standoff=9, title_font=dict(size=14))
    return fig

def generate_distplot(hist_data, group_labels):
    fig = ff.create_distplot(hist_data, group_labels)
    fig.update_layout(title='Dist of Avg Commit Count', title_x=0.25, height=400, paper_bgcolor='#6495ED', plot_bgcolor='#8FBC8F', showlegend=False)
    return fig
