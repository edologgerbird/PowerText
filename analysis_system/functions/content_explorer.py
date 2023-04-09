import pandas as pd
import numpy as np
import streamlit as st
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.metric_cards import style_metric_cards
import altair as alt
import utils.utility as util
import utils.design_format as format


def load_data(time_delta=24*60*60):
    '''This function loads the data from the cache file. If the cache file is not present, it downloads the data from the data folder.

    Args:
        time_delta (int): The time delta in seconds.

    Returns:
        reddit_content (pd.DataFrame): The reddit content.
    '''

    # load data
    reddit_content = st.session_state["csv_file"]

    # filter out posts older than 24 hours
    reddit_content = reddit_content[reddit_content["timestamp"] > (
        reddit_content["timestamp"].max() - time_delta)]

    return reddit_content


def display_overall_stats(reddit_content):
    '''This function displays the overall stats of the reddit content.
    
    Args:
        reddit_content (pd.DataFrame): The reddit content.
        
    Returns:
        None
    '''

    n_rows = reddit_content.shape[0]
    n_subreddits = len(get_unique_subreddits(reddit_content))

    # Display as metric cards
    columns = st.columns(2)
    columns[0].metric(label="Number of Posts", value=n_rows)
    columns[1].metric(label="Number of Subreddits", value=n_subreddits)


def get_unique_subreddits(reddit_content):
    '''This function returns the unique subreddits in the reddit content.
    
    Args:
        reddit_content (pd.DataFrame): The reddit content.
        
    Returns:
        unique_subreddits (list): The unique subreddits.
    '''

    unique_subreddits = reddit_content["thread"].unique()
    return unique_subreddits


def display_subreddit_stats(reddit_content):
    '''This function displays the stats of the subreddits in the reddit content.
    
    Args:
        reddit_content (pd.DataFrame): The reddit content.
        
    Returns:
        None
    '''

    # display subreddit stats
    subreddit_stats = reddit_content.groupby(
        "thread").count()["id"].reset_index()
    subreddit_stats.columns = ["Subreddit", "Number of Posts"]
    subreddit_stats = subreddit_stats.sort_values(
        "Number of Posts", ascending=False)
    subreddit_stats = subreddit_stats.reset_index(drop=True)

    # Display as metric cards
    n_subreddits = subreddit_stats.shape[0]
    columns = st.columns(n_subreddits)
    for idx, col in enumerate(columns):
        col.metric(label=subreddit_stats["Subreddit"][idx],
                   value=subreddit_stats["Number of Posts"][idx])

    # Visualise as altair chart
    st.write("")
    st.altair_chart(alt.Chart(subreddit_stats).mark_bar().encode(
        x=alt.X("Subreddit", sort='-y'),
        y="Number of Posts",
        color="Subreddit"
    ), use_container_width=True)

    return


def visualise_reddit_content(reddit_content):
    '''This function visualises the reddit content.
    
    Args:
        reddit_content (pd.DataFrame): The reddit content.
        
    Returns:
        None
    '''

    filtered_content = dataframe_explorer(reddit_content)
    st.dataframe(filtered_content)


# show random post from a given subreddit

def show_random_post_from_subreddit(reddit_content, subreddit):
    '''This function shows a random post from a given subreddit.
    
    Args:
        reddit_content (pd.DataFrame): The reddit content.
        
    Returns:
        None
    '''

    # filter out posts from a given subreddit
    filtered_content = reddit_content[reddit_content["thread"] == subreddit]

    # select a random post
    random_post = filtered_content.sample(1)
    reddit_card(random_post)
    return random_post


def generate_random_post(reddit_content):
    '''This function generates a random post.
    
    Args:
        reddit_content (pd.DataFrame): The reddit content.
        
    Returns:
        random_post (pd.DataFrame): The random post.
    '''

    random_post = reddit_content.sample(1)
    return random_post


def reddit_card(content):
    '''This function displays the content of a reddit post.
    
    Args:
        content (pd.DataFrame): The reddit content.
        
    Returns:
        None
    '''
    
    st.subheader(content["thread"].values[0])
    st.write(content["body"].values[0])
    st.caption(content["author"].values[0])
