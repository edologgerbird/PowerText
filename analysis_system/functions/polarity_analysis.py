import pandas as pd
import numpy as np
import pickle
import streamlit as st
import altair as alt
import functions.text_preprocessing as text_preprocessing
import functions.content_explorer as content_explorer
import utils.design_format as format
from textblob import TextBlob


def polarity_analysis(text):
    polarity = TextBlob(text).sentiment.polarity
    return polarity

def visualise_polarity(content):
    # visualise polarity
    polarity_chart = alt.Chart(content).mark_bar().encode(
        x=alt.X("polarity"),
        y="count()",
        tooltip=["count()"]
    )

    st.altair_chart(polarity_chart, use_container_width=True)

    return

def display_polarity_stats(content, n_decimals=5):
    # Mean and SD
    mean_polarity = content["polarity"].mean()
    sd_polarity = content["polarity"].std()

    # Display as metric cards
    columns = st.columns(2)
    if mean_polarity > 0:
        help_text = "A polarity score above 0 indicates a generally positive sentiment."
    else:
        help_text = "A polarity score below 0 indicates a generally negative sentiment."
    columns[0].metric(label="Mean Polarity", value=round(mean_polarity, n_decimals), help=help_text)
    columns[1].metric(label="SD Polarity", value=round(sd_polarity, n_decimals), help="A high SD indicates a lot of variation in the polarity scores.")

    return

def display_polarity_stats_by_subreddit(content, subreddit, n_decimals=5):
    # filter by subreddit
    content_filtered = content[content["thread"] == subreddit]

    display_polarity_stats(content_filtered, n_decimals=n_decimals)

    return

def visualise_polarity_by_subreddit(content, subreddit):
    # filter by subreddit
    content_filtered = content[content["thread"] == subreddit]

    # visualise polarity
    visualise_polarity(content_filtered)

    return