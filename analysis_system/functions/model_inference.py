import pandas as pd
import numpy as np
import pickle
import streamlit as st
import altair as alt
import functions.text_preprocessing as text_preprocessing
import functions.content_explorer as content_explorer
import utils.design_format as format
import os 

path = os.path.dirname(__file__)


def load_model_dict():

    # load model dictionary
    model_dict = pickle.load(open("PassiveAggressiveClassifier_model_dict.pkl", "rb"))

    return model_dict

def get_targets():
    # load model dict
    model_dict = load_model_dict()

    targets = model_dict["target_list"]

    return targets


def predict_targets(reddit_content):
    # load model dict
    model_dict = load_model_dict()

    model = model_dict["model"]
    vectorizer = model_dict["vectorizer"]
    targets = model_dict["target_list"]

    # vectorize text
    vectorized_text = vectorizer.transform(reddit_content["body"])

    # predict targets
    predictions = model.predict(vectorized_text)

    # convert predictions to dataframe
    predictions = pd.DataFrame(predictions, columns=targets)

    reddit_content = reddit_content.reset_index(drop=True)
    # merge predictions with reddit_content
    reddit_content = pd.concat([reddit_content, predictions], axis=1)

    return reddit_content

def get_posts_without_violation(predicted_content):
    # load model dict
    model_dict = load_model_dict()

    targets = model_dict["target_list"]

    # get posts without violation
    posts_without_violation = predicted_content[predicted_content[targets].sum(axis=1) == 0]

    return posts_without_violation

def get_posts_with_violation(predicted_content, target):

    # get posts without violation
    posts_with_violation = predicted_content[predicted_content[target] == 1]

    return posts_with_violation

def display_post_from_df(predicted_content, n_post):
    n_post = min(n_post, predicted_content.shape[0])
    if n_post == 0:
        st.write("No posts found")
    for i in range(n_post):
        st.write(predicted_content["body"].values[i])
        st.caption(predicted_content["author"].values[i])
        format.horizontal_line()



def display_overall_prediction_stats(predicted_content):
    n_rows = predicted_content.shape[0]
    targets = load_model_dict()["target_list"]
    
    n_violations_by_target = predicted_content[targets].sum(axis=0)
    n_violations_total = int(n_violations_by_target.sum())

    # Display as metric cards
    columns = st.columns(2)
    columns[0].metric(label="Number of Posts Predicted", value=n_rows)
    columns[1].metric(label="Number of Violations", value=n_violations_total)

def display_prediction_stats_by_target(predicted_content):
    targets = load_model_dict()["target_list"]
    n_targets = len(targets)

    # display subreddit stats
    target_stats = predicted_content[targets].sum().reset_index()
    target_stats.columns = ["Target", "Number of Violations"]
    target_stats = target_stats.sort_values(
        "Number of Violations", ascending=False)
    target_stats = target_stats.reset_index(drop=True)

    # Display as metric cards
    columns = st.columns(n_targets)
    for idx, col in enumerate(columns):
        col.metric(label=target_stats["Target"][idx],
                   value=int(target_stats["Number of Violations"][idx]))
        
    # Visualise as altair chart
    st.write("")
    st.altair_chart(alt.Chart(target_stats).mark_bar().encode(
        x=alt.X("Target", sort='-y'),
        y="Number of Violations",
        color = alt.Color("Target", legend=None),
    ), use_container_width=True)

    return