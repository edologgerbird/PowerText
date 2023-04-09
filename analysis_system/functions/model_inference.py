import pandas as pd
import numpy as np
import pickle
import streamlit as st
import altair as alt
import functions.text_preprocessing as text_preprocessing
import functions.content_explorer as content_explorer
import utils.design_format as format
import os

deployed_on_st = True
if deployed_on_st:
    path = "analysis_system/"
else:
    path = ""


def load_model_dict():
    '''This function loads the model dictionary from the cache file. If the cache file is not present, it downloads the model dictionary from the models folder.

    Returns:
        model_dict (dict): The model dictionary.
    '''

    # load model dictionary
    model_dict = pickle.load(
        open(f"{path}models/PassiveAggressiveClassifier_model_dict.pkl", "rb"))
    # model_dict = pickle.load(open("analysis_system/models/PassiveAggressiveClassifier_model_dict.pkl", "rb"))

    return model_dict


def get_targets():
    '''This function returns the list of targets.

    Returns:
        targets (list): The list of targets.
    '''

    # load model dict
    model_dict = load_model_dict()

    targets = model_dict["target_list"]

    return targets


def predict_targets(reddit_content):
    '''This function predicts the targets for the reddit content.

    Args:
        reddit_content (pd.DataFrame): The reddit content.

    Returns:
        reddit_content (pd.DataFrame): The reddit content with the predicted targets.
    '''

    # load model dict
    model_dict = load_model_dict()

    model = model_dict["model"]
    vectorizer = model_dict["vectorizer"]
    targets = model_dict["target_list"]

    # vectorize text
    vectorized_text = vectorizer.transform(reddit_content["body_processed"])

    # predict targets
    predictions = model.predict(vectorized_text)

    # convert predictions to dataframe
    predictions = pd.DataFrame(predictions, columns=targets)

    reddit_content = reddit_content.reset_index(drop=True)
    # merge predictions with reddit_content
    reddit_content = pd.concat([reddit_content, predictions], axis=1)

    return reddit_content


def get_posts_without_violation(predicted_content):
    '''This function returns the posts without violation.

    Args:
        predicted_content (pd.DataFrame): The predicted content.

    Returns:
        posts_without_violation (pd.DataFrame): The posts without violation.
    '''

    # load model dict
    model_dict = load_model_dict()

    targets = model_dict["target_list"]

    # get posts without violation
    posts_without_violation = predicted_content[predicted_content[targets].sum(
        axis=1) == 0]

    return posts_without_violation


def get_posts_with_violation(predicted_content, target):
    '''This function returns the posts with violation.

    Args:
        predicted_content (pd.DataFrame): The predicted content.
        target (str): The target.

    Returns:
        posts_with_violation (pd.DataFrame): The posts with violation.
    '''

    # get posts without violation
    posts_with_violation = predicted_content[predicted_content[target] == 1]

    return posts_with_violation


def display_post_from_df(predicted_content, n_post):
    '''This function displays the posts from the dataframe.

    Args:
        predicted_content (pd.DataFrame): The predicted content.
        n_post (int): The number of posts to be displayed.

    Returns:
        None
    '''

    n_post = min(n_post, predicted_content.shape[0])
    feedback_content = predicted_content.copy()

    if n_post == 0:
        st.write("No posts found")
    for i in range(n_post):
        st.subheader(f"Post {i+1}")
        st.write(predicted_content["body"].values[i])
        st.caption(predicted_content["author"].values[i])

        # Feedback mechanism
        with st.expander("This post was wrongly classified!"):
            feedback_encoded = feedback_on_post(i)
            if len(feedback_encoded):
                if st.button("Submit Feedback", key=(i+1)*1000):
                    row_concern = predicted_content.iloc[i]
                    row_concern[get_targets()] = feedback_encoded
                    row_concern_df = pd.DataFrame(row_concern).T
                    save_feedback_to_datastore(row_concern_df)
                    st.success(
                        "Feedback submitted! This will be used in the next model re-training cycle.")

        format.horizontal_line()


def save_feedback_to_datastore(feedback):
    '''This function saves the feedback to the data store.

    Args:
        feedback (pd.DataFrame): The feedback.

    Returns:
        None
    '''

    try:
        feedback_df = pd.read_csv(f"{path}data_store/feedback/feedback.csv")
        feedback_df = pd.concat([feedback_df, feedback], axis=0)
        feedback_df.to_csv(
            f"{path}data_store/feedback/feedback.csv", index=False)

    except:
        feedback.to_csv(f"{path}data_store/feedback/feedback.csv", index=False)
    return


def feedback_on_post(i):
    ''' This function displays the feedback mechanism for the post.

    Args:
        i (int): The index of the post.

    Returns:
        feedback_encoded (list): The encoded feedback.
    '''

    targets = get_targets()
    feedback = st.multiselect(
        "Select the correct target(s). Leave blank if no violation.", targets, key=i)
    feedback_encoded = []
    for target in targets:
        if target in feedback:
            feedback_encoded.append(1)
        else:
            feedback_encoded.append(0)
    return feedback_encoded


def display_overall_prediction_stats(predicted_content):
    '''This function displays the overall prediction stats.

    Args:
        predicted_content (pd.DataFrame): The predicted content.

    Returns:
        None
    '''

    n_rows = predicted_content.shape[0]
    targets = load_model_dict()["target_list"]

    n_violations_by_target = predicted_content[targets].sum(axis=0)
    n_violations_total = int(n_violations_by_target.sum())

    # Display as metric cards
    columns = st.columns(3)
    columns[0].metric(label="Number of Posts Predicted", value=n_rows)
    columns[1].metric(label="Number of Predicted Violations",
                      value=n_violations_total)
    columns[2].metric(label="Percentage of Predicted Violations",
                      value=f"{round(n_violations_total/n_rows*100, 2)}%")


def display_prediction_stats_by_target(predicted_content):
    '''This function displays the prediction stats by target.

    Args:
        predicted_content (pd.DataFrame): The predicted content.

    Returns:
        None
    '''

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
        color=alt.Color("Target", legend=None),
    ), use_container_width=True)

    return
