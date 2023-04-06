# Importing relevant modules

import streamlit as st
import utils.utility as util
import utils.design_format as format
import functions.model_inference as model_inference
import functions.text_preprocessing as text_preprocessing
import functions.content_explorer as content_explorer


st.title("🔮 Content Classification")
format.horizontal_line()
format.align_text("In this page, we will classify the Reddit posts from the past 24 hours, based on their Terms-of-Service violations.", "justify")
format.align_text("For this demonstration, we trained and deployed a PassiveAggressiveClassifier to classify the scraped content.", "justify")
st.write("")
st.write("")

def run():
    # Load data
    reddit_content = content_explorer.load_data()

    # Preprocess text
    reddit_content["body"] = reddit_content["body"].apply(text_preprocessing.preprocess_text)

    # Predict targets
    predicted_outputs = model_inference.predict_targets(reddit_content)

    with st.expander("View Raw Data with Predicted Output"):
        content_explorer.visualise_reddit_content(predicted_outputs)

    format.horizontal_line()

    # Display overall stats

    st.subheader("Overall Statistics")
    st.write("Here is an overall summary of the predicted class of Reddit posts from the past 24 hours: ")
    st.write("")
    model_inference.display_overall_prediction_stats(predicted_outputs)

    format.horizontal_line()

    # Display statistics by target and subreddit
    st.subheader("Target Statistics by Subreddit")
    st.write("Here is an breakdown of the predicted class of Reddit posts from the past 24 hours: ")
    st.write("")
    selection_options = ["All Subreddits"] + list(content_explorer.get_unique_subreddits(reddit_content))
    subreddit = st.selectbox("Select Subreddit", selection_options)
    st.write("")
    if subreddit == "All Subreddits":
        model_inference.display_prediction_stats_by_target(predicted_outputs)
    else:
        model_inference.display_prediction_stats_by_target(predicted_outputs[predicted_outputs["thread"] == subreddit])

    format.horizontal_line()

    # Viewing violated posts by target

    st.subheader("View Violated Posts by Target")
    st.write("Here are posts that violated respective Terms-of-Services: ")
    st.write("")
    
    targets = ["No Violations"] + model_inference.get_targets()

    target = st.selectbox("Select Target", targets)
    st.write("")
    n_post = st.slider("Number of Posts to Display", min_value=1, max_value=30, value=10)

    show_posts = st.button("Show")

    if show_posts:
        format.horizontal_line()
        if target == "No Violations":
            filtered_output = model_inference.get_posts_without_violation(predicted_outputs)
            model_inference.display_post_from_df(filtered_output, n_post)
        else:
            filtered_output = model_inference.get_posts_with_violation(predicted_outputs, target)
            model_inference.display_post_from_df(filtered_output, n_post)
    



    
    return

if __name__ == "__main__":
    # try:
    run()
    # except:
    #     util.page_under_construction("Content Classification")
