# Importing relevant modules

import streamlit as st
import utils.utility as util
import utils.design_format as format
import functions.content_explorer as content_explorer
import functions.text_preprocessing as text_preprocessing
import functions.polarity_analysis as polarity_analysis

st.title("😂 Sentiment Polarity Analysis")
format.horizontal_line()

def run():
    if util.check_session_state_key("csv_file"):
        # Load data
        reddit_content = content_explorer.load_data()

        # Preprocess text
        reddit_content["body"] = reddit_content["body"].apply(text_preprocessing.preprocess_text)

        # Calculate polarity
        reddit_content["polarity"] = reddit_content["body"].apply(polarity_analysis.polarity_analysis)

        st.subheader("Overall Polarity")
        st.write("Here is an overall polarity of the Reddit posts from the past 24 hours: ")
        st.write("")

        # Display overall polarity
        polarity_analysis.display_polarity_stats(reddit_content)
        # Visualise Polarity
        polarity_analysis.visualise_polarity(reddit_content)

        format.horizontal_line()

        # Polarity analysis by subreddit
        st.subheader("Polarity by Subreddit")
        st.write("Here is a summary of the polarity of individual Subreddits from the past 24 hours:")
        st.write("")

        # display subreddit stats
        selection_options = list(content_explorer.get_unique_subreddits(reddit_content))
        subreddit = st.selectbox("Select Subreddit", selection_options)
        st.write("")

        # Display polarity by subreddit
        polarity_analysis.display_polarity_stats_by_subreddit(reddit_content, subreddit)

        # Visualise Polarity by subreddit
        polarity_analysis.visualise_polarity_by_subreddit(reddit_content, subreddit)
    else:
        util.no_file_uploaded()

        

if __name__ == "__main__":
    # try:
    #     run()
    # except:
    #     util.page_under_construction("Sentiment Polarity Analysis")
    run()
