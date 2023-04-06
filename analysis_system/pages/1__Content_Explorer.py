# Importing relevant modules

import streamlit as st
import utils.utility as util
import utils.design_format as format
import functions.content_explorer as content_explorer


st.title("🔍 Content Explorer")
format.horizontal_line()

def run():
    if util.check_session_state_key("csv_file"):
        # load data
        reddit_content = content_explorer.load_data()

        # Display overall stats
        st.subheader("Overall Statistics")
        st.write("Here is an overall summary of the Reddit posts from the past 24 hours: ")
        st.write("")
        content_explorer.display_overall_stats(reddit_content)
        format.horizontal_line()


        # display subreddit stats
        st.subheader("Subreddit Statistics")
        st.write("Here is a summary of the Subreddits from the past 24 hours:")
        st.write("")

        # display subreddit stats
        content_explorer.display_subreddit_stats(reddit_content)
        format.horizontal_line()

        # display random post
        st.subheader("View a Random Reddit Post")
        st.write("Here is a random post from the past 24 hours:")
        st.write("")
        col1, col2 = st.columns([4,1])
        with col1:
            subreddit = st.selectbox("Select Subreddit", content_explorer.get_unique_subreddits(reddit_content))
        with col2:
            st.write("")
            st.write("")
            generate_random = st.button("Generate")
        
                
        if generate_random:
            content_explorer.show_random_post_from_subreddit(reddit_content, subreddit)

        format.horizontal_line()

        # visualise data
        with st.expander("View Raw Data"):
            content_explorer.visualise_reddit_content(reddit_content)
    else:
        util.no_file_uploaded()

if __name__ == "__main__":
    run()
