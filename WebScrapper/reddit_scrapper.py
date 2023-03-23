# import libraries
import praw
from praw.models import MoreComments
import pandas as pd
import time

# set up reddit API account as a praw instance
'''
function reddit_agent_setup:

description: 
This function sets up reddit API account as a praw instance.

input:
- client_id: Reddit API app client id
- client_secret: Reddit API app client secret
- user_agent: Reddit API app name

output:
- praw instance for reddit API account
'''
def reddit_agent_setup(client_id, client_secret, user_agent):
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


# function to scrap reddit data
'''
function reddit_scrapper:

description: 
This function is used to scrap reddit data.

inputs:
- reddit: praw instance for reddit API account
- subreddit_lists: list of all subreddits required to scrap
- limit: maximum number of posts in each subreddit, default = 5
- topic: topic name and exported file name

outputs:
- an excel file stored as "{topic}{unix_timestamp}.xlsx" in current directory
'''
def reddit_scrapper(reddit, subreddit_lists, limit=5, topic="reddit_output"):
    # set list to store records
    records = []

    # iterate by subreddit
    for instance in subreddit_lists:
        subreddit = reddit.subreddit(instance)
        thread = subreddit.title

        # process update
        print("Scrapping subreddit: " + thread)

        # set limit of posts due to maximum request limit
        hot_posts = subreddit.hot(limit=limit)

        # iterate by posts
        for post in hot_posts:
            author = post.author
            timestamp = post.created_utc
            body = post.selftext
            records.append([topic, thread, author, timestamp, body])

            # iterate by comments
            for comment in post.comments:
                if isinstance(comment, MoreComments):
                    continue
                author = comment.author
                timestamp = comment.created_utc
                body = comment.body
                records.append([topic, thread, author, timestamp, body])

    # convert records to dataframe and output in excel
    output = pd.DataFrame(records, columns=["topic", "thread", "author", "timestamp", "body"])
    output_timestamp = str(int(time.time()))
    output_name = topic + output_timestamp + ".xlsx"
    output.to_excel(output_name)


'''
# working example
reddit_agent = praw.Reddit(client_id='my_client_id', client_secret='my_client_secret', user_agent='my_user_agent')
reddit_scrapper(reddit_agent, ["MachineLearning", "learnmachinelearning", "nlp", "GPT"], limit=6, topic="machinemind")
'''