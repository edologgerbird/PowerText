# import libraries
import praw
from praw.models import MoreComments
import pandas as pd
import time
import dotenv
import os

# load environment variables
dotenv.load_dotenv()

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")

REDDIT_SCRAPED_DATA_PATH = "analysis_system/data_store/scraped/"

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


def reddit_scrapper(reddit, subreddit_lists, limit=5, comment_limit=5, topic="reddit_output"):
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
        count_posts = 0
        for post in hot_posts:
            print("=> post no. " + str(count_posts + 1))
            author = post.author
            timestamp = post.created_utc
            body = post.selftext
            id = post.id
            records.append([topic, thread, id, author, timestamp, body])

            # iterate by comments
            count_comments = 0
            for comment in post.comments:
                if isinstance(comment, MoreComments):
                    continue
                print("=>=> comment no. " + str(count_comments + 1))
                author = comment.author
                timestamp = comment.created_utc
                body = comment.body
                id = comment.id
                records.append([topic, thread, id, author, timestamp, body])
                count_comments += 1
                if count_comments >= comment_limit:
                    break

            count_posts += 1

    # convert records to dataframe and output in Excel
    output = pd.DataFrame(
        records, columns=["topic", "thread", "id", "author", "timestamp", "body"])
    output_timestamp = str(int(time.time()))
    output["scraped_at"] = output_timestamp

    # concat to existing file if exists
    try:
        reddit_store = pd.read_csv(
            f"{REDDIT_SCRAPED_DATA_PATH}reddit_store.csv")
        print("INFO: reddit_store.csv exists, appending to existing file")
    except:
        reddit_store = pd.DataFrame()
        print("INFO: reddit_store.csv does not exist, creating new file")
    reddit_store = pd.concat([reddit_store, output], axis=0).drop_duplicates(
        'id').reset_index(drop=True)
    reddit_store.to_csv(
        f"{REDDIT_SCRAPED_DATA_PATH}reddit_store.csv", index=False)


if __name__ == "__main__":
    reddit_agent = praw.Reddit(
        client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    reddit_scrapper(reddit_agent, [
                    "school", "food", "travel"], limit=10, comment_limit=10, topic="analysis-demo")


'''
# working example
reddit_agent = praw.Reddit(client_id='my_client_id', client_secret='my_client_secret', user_agent='my_user_agent')
reddit_scrapper(reddit_agent, ["MachineLearning", "learnmachinelearning", "nlp", "GPT"], limit=10, comment_limit=10, topic="machinemind")
'''
