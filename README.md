<!-- PROJECT LOGO -->

<div align="center">
  <a>
    <img src="read_me_files/logo/logo-color.png" alt="Logo" width="300">
  </a>
    <br/>
    <br/>
  <h3 align="center">PowerText</h3>

  <p align="center">
    Automatic Content Modereation for Text in Social Media: Term of Service violations and AI content detection.
    <br />
    <a href=""><strong>View the Full Project Report »</strong></a>
    <br />
    <a href="https://edologgerbird-is4242-group8-analysis-systemhome-edoautom-esii7q.streamlit.app/"><strong>View the Demo Application for Content Regulators »</strong></a>
    <br />
    <a href=""><strong>View the Demo Application for Common Users » »</strong></a>
    <br />
  </p>
</div>

<!-- LEGAL DISCLAIMERS-->

This project was created using publicy available APIs and was created for educational reasons.
Contents of this project should ONLY be used for <strong>NON-COMMERICAL</strong> reasons.

<!-- TABLE OF CONTENTS -->

### Table of Contents

<ol>
<li><a href="#project-overview">Project Overview</a></li>
<li><a href="#authors">Authors</a></li>
<li><a href="#codes-and-resources-used">Codes and Resources Used</a></li>
<li><a href="#data-ingestion-sources">Data Ingestion Sources</a></li>
<li><a href="#getting-started">Getting Started</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#contact">Contact</a></li>
<li><a href="#acknowledgements">Acknowledgements</a></li>
</ol>

<br />

# PowerText Implementation

## Project Overiview:

The objective of this project is to develop an automated content moderator for text content in social media platforms. Our proposed system is designed to accurately identify and flag content that violates terms of service in various categories such as hate speech, cyberbullying, and advertisements, while also being capable of distinguishing between human-generated and AI-generated content. To achieve this, we will leverage four commonly used Natural Language Processing (NLP) algorithms, namely NaiveBayes, PassiveAggressive, XGBoost, CNN, LSTM, GRU, and Transformers.

For our models, we will utilize a self-tagged dataset scrapped from Reddit posts (A global social media platform with diverse user-generated content), and a high-quality dataset of AI-generated content using GPT-3.5 and GPT-4. This combined dataset ensures comprehensive training on a variety of real-world posts, ensuring accuracy, effectiveness, and applicability across all the domains for social media.

The system offers two distinct end products: an automated content collection and screening service for social-media platforms, and a user side plug-in for post/comments check. Successful implementation will bring huge benefits to social media platforms, content creators and users, fostering a safe and healthy online environment.


### _Solution Architecture:_

  <a>
    <img src="read_me_files/archi.png" alt="Solution Architecture" height="400">
  </a>

### _Keywords:_

_Data Pipeline, Sentiment Analysis, Transformers, Roberta BERT, hateBERT, CNN, LSTM, Hugging Face, Natural Language Processing, TOS Violation Analysis, Web Scraping, Data Visualisation_

<p align="right">(<a href="#top">back to top</a>)</p>

## Authors:

- Bikramjit Dasgupta 
- Lee Leonard
- Lin Yongqian
- Loh Hong Tak Edmund
- Tang Hanyang
- Tay Zhi Sheng
- Wong Deshun


<p align="right">(<a href="#top">back to top</a>)</p>

## Codes and Resources Used

**Python Version:** 3.9.10

**Built with:** [Microsoft Visual Studio Code](https://code.visualstudio.com/), [Google Colab](https://colab.research.google.com/), [Streamlit](https://streamlit.io/), [Git](https://git-scm.com/)

**Notable Packages:** praw, pandas, numpy, scikit-learn, xgboost, transformers, pytorch, torchvision, tqdm (view requirements.txt for full list)

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

### **Prerequisites**

Make sure you have installed all of the following on your development machine:

- Python 3.8.0 or above

<p align="right">(<a href="#top">back to top</a>)</p>

## **Installation**

We recommend setting up a virtual environment to run this project.


### _1. Python Virtual Environment_

Installing and Creation of a Virtual Environment

```sh
pip install virtualenv
virtualenv <your_env_name>
source <your_env_name>/bin/active
```

The requirements.txt file contains Python libraries that your notebooks depend on, and they will be installed using:

```sh
pip install -r requirements.txt
```

### _2. Google Colab Environment_

Our deep learning models were primarily trained on [Google Colab Pro](https://colab.research.google.com) due to the access to high performance GPUs required for the training of complex neural network systems. 

You can set up the Google Colab environment to run our model training codes by executing the following:

```py
from google.colab import drive
drive.mount('/content/drive')
content_path = "insert/path/to/your/data"
```
You will be prompted to log in with your Google Account. Simply replace the content_path with the path to directory where you have uploaded the data.

Alternatively, if you do not wish to authenticate with your Google Account, you may simply run the following code to retrieve the data from a permanent link:

```py
train_path = 'https://drive.google.com/uc?export=download&id=1ZTfYOXeZLW57mLR7IegIFovi7FW1chS6'
val_path = 'https://drive.google.com/uc?export=download&id=1ZMJI7DyKMLHpHp-HBUO64kWbP6A-qj9k'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
```

The required additional modules required for each ```.ipynb``` notebook runned on Google Colab have been included in each notebook to be installed using ```pip```.

<p align="right">(<a href="#top">back to top</a>)</p>

## Code Structure

> INSERT CODE STRUCTURE

## Source Layer

### Data Ingestion Sources

Our team extracted both structured and unstructred data from the following sources:

| Source | Description | Size |
| ----------- | ----------- | ----------- |
| [Reddit](https://www.reddit.com/) | Extracted using the [PRAW API](https://praw.readthedocs.io/en/stable/) endpoint | 16828 |
| [ChatGPT](https://chat.openai.com/chat) | Generated using ChatGPT 3.5 and 4 | 6043 |
| Confounding Dataset | Manually created to include confounding and additional hate & ads | 26676 |

### Reddit Scraper Agent

The Reddit scraper agent can be location in ``` WebScrapper/reddit_scrapper.py```

This script defines methods that scrapes posts and comments based on user-specified subreddits, post counts and comment counts.

To initiate a ```reddit_scraper``` instance, execute the following Python code:

```py
from reddit_scaper import reddit_scrapper

reddit_agent = praw.Reddit(client_id='my_client_id', client_secret='my_client_secret', user_agent='my_user_agent')
reddit_scrapper(reddit_agent, ['MachineLearning', 'learnmachinelearning', 'GPT'], limit=10, comment_limit=10, topic="machinemind")
```

> 🔍To generate and specify the client_id, client_secret and user_agent, please follow the steps detailed [here](https://www.geeksforgeeks.org/how-to-get-client_id-and-client_secret-for-python-reddit-api-registration/)

### Storing of Raw Data

Our scraped data are compiled as xlsx and csv files, and stored within the ```Dataset/``` folder, categorised into
- Reddit Content: ```Dataset/Reddit Tagged Content``` 
- AI Content: ```Dataset/AI Content``` 
- Confounding Content: ```Dataset/Additional Data``` 

<p align="right">(<a href="#top">back to top</a>)</p>

## Data Processing Layer

### Data Preprocessing Notebook

Following the compilation of data from the multiple sources as detailed in the previous section, we proceeded to process the data to prepare it for model training and exploratory data analysis.

The preprocessing pipeline consist of the following steps:
1. Concatenating all datasets into a single ```DataFrame```.
2. Setting up target columns for each data entry 
3. Text preprocessing:

    i. Remove Punctuations

    ii. Convert to lowercase

    iii. Remove non-alphanumeric characters
    
    iv. Remove stopwords

    v. Remove extra spaces, new lines, tabs
    
    vi. Lemmetize and Stem text

    vii. Remove words with length < 2

To run the Data Preprocessing step, simply run the ```text_preprocessing.ipynb``` notebook in ```data_processing/``` folder.

### Exploratory Data Analysis

With the processed data, we conducted a comprehensive Exploratory Data Analysis of the combined dataset. The analyses undertaken include:

1. Class distribution
2. N-gram analysis
3. Cluster analysis (Document and word)
4. Polarity analysis
5. Perplexity analysis
6. Burstiness analysis

To view the Data Preprocessing step, simply run the ```data_eda.ipynb``` notebook in ```data_processing/``` folder.

> ⚠️ Do remember to adjust the paths to where you store your dataset on your local machine!

## Development Layer

We proceed to build and train a suite of models to 

## Deployment Layer

## Feedback Layer









