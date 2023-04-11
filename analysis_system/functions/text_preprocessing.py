# importing relevant modules

import re
import string
import numpy as np
import pandas as pd
import nltk
import streamlit as st


deployed_on_st = True
if deployed_on_st:
    path = "analysis_system/"
else:
    path = ""


def load_stopwords():
    '''This function loads the stopwords from the cache file. If the cache file is not present, it downloads the stopwords from nltk.

    Returns:
        stopwords (list): The list of stopwords.
    '''

    try:
        with open(f'{path}cache_files/stopwords.txt', "r") as word_list:
            stopwords = word_list.read().split('\n')
    except:
        with st.spinner("Downloading stopwords..."):
            nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')
        with open(f'{path}cache_files/stopwords.txt', "w") as word_list:
            word_list.write('\n'.join(stopwords))

    return stopwords


# Removing punctuations

def remove_punctuations(text):
    '''This function removes the punctuations from the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# Coverting to lowercase


def to_lowercase(text):
    '''This function converts the text to lowercase.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    return text.lower()

# Removing non-alphanumeric characters


def remove_non_alphanumeric(text):
    '''This function removes the non-alphanumeric characters from the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    return re.sub(r'\W+', ' ', text)

# Removing stopwords


def remove_stopwords(text):
    '''This function removes the stopwords from the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    stopwords = load_stopwords()
    text = text.split()
    text = [word for word in text if word not in stopwords]
    return " ".join(text)

# Removing numbers


def remove_numbers(text):
    '''This function removes the numbers from the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    return re.sub(r'\d+', '', text)

# Removing words with length less than 2


def remove_words_with_length_less_than_2(text):
    '''This function removes the words with length less than 2 from the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    text = text.split()
    text = [word for word in text if len(word) > 1]
    return " ".join(text)

# Removing extra spaces


def remove_extra_spaces(text):
    '''This function removes the extra spaces from the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    return " ".join(text.split())

# Removing extra newlines


def remove_extra_newlines(text):
    '''This function removes the extra newlines from the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    return re.sub(r'    ', '    ', text)

# Removing extra tabs


def remove_extra_tabs(text):
    '''This function removes the extra tabs from the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    return re.sub(r'  ', '  ', text)

# Removing punctuations


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# Lemmatizing text


def lemmatize_text(text):
    '''This function lemmatizes the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    try:
        lemmatizer = nltk.stem.WordNetLemmatizer()
    except:
        nltk.download('wordnet')
        lemmatizer = nltk.stem.WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

# Stemming text


def stem_text(text):
    '''This function stems the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    stemmer = nltk.stem.PorterStemmer()
    text = text.split()
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

# Text Preprocessing Pipeline


def preprocess_text(text,
                    _remove_punctuations=True,
                    _to_lowercase=True,
                    _remove_non_alphanumeric=True,
                    _remove_stopwords=True,
                    _remove_numbers=True,
                    _remove_extra_spaces=True,
                    _remove_extra_newlines=True,
                    _remove_extra_tabs=True,
                    _lemmatize_text=True,
                    _stem_text=True,
                    _remove_words_with_length_less_than_2=True):
    '''This function preprocesses the text.

    Args:
        text (str): The text to be processed.

    Returns:
        text (str): The processed text.
    '''

    text = str(text)

    if _remove_punctuations:
        text = remove_punctuations(text)
    if _to_lowercase:
        text = to_lowercase(text)
    if _remove_non_alphanumeric:
        text = remove_non_alphanumeric(text)
    if _remove_stopwords:
        text = remove_stopwords(text)
    if _remove_numbers:
        text = remove_numbers(text)
    if _remove_extra_spaces:
        text = remove_extra_spaces(text)
    if _remove_extra_newlines:
        text = remove_extra_newlines(text)
    if _remove_extra_tabs:
        text = remove_extra_tabs(text)
    if _lemmatize_text:
        text = lemmatize_text(text)
    if _stem_text:
        text = stem_text(text)
    if _remove_words_with_length_less_than_2:
        text = remove_words_with_length_less_than_2(text)
    return text


# Testing
if __name__ == "__main__":
    text = "This is a sample text. It contains numbers 123, punctuations !@#$%^&*()_+, non-alphanumeric characters #@$%^&*()_+, stopwords like the, a, and, and extra spaces. It also contains extra newlines   and extra tabs  ."
    print(preprocess_text(text))
