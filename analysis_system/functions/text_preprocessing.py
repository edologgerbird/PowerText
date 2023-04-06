# importing relevant modules

import re
import string
import numpy as np
import pandas as pd
import nltk

# nltk.download('stopwords')
# stop_words = nltk.corpus.stopwords.words('english')


def load_stopwords():
    try:
        with open('cache_files/stopwords.txt', "r") as word_list:
            stopwords = word_list.read().split('\n')
    except:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')
        with open('cache_files/stopwords.txt', "w") as word_list:
            for word in stopwords:
                word_list.write(word + '\n')
    return stopwords

# Removing punctuations

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# Coverting to lowercase

def to_lowercase(text):
    return text.lower()

# Removing non-alphanumeric characters

def remove_non_alphanumeric(text):
    return re.sub(r'\W+', ' ', text)

# Removing stopwords

def remove_stopwords(text):
    stopwords = load_stopwords()
    text = text.split()
    text = [word for word in text if word not in stopwords]
    return " ".join(text)

# Removing numbers

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# Removing words with length less than 2

def remove_words_with_length_less_than_2(text):
    text = text.split()
    text = [word for word in text if len(word) > 1]
    return " ".join(text)

# Removing extra spaces

def remove_extra_spaces(text):
    return " ".join(text.split())

# Removing extra newlines

def remove_extra_newlines(text):
    return re.sub(r'    ', '    ', text)

# Removing extra tabs

def remove_extra_tabs(text):
    return re.sub(r'  ', '  ', text)

# Removing punctuations

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# Lemmatizing text

def lemmatize_text(text):
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
    stemmer = nltk.stem.PorterStemmer()
    text = text.split()
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

# Converting text to a list of paragraphs

def split_text_to_para(text):
    return text.split("\n\n")

# Converting text to a list of sentences

def split_text_to_sentences(text):
    return nltk.sent_tokenize(text)

# Converting text to a list of words

def split_text_to_words(text):
    return nltk.word_tokenize(text)

# Text summarization

def summarize_text(text, ratio=0.2):
    return nltk.summarize(text, ratio=ratio)

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