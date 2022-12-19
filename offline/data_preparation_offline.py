import os
from bs4 import BeautifulSoup 
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow import keras
import re
import joblib 

from gensim.parsing.preprocessing import remove_stopwords

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils.class_weight import compute_class_weight

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from river import imblearn, optim
from river import compose, linear_model, metrics, preprocessing
from river.compat import convert_sklearn_to_river

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

def stemSentence(sentence):
    lemmatizer=WordNetLemmatizer()
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(lemmatizer.lemmatize(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def preprocess_data(df):
#   TITLE CLEAN
    df['title_clean'] = df['title'].loc[df['title'] ==  ' [removed] '] = ' '
    df['title_clean'] = df['title'].str.lower()
    df['title_clean'] = df['title_clean'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df['title_clean'] = df['title_clean'].apply(lambda elem: re.sub(r"\d+", "", elem))
    # remove duplicate spaces
    df['title_clean'] = df['title_clean'].apply(lambda elem: re.sub(' +', ' ', elem))
    # remove stop words
    df['title_clean'] = df['title_clean'].apply(lambda elem: remove_stopwords(elem))
    df['title_clean'] = df['title_clean'].apply(lambda elem: stemSentence(elem))
    
#   TEXT CLEAN
    df['text_clean'] = df['text'].loc[df['title'] ==  ' [removed] '] = ' '
    df['text_clean'] = df['text'].str.lower()
    df['text_clean'] = df['text_clean'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df['text_clean'] = df['text_clean'].apply(lambda elem: re.sub(r"\d+", "", elem))
    # remove duplicate spaces
    df['text_clean'] = df['text_clean'].apply(lambda elem: re.sub(' +', ' ', elem))
    # remove stop words
    df['text_clean'] = df['text_clean'].apply(lambda elem: remove_stopwords(elem))
    df['text_clean'] = df['text_clean'].apply(lambda elem: stemSentence(elem))
    
    df['final_text'] = df['title_clean'] + df['text_clean']
    
    final_dataset = pd.DataFrame(df['subject_id'])
    final_dataset['text'] = df['title_clean'] + ' ' + df['final_text']
    if 'label' in df.columns:
        final_dataset['label'] = df['label']
    
    return final_dataset

def hash_data(df, vectorizer, isTrainMode=True):
    if 'label' in df.columns:
        labels = df['label']
    else:
        labels = []

    if isTrainMode:
        vectorizer.partial_fit(df['text'].values)
        data_vectorized = vectorizer.transform(df['text'].values)
        return data_vectorized, labels, vectorizer
    else:
        data_vectorized = vectorizer.transform(df['text'].values)
        return data_vectorized, labels


def bert_vectorize_data(df, vectorizer, isTrainMode=True):
    if 'label' in df.columns:
        labels = df['label']
    else:
        labels = []

    if isTrainMode:
        vectorizer.fit(df.text, df.label, class_weight={0:0.5,1:3.8}, epochs=5, verbose=0)
        vectroizer_layer = Model(vectorizer.input, outputs=vectorizer.get_layer('features').output)
        data_vectorized = vectroizer_layer.predict(df.text)
        return data_vectorized, labels, vectroizer_layer

    else:
        data_vectorized = vectorizer.predict(df.text, verbose=0)
        return data_vectorized, labels

def vectorize_data(df, vectorizer, vec_type, isTrainMode=True):
    # Use Hashing Vectorizer
    if vec_type == 'Hash':
        if isTrainMode:
            data, labels, vectorizer = hash_data(df, vectorizer, isTrainMode)
            return data, labels, vectorizer
        else:
            data, _ = hash_data(df, vectorizer, isTrainMode)
            return data
    # User BERT as vectorizer
    elif vec_type == 'BERT':
        if isTrainMode:
            data, labels, vectorizer = bert_vectorize_data(df, vectorizer,isTrainMode)
            return data, labels, vectorizer
        else:
            data, _  = bert_vectorize_data(df, vectorizer,isTrainMode)
        return data
        