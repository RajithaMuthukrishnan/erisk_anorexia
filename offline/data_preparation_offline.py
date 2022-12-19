import numpy as np
import pandas as pd
import re

from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import HashingVectorizer

from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from official.nlp import optimization

import warnings
warnings.filterwarnings('ignore')

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
        