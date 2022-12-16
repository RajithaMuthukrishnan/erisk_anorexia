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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse

from gensim.parsing.preprocessing import remove_stopwords

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from river import imblearn, optim
from river import compose, linear_model, metrics, preprocessing
from river.compat import convert_sklearn_to_river

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

import data_preparation 

def extract_train_filenames(path):
    files = [] 
    for filename in os.listdir(path):
        if not filename.endswith('.xml'):
            continue
        filepath = os.path.join(path, filename)
        files.append(filepath)
    return files

def extract_train_chunks(train_data_path):
    dataframe_collection = {} 
    for ctr in range(1,11):
        positive_file_path = str(train_data_path)+"/positive_examples/chunk"+str(ctr)
        negative_file_path = str(train_data_path)+"/negative_examples/chunk"+str(ctr)
        positive_files = extract_train_filenames(positive_file_path)
        negative_files = extract_train_filenames(negative_file_path)
        files = positive_files + negative_files
        data_list = []
        for file in files:
            if 'positive' in file:
                label = 1
            elif 'negative' in file:
                label = 0
            fd = open(file,'r')
            data = fd.read()
            soup = BeautifulSoup(data,'xml')
            subject_id = soup.find('ID')
            writings = soup.find_all('WRITING')
            title = ''
            text = ''
            for writing in writings:
                title = title + writing.find('TITLE').get_text() + ' '
                text = text + writing.find('TEXT').get_text() + ' '
                row = [subject_id.get_text(), title, text, label]
            data_list.append(row)
        chunk_name = 'chunk'+str(ctr)
        dataframe_collection[chunk_name] = pd.DataFrame(data_list, columns = ['subject_id', 'title', 'text', 'label'])
    return dataframe_collection


def validate(df, model, vectorizer):
    X_val, Y_val = data_preparation.vectorize_data(df, vectorizer, isTrainMode=False)
    
    print('\nClassification Report\n')
    if 'sklearn' in str(type(model)):
        predictions = model.predict(X_val)
    if 'river' in str(type(model)):
        predictions = model.predict_many(pd.DataFrame(X_val.toarray()))
    
    print ('F1 Score : {:.2f}'.format(f1_score(Y_val, predictions, average='weighted')))
    print()
    print(classification_report(Y_val, predictions, target_names=['Non-Anorexic', 'Anorexic']))

def build_bert_classifier():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(100, activation=None, name='features')(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    bert = tf.keras.Model(text_input, net)
    return bert

def train(train_df_collection, model, vectorizer, vec_type):
    print('-- TRAIN MODEL --')
    train_metrics_list = []

    for chunk in range (len(train_df_collection)):
        chunk_name = 'chunk'+str(chunk+1)
        df = data_preparation.preprocess_data(train_df_collection[chunk_name])
        
        print('Training ' + chunk_name + '...')

        X_train, Y_train, vectorizer = data_preparation.vectorize_data(df, vectorizer, vec_type=vec_type, isTrainMode=True)

        if 'sklearn' in str(type(model)):
            if 'LogisticRegression' in str(type(model)):
                model.fit(X_train, Y_train)
            elif 'SGDClassifier' in str(type(model)):
                if chunk_name == 'chunk1':
                    model.fit(X_train, Y_train)
                else:
                    model.partial_fit(X_train, Y_train)
            
            predictions = model.predict(X_train)
            
        elif 'river' in str(type(model)):
            model.learn_many(pd.DataFrame(X_train.toarray()), pd.Series(Y_train))
            predictions = model.predict_many(pd.DataFrame(X_train.toarray()))

        train_score = f1_score(Y_train, predictions, average='weighted')
        train_metrics_list.append(train_score)
        print ('F1 Score :',f1_score(Y_train, predictions, average=None))

    train_metrics_df = pd.DataFrame(train_metrics_list, columns=['F1_score'])
    return model, vectorizer

def save_model(model, name):
    joblib.dump(model, 'trained_models/'+name+'.joblib')

def save_vectorizer(vectorizer, name):
    joblib.dump(vectorizer, 'trained_models/'+name+'_vectorizer.joblib')

def save_vectorizer_bert(vectorizer, name):
    vectorizer.save('trained_models/'+name+'_vectorizer.h5')

def train_models(models_list, vectorize):
    for name, model in models_list.items():
        print('     ***** '+ name +' *****    ')
        if vectorize == 'BERT':
            bert_model = build_bert_classifier()
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = tf.metrics.BinaryAccuracy()

            init_lr = 3e-5
            optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                    num_train_steps=1000,
                                                    num_warmup_steps=100,
                                                    optimizer_type='adamw')
            bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            trained_model, vectorizer = train(train_dataframe_collection, model, bert_model, vec_type='BERT')
            model_name = name+'_BERT'
            save_vectorizer_bert(vectorizer, model_name)

        if vectorize == 'Hash':
            hasher = HashingVectorizer()
            trained_model, vectorizer = train(train_dataframe_collection, model, hasher, vec_type='Hash')
            model_name = name+'_Hash'
            save_vectorizer(vectorizer, model_name)

        save_model(trained_model, model_name)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-data', help='path to train data chunks', required=True, nargs=1, dest="train_data_path")
    args = parser.parse_args()

    train_data_path = args.train_data_path[0]

    bert_preprocesser = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='preprocessing')
    train_dataframe_collection = extract_train_chunks(train_data_path)
    class_weights = {0:0.57, 1:3.8}
    models_list = {
        # 'SGDClassifier': SGDClassifier(loss='log_loss', warm_start=True, class_weight={0:0.58,1:3.8}),
        'LogisticRegression': LogisticRegression(solver='lbfgs', class_weight='balanced', warm_start=True)
    }
    # Train models
    # train_models(models_list, vectorize='Hash')
    train_models(models_list, vectorize='BERT')
