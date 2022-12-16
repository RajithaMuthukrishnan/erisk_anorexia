import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from bs4 import BeautifulSoup 
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow import keras
import re
import joblib 
import argparse

from gensim.parsing.preprocessing import remove_stopwords

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils.class_weight import compute_class_weight

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


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import data_preparation_offline as data_preparation
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from evaluate.aggregate_results import aggregate_chunk_results
from evaluate.erisk_eval import calculate_erde

def extract_test_filenames(path):
    files = [] 
    for filename in os.listdir(path):
        if not filename.endswith('.xml'):
            continue
        filepath = os.path.join(path, filename)
        files.append(filepath)
    return files

def extract_test_chunks(test_data_path):
    dataframe_collection = {} 
    for ctr in range(1,11):
        file_path = str(test_data_path)+"/chunk"+str(ctr)
        files = extract_test_filenames(file_path)
        data_list = []
        for file in files:
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
                row = [subject_id.get_text(), title, text]
            data_list.append(row)
        chunk_name = 'chunk'+str(ctr)
        dataframe_collection[chunk_name] = pd.DataFrame(data_list, columns = ['subject_id', 'title', 'text'])
    return dataframe_collection

def test_chunks_predict(model_name, model, vectorizer, vec_type, threshold):
    # Variables for no decisions subjects and their data(aggregated from previous chunks)
    nodec_subjects = np.empty(0)
    nodec_aggregated = pd.DataFrame()
    print('Predicting ...')
    
    for chunk in range(1,11):
            
        chunk_name = 'chunk'+str(chunk)
        chunk_df = test_chunk_collection[chunk_name]

        # Let the delay in prediction be handled in the eval_erisk file
        if 'SVM' in model_name:

            clean_df = data_preparation.preprocess_data(chunk_df)
            X_test = data_preparation.vectorize_data(clean_df, vectorizer, vec_type=vec_type, isTrainMode=False)
    
            if 'sklearn' in str(type(model)):
                chunk_pred = model.predict(X_test)
            elif 'river' in str(type(model)):
                 chunk_pred = model.predict_proba_many(pd.DataFrame(X_test.toarray()))

            # Save prediction
            pred_df = pd.DataFrame(chunk_pred, columns=['pred'])
            pred_df.pred = pred_df.pred.astype('int')

            # save predictions to dataframe
            chunks_pred_df = pd.DataFrame(clean_df['subject_id'])
            chunks_pred_df['pred'] = pred_df['pred'].values

        # Handle the delay in prediction based on confidence score
        else:
            # if chunk1
            if chunk==1:
                # preprocess, vectorize and predict
                clean_df = data_preparation.preprocess_data(chunk_df)
                X_test = data_preparation.vectorize_data(clean_df, vectorizer, vec_type=vec_type, isTrainMode=False)
                
                if 'sklearn' in str(type(model)):
                    chunk_prob_classes = model.predict_proba(X_test)
                elif 'river' in str(type(model)):
                    chunk_prob_classes = model.predict_proba_many(pd.DataFrame(X_test.toarray()))
                
                # Save prediction and probability score
                prob_pred_df = pd.DataFrame(np.concatenate((np.max(chunk_prob_classes, axis=1).reshape(-1,1), np.argmax(chunk_prob_classes, axis=1).reshape(-1,1)), axis=1), columns=['prob', 'pred'])
                prob_pred_df.pred = prob_pred_df.pred.astype('int')
                # If confidence score less than threshold - no decision
                prob_pred_df.loc[prob_pred_df.prob < threshold, 'pred'] = 2
                
                # save predictions to dataframe
                chunks_pred_df = pd.DataFrame(clean_df['subject_id'])
                chunks_pred_df['pred'] = prob_pred_df['pred'].values
                # find the subject ids who have no decision (2)
                nodec_subjects = chunks_pred_df[chunks_pred_df.pred.isin([2])]['subject_id'].values
                # filter data only for the subjects with no decision (2)
                nodec_aggregated = chunk_df[chunk_df.subject_id.isin(nodec_subjects)]

            # if not chunk1 and no decisions array is not empty
            if (chunk!=1 and nodec_subjects.size!=0):            
                # concatenate previous chunk no decision subjects text with current chunk text
                for sub_id in nodec_subjects:
                    nodec_aggregated_text = nodec_aggregated.loc[nodec_aggregated['subject_id'] == sub_id]['text'].values[0]
                    if chunk_df.loc[chunk_df['subject_id'] == sub_id]['text'].values.size != 0:
                        current_chunk_text = chunk_df.loc[chunk_df['subject_id'] == sub_id]['text'].values[0]
                    else:
                        current_chunk_text = ''
                    concat_chunk_text = nodec_aggregated_text + ' ' + current_chunk_text
                    nodec_aggregated['text'].mask(nodec_aggregated['subject_id'] == sub_id , concat_chunk_text, inplace=True)
                
                # preprocess, vectorize and predict updated text for no decisions
                clean_df = data_preparation.preprocess_data(nodec_aggregated)

                X_test = data_preparation.vectorize_data(clean_df, vectorizer, vec_type=vec_type, isTrainMode=False)
                
                if 'sklearn' in str(type(model)):
                    nodec_prob_classes = model.predict_proba(X_test)
                elif 'river' in str(type(model)):
                    nodec_prob_classes = model.predict_proba_many(pd.DataFrame(X_test.toarray()))
                
                if nodec_subjects.size!=0:
                    if chunk==10:
                        nodec_prob_pred_df = pd.DataFrame(np.concatenate((np.max(nodec_prob_classes, axis=1).reshape(-1,1), np.argmax(nodec_prob_classes, axis=1).reshape(-1,1)), axis=1), columns=['prob', 'pred'])
                        nodec_prob_pred_df.pred = nodec_prob_pred_df.pred.astype('int')
                    else:
                        nodec_prob_pred_df = pd.DataFrame(np.concatenate((np.max(nodec_prob_classes, axis=1).reshape(-1,1), np.argmax(nodec_prob_classes, axis=1).reshape(-1,1)), axis=1), columns=['prob', 'pred'])
                        nodec_prob_pred_df.pred = nodec_prob_pred_df.pred.astype('int')
                        nodec_prob_pred_df.loc[nodec_prob_pred_df.prob < threshold, 'pred'] = 2
                
                # create a dataframe the predictions
                nodec_pred_df = pd.DataFrame(clean_df['subject_id'])
                nodec_pred_df['pred'] = nodec_prob_pred_df['pred'].values
                
                # update predictions 
                for sub_id in nodec_subjects:
                    updated_pred = nodec_pred_df.loc[nodec_pred_df['subject_id'] == sub_id]['pred'].values
                    chunks_pred_df['pred'].mask(chunks_pred_df['subject_id'] == sub_id , updated_pred, inplace=True)

                # update subject ids with no decision (2) 
                nodec_subjects = nodec_pred_df[nodec_pred_df.pred.isin([2])]['subject_id'].values
                # update list of subjects in no decision chunk
                nodec_aggregated = nodec_aggregated[nodec_aggregated.subject_id.isin(nodec_subjects)]
        
        # save predictions to file                
        file_name = 'test_predictions/usc_'+str(chunk)+'.txt'
        chunks_pred_df.to_csv(file_name, header=None, index=None, sep='\t', mode='w')    
    return chunks_pred_df


def extract_models(path):
    model_files = []
    vectorizer_files = []
    for filename in os.listdir(path):
        if '_vectorizer' in str(filename):
            vectorizer_filepath = os.path.join(path, filename)
            vectorizer_files.append(vectorizer_filepath)
        elif '.DS_Store' in str(filename):
            continue
        else:
            model_filepath = os.path.join(path, filename)
            model_files.append(model_filepath)
    return model_files, vectorizer_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-data', help='path to test data chunks', required=True, nargs=1, dest="test_data_path")
    args = parser.parse_args()

    test_data_path = args.test_data_path[0]

    # Extract Test chunks
    test_chunk_collection = extract_test_chunks(test_data_path)
    test_truth = pd.read_csv('../evaluate/risk-golden-truth-test.csv')
    
    path = "trained_models"
    model_files, vectorizer_files = extract_models(path)
    
    print('    *** TEST MODELS ***   ')

    for model_path in model_files:
        model_name = (model_path.split('/')[-1]).split('.')[0]
        for vectorizer_path in vectorizer_files:
            vectorizer_name = (vectorizer_path.split('/')[-1]).split('.')[0]
            if model_name+str('_vectorizer') == vectorizer_name:

                if '.h5' in model_path:
                    clf = tf.keras.models.load_model(
                        (model_path),
                        custom_objects={'KerasLayer':hub.KerasLayer}
                        )
                elif '.joblib' in model_path:
                    clf = joblib.load(model_path)
                    
                if '.h5' in vectorizer_path:
                    vectorizer = tf.keras.models.load_model(
                        (vectorizer_path),
                        custom_objects={'KerasLayer':hub.KerasLayer}
                        )
                elif '.joblib' in vectorizer_path:
                    vectorizer = joblib.load(vectorizer_path)

                print('\n--- '+model_name+' ---')
                if 'BERT' in model_name:
                    test_pred_df = test_chunks_predict(model_name, clf, vectorizer, vec_type='BERT', threshold=0.7)
                else:
                    test_pred_df = test_chunks_predict(model_name, clf, vectorizer, vec_type='Hash', threshold=0.7)
                
                test_true_list = []
                for subject in test_pred_df['subject_id']:
                    value = test_truth.loc[test_truth['subject_id']==subject]['label'].values[0]
                    value_list = [subject, value]
                    test_true_list.append(value_list)
                final_test_df = pd.DataFrame(test_true_list, columns=['subject_id', 'label'])

                print(classification_report(final_test_df['label'], test_pred_df['pred']))

                # Calculate ERDE 
                aggregate_chunk_results(isOnline=True)
                calculate_erde(isOnline=True)