# Early Detection of Anorexia based on Social Media Comments using Incremental Learning

Social media comments have the potential to expose various user behaviors. Anorexia is one such condition that can be noticed from the content posted by the affected individuals. This project concentrates on leveraging machine and deep learning models to automatically detect the signs of anorexia based on social media comments, focusing on the behavior of the users. 

The [CLEF erisk 2019](https://erisk.irlab.org/2019/index.html) challenge focuses on three tasks, one among which is the Early Detection of Signs of Anorexia. The Reddit posts are in chunks for this task to mimic the real-world difference in posting dates and times.

Analyzing social media comments to comprehend user behaviors is an ongoing process, and continuous monitoring of user comments is crucial. Online Training is a Machine Learning technique that enables updating models with new data.

>> [1] Losada, David &amp; Crestani, Fabio &amp; Parapar, Javier. (2019). Overview of eRisk 2019 Early Risk Prediction on the Internet. 10.1007/978-3-030-28577-7_27.

This implementation includes training the following models
- SGDClassifier with modified huber loss
- Logistic Regression
- Support Vector Classifier
- BERT Classifier

with feature extraction techniques
- Hashing Vectorizer
- BERT 

## RESULTS
The results include a classification report with Precision, Recall, F1 score for class 0 'Anorexia', class 1 'Non-Anorexia', and weighted average for every model.
The Early Risk Detection Error (ERDE) rate is also included for parameters o=5 and o=50, where o represents the number of writings processed from where the ERDE increases rapidly [1]

The results of the trained *offline* models with Hashing Vectorizer and BERT Feature Extractor 
 ```sh
cd offline/results
```
The results of the trained *online* models with HashingVectotizer and BERT Feature Extractor 
 ```sh
cd online/results
```

## DATA
The dataset for the [CLEF erisk 2019](https://erisk.irlab.org/2019/index.html) challenge is used for this implementation.

## EVALUATION

Note: Online and Offline models with Hashing Vectorizer are available for evaluation. BERT classifiers and models based on BERT feature extraction must be built first and then evaluated. Please refer to the Training section to build BERT-based models.

### Offline Models (Single Batch Training)
```sh
cd offline
python test_offline_model.py -data <path_to_test_data_chunks>
```
### Online Models (Incremental Training)
```sh
cd online
python test_online_model.py -data <path_to_test_data_chunks>
```

## TRAINING
Requirement: Train and Test data chunks
### Offline Models (Single Batch Training)
```sh
cd offline
python train_offline_model.py -data <path_to_train_data_chunks>
```
### Online Models (Incremental Training)
```sh
cd online
python train_online_model.py -data <path_to_train_data_chunks>
```
