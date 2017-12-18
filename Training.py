
# coding: utf-8
import numpy as np
import pandas as pd
import json
import pickle
import ast
import re

from pandas.io.json import json_normalize
from flatten_json import flatten
from sklearn import preprocessing
from sklearn.externals import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import StandardScaler

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression


#for debugging (remove from main code)
np.set_printoptions(threshold=np.inf)


#load the json file
#converted all empty arrays to null. find a way to fix it in program
filename='Simplified_Prescription_Corpus.json';
with open(filename, 'r', encoding="utf8") as f:
        data_json = json.load(f)

#normalized json is loaded (where each vital is in different column)
data_all=json_normalize(data_json)


# All the normalizing functions

def ohc_encode(data):
    # First, encode labels with value between 0 and n_classes-1.
    le = preprocessing.LabelEncoder()
    #transform all columns with label encoding
    data_le = data.apply(le.fit_transform)
    #transform all columns with one hot encoding
    enc = preprocessing.OneHotEncoder()
    enc.fit(data_le)
    data_ohc = enc.transform(data_le).toarray()
    return data_ohc


#tokenize function (used in CountVectorizer)
REGEX = re.compile(r",\s*")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]

#get unique vocab present from categorical dataframe
def get_vocab (data):
    #get col header
    col_header=data.columns.tolist()[0]   
    #create vocab list from all the values 
    vocab_all=data.apply(lambda x: pd.Series(x[col_header]),axis=1).stack().reset_index(level=1, drop=True)
    vocab_all = vocab_all.str.lower()
    #get unique vocab
    vocab=vocab_all.drop_duplicates(keep='first')
    return vocab


#Create vectorizer (for symptoms and diagnoses)
def create_vectorizer (data):
    #initialize vectorizer
    vocab = get_vocab(data)
    vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=tokenize)
    return vectorizer

#vectorize features (symptoms and diagnoses)
def vectorize_feature (data, vectorizer):
    #get column header
    col_header=data.columns.tolist()[0]    
    #find number of rows
    num_rows = data.shape[0]
    #process the arrays so that it can be used in countvectorizer
    temp=[]
    for i in range(0, num_rows):
        temp_str = data[col_header][i]
        temp_str = ','.join(temp_str).lower()
        temp.append(temp_str)
    final=vectorizer.transform(temp).toarray()
    return final


# In[10]:

#normalize_input_features
def normalize_input_train(data):
    #separate the numeric and categorical data
    data_numeric = data.select_dtypes(include=[np.number])

    #Replace NaNs with 0s
    data_numeric=data_numeric.fillna(0)

    #Convert dataframe to numpy array for future processing
    data_numeric=data_numeric.values

    #data with text
    data_categorical = data.select_dtypes(include=[object])
    
    #give null values to empty arrays 
    #because we cant remove them since number of features has to always be the same
    data = data_categorical.apply(lambda y: np.nan if len(y)==0 else y)
    
    #Replace NaNs with any string not present in datasets. Used "None" here. 
    #Encoding cannot be done if the missing values are not replaced with characters 
    data_categorical=data_categorical.fillna("None")
    
    #Since the categorical columns are of different types,
    #process them separately
    data_sex = data_categorical.filter(regex='sex')
    data_symptoms = data_categorical.filter(regex='symptoms')
    data_diagnoses = data_categorical.filter(regex='diagnoses')

    vectorizer_symptoms = create_vectorizer(data_symptoms)
    vectorizer_diagnoses = create_vectorizer(data_diagnoses)
    
    #process data differently depending on the column names
    #find better way to separate the processing (more generalized)
    #one hot encode sex column
    data_sex_final = ohc_encode(data_sex)
      
    #vectorize symptoms and diagnoses
    data_symptoms_final = vectorize_feature(data_symptoms, vectorizer_symptoms)
    data_diagnoses_final = vectorize_feature(data_diagnoses, vectorizer_diagnoses)
    #combine numeric array with onehotlabelArray
    data_out=np.column_stack((data_numeric, data_sex_final, data_symptoms_final, data_diagnoses_final))

    #scale the features
    scaling = StandardScaler()
    data_normalized = scaling.fit_transform(data_out) 
    return data_normalized


# In[11]:

#normalize target labels
def normalize_target(data):
    #get col header
    #give null values to empty arrays 
    #because we cant remove them since number of features has to always be the same
    data = data.apply(lambda y: np.nan if len(y)==0 else y)
    #Replace NaNs with any string not present in datasets. Used "None" here. 
    #Encoding cannot be done if the missing values are not replaced with characters 
    data=data.fillna("None")
    #get unique vocabulary
    vocab_uniq = get_vocab(data)
    vocab = vocab_uniq.tolist()
    # First, encode labels with value between 0 and n_classes-1.
    le = preprocessing.LabelEncoder()
    #transform all columns with label encoding
    le.fit(vocab)

    num_rows = data.shape[0]
    labels = []
    sizes = []
    #process the arrays so that it can be used in labelencoder
    for i in range(0, num_rows):
        temp_list = data['medicines'][i]
        temp_list2 = [item.lower() for item in temp_list]
        temp = le.transform(temp_list2)
        sizes.append(temp.shape[0])
        labels.append(temp)
    max_size = max(sizes)
    final = np.zeros((num_rows, max_size))
    #rearrange the list into numpy array since models cannot accept lists
    #find better ways to convert / use multivariable labelling for modelling
    for x in range(0, num_rows):
        size_list = labels[x].shape[0]
        for y in range(0, size_list):
            final[x,y] = labels[x][y]
    return final



#Separate the dataset into input features and target labels
data_input = data_all[data_all.columns.drop(list(data_all.filter(regex='medicines')))]
data_target = data_all.filter(regex='medicines')


#normalize features and labels in the format that a sklearn model can accept
data_input_processed = normalize_input_train(data_input)

data_target_normalized = normalize_target(data_target)


#split the file into training and test data. 10% of the training data is used as test data
train_input, test_input, train_target, test_target = train_test_split(data_input_normalized, data_target_normalized, test_size=0.1, random_state=50)

#model using LogisticRegression
#Binary Relevance is used to transform the multilabel problem into a single label problem
#One versus all is used since its multi class
model = BinaryRelevance(LogisticRegression(multi_class='ovr'))
model.fit(train_input, train_target)
joblib.dump(model, 'model_v1.out')