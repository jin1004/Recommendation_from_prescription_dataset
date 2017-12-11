
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import json

from pandas.io.json import json_normalize
from flatten_json import flatten
from sklearn import preprocessing

from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[41]:


#load the json file
filename='Simplified_Prescription_Corpus.json';
with open(filename, 'r', encoding="utf8") as f:
        data_json = json.load(f)

#normalized json is loaded (where each vital is in different column)
data=json_normalize(data_json)

#flatten json (so that each element is in a different column)
#data_flattened = (flatten(d) for d in data_json)
#load the flattened json to panda's dataframe
#data_normalized = pd.DataFrame(data_flattened)


# In[42]:


#Creates a separate row for each element in lists inside medicines 

meds = data.apply(lambda x: pd.Series(x['medicines']),axis=1).stack().reset_index(level=1, drop=True)
meds.name = 'medicines'

data2=data.drop('medicines', axis=1).join(meds)

data2[['diagnoses_0','diagnoses_1','diagnoses_2', 'diagnoses_3', 'diagnoses_4']] = pd.DataFrame(data2.diagnoses.values.tolist(), index= data2.index)
data3=data2.drop('diagnoses',axis=1)

#Converts null to empty array
data3.symptoms.loc[data3.symptoms.isnull()] = data3.symptoms.loc[data3.symptoms.isnull()].apply(lambda x: [])

data3[['symptoms_0','symptoms_1','symptoms_2', 'symptoms_3', 'symptoms_4', 'symptoms_5', 'symptoms_6', 'symptoms_7', 'symptoms_8', 'symptoms_9', 'symptoms_10']] = pd.DataFrame(data3.symptoms.values.tolist(), index= data3.index)
data_normalized=data3.drop('symptoms',axis=1)


# In[43]:


data4


# In[44]:


#separate the numeric and categorical data
data_numeric = data_normalized.select_dtypes(include=[np.number])

#Replace NaNs with 0s
data_numeric=data_numeric.fillna(0)

#Convert dataframe to numpy array for future processing
data_numeric=data_numeric.values

#data with text
data_categorical = data_normalized.select_dtypes(include=[object])

#Replace empty arrays with NULL objects
#data.diagnoses = data.diagnoses.apply(lambda y: np.nan if len(y)==0 else y)
#data.medicines = data.medicines.apply(lambda y: np.nan if len(y)==0 else y)

#Replace NaNs with unique character "NaN". Encoding cannot be done 
#if the missing values are not replaced with characters 
data_categorical=data_categorical.fillna("None")

#Extract the categorical datasets separately for processing
data_sex = data_categorical.filter(regex='sex')
data_symptoms = data_categorical.filter(regex='symptoms')
data_diagnoses = data_categorical.filter(regex='diagnoses')
data_medicines = data_categorical.filter(regex='medicines')
#Set 1st medicine column as target data and the rest as input features
#data_medicines_input = data_medicines.drop('medicines_0', axis=1)
#data_medicines_target = data_medicines.filter(regex='medicines_0')


#All the text data are concatenated except for medicines which is the target dataset
data_input = pd.concat([data_sex,data_symptoms,data_diagnoses],axis=1)

#All the categorical data are encoded using One Hot Encoder

# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()

#transform all columns with label encoding
data_input2 = data_input.apply(le.fit_transform)
data_medicines_target2 = data_medicines.apply(le.fit_transform)

#transform all columns with one hot encoding
enc = preprocessing.OneHotEncoder()
enc.fit(data_input2)
input_onehotlabels = enc.transform(data_input2).toarray()

enc.fit(data_medicines_target2)
data_target = enc.transform(data_medicines_target2).toarray()

#combine numeric array with onehotlabelArray
data_main=np.column_stack((data_numeric,input_onehotlabels))


# In[45]:


data_main.shape


# In[14]:


df=pd.get_dummies( d, columns = cols_to_transform )
df.iloc[48]


# In[46]:


#split the file into training and test data. 10% of the training data is used as test data
train_input, test_input, train_target, test_target = train_test_split(data_main, data_target, test_size=0.1, random_state=50)


# In[ ]:


# Spot Check Algorithms
#train with
model=RandomForestClassifier()
#Get the best parameters for the model training using 10 fold cross validation
kfold = model_selection.KFold(n_splits=10, random_state=10)
cv_results = model_selection.cross_val_score(model, train_input, train_target, cv=kfold, scoring='accuracy')
model.fit(train_input,train_target)
# save the model to disk
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[162]:


cv_results

data_numeric.shape
# In[203]:


data_categorical.shape


# In[204]:


data.shape

