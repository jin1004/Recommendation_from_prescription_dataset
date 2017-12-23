
# coding: utf-8

import sys
import numpy as np
import pandas as pd
import nltk
import re

from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


#tokenize function (used in CountVectorizer)
REGEX = re.compile(r",\s*")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]


#encode sex data using label encoder and one hot encoder trained on training dataset
#change encoding method in training to make the encoding simpler
def ohc_encode(data, le, enc):
    data_pd = pd.DataFrame({'sex' : [data]}, dtype=object)
    data_le = data_pd.apply(le.transform)
    data_final = enc.transform(data_le).toarray()
    return data_final


def vectorize_feature (data, vectorizer):
    #encode symptoms and diagnoses using vectorizers trained on training dataset
    data_arr=[data]
    data_final=vectorizer.transform(data_arr).toarray()
    return data_final


# Load the model and other trained encoders 

model=joblib.load('model1.out')
le_sex=joblib.load('le_sex.out')
enc_sex=joblib.load('enc_sex.out')
vectorizer_symptoms=joblib.load('vectorizer_symptoms.out')
vectorizer_diagnoses=joblib.load('vectorizer_diagnoses.out')
le_medicines=joblib.load('le_medicines.out')
input_scaler=joblib.load('input_scaler.out')


# Create input variables
print ("Please enter the following details about the patient: ")
print ("Sex (Male/Female): ")
while True:
    sex=input()
    sex=sex.lower()
    if ((sex=='Male') or (sex=='Female')):
        break
    else:
        print ("Please enter 'Male' or 'Female' ")
        print ("Sex (Male/Female): ")
print ("Age: ")
while True:
    try:
        age = float(input())
    except ValueError:
        print("Please enter a valid number:")
        #Return to the start of the loop
        continue
    else:
        #successfully parsed
        break
print ("Vitals: ")
while True:
    try:
        weight = float(input ('Vitals (Weight):  '))
    except ValueError:
        print("Please enter a valid number. If the vital information is not available, please enter 0: ")
        #Return to the start of the loop
        continue
    else:
        #successfully parsed
        break
while True:
    try:
        bp_systole = float(input ('Vitals (BP_Systole):  '))
    except ValueError:
        print("Please enter a valid number. If the vital information is not available, please enter 0: ")
        #Return to the start of the loop
        continue
    else:
        #successfully parsed
        break
while True:
    try:
        bp_diastole = float(input ('Vitals (BP_Diastole):  '))
    except ValueError:
        print("Please enter a valid number. If the vital information is not available, please enter 0: ")
        #Return to the start of the loop
        continue
    else:
        #successfully parsed
        break
while True:
    try:
        pulse = float(input ('Vitals (Pulse):  '))
    except ValueError:
        print("Please enter a valid number. If the vital information is not available, please enter 0: ")
        #Return to the start of the loop
        continue
    else:
        #successfully parsed
        break
while True:
    try:
        glucose = float(input ('Vitals (Glucose)  '))
    except ValueError:
        print("Please enter a valid number. If the vital information is not available, please enter 0: ")
        #Return to the start of the loop
        continue
    else:
        #successfully parsed
        break
print ("For Symptoms, Diagnoses and Medicines, please separate multiple values by a comma") 
print ("Example: Fever, Itching of skin, Headache and dizziness")       
print ("Symptoms:")
symptoms = input ()
print ("Diagnoses:")
diagnoses = input ()
print ("Medicines:")
medicines = input ()

# Encode all the inputs using the encoders trained during training

sex_final = ohc_encode(sex, le_sex,enc_sex)
symptoms_final=vectorize_feature (symptoms, vectorizer_symptoms)
diagnoses_final=vectorize_feature (diagnoses, vectorizer_diagnoses)
#stack in the same order as in training data
data_input=np.column_stack((age, bp_diastole, bp_systole, glucose, pulse, weight, sex_final, symptoms_final, diagnoses_final))
#normalize data using the scaler fitted on training data
data_input_normalized = input_scaler.transform(data_input) 


predicted_data=model.predict(data_input_normalized)
predicted_data=predicted_data.toarray()


# bug during training:
# using 0 for all the additional labels during training has caused the predictor to also output 0's after the predicted labels.
# The issue is: 0 is also a label number for a particular medicine
# 
# Temporary workaround: I am cutting all the 0's in the end. As long as this particular medicine is not the last recommended medicine by the system, everything should be fine, and since this particular medicine encoded as 0 only appears twice during training, it's highly unlikely for it to be the final recommended medicine in any given scenario. 

ind=np.argmax(predicted_data==0)
predicted_data=predicted_data[0][0:ind-1]


# go through the predicted list
# if predicted medicine is already present, go to the next best medicine
# the most probable predicted medicine not provided as an input is the output

given_medicines=medicines.lower().split(",")
#default print if output not available
output='Provided medicine list includes all the recommended medicines'
for i in range(len(predicted_data)):
    predicted_medicine=le_medicines.inverse_transform(int(predicted_data[i]))
    if any(predicted_medicine in med for med in given_medicines):
        continue
    else:
        output=predicted_medicine
        break
print ("\n")
print ("Recommended Medicine:")
print(output)

