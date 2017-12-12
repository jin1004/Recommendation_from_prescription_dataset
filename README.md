## Project Description
This repository contains the codes and description of the medicine recommendation system (Jeeon).

## Recommendation System
I decided to create a classifier using the dataset with the medicines as the output label and all the other information as input features.

I am currently using a RandomForestClassifier for the classification. The hyperparameters are chosen using 10-fold cross validation.

The classifier model is then used to predict the medicine labels with their probabilities given the input. The medicine label with the highest probability that is not provided as one of the inputs is chosen to be the recommended medicine.

## Things to do

Try a different classification model.

Choose a feature set using a feature selection method such as RFE before training the model so that the model is trained using only the most relevant features.

Think of other possble approaches


