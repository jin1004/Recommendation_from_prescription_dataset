## Project Overview
This repository contains the codes and description of the medicine recommendation system (Jeeon).

## Objective
The goal is to create a recommendation applicaiton that, given an unfinished prescription (that has some data already), can suggest the next best medicine to prescribe.


## How to run

* Install Docker (https://docs.docker.com/engine/installation)

I have uploaded the image to docker hub. So in order to run it, just follow these steps:
* Pull the image from Docker Hub (image size is ~1.34GB)  It may take ~10-15 minutes:
`docker pull naziba/medicine_recommendation`
* Run the application and follow the instructions on screen:
`docker run -ti naziba/medicine_recommendation`

 Alternatively, you can also build your image:
 * First clone the git repository to a local directory
 * Download the model from https://www.dropbox.com/s/xe2994bu9g96b7d/model1.out?dl=0 and put it in the same directory
 * cd to the local directory
 * Run the following commands:
 `docker build -t medicine-recommendation .`
 `docker run -ti medicine-recommendation`

#Algorithm Description
A dataset is provided for the task, which I am using to train a supervised machine learning classifier that uses the different medicines as different classes and predicts which class a particular data will belong to given all the other values in the prescription.

## Dataset Description
A snapshot of the JSON dataset is given below:
![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/Initial_json.png)

The JSON file is loaded as a Pandas dataframe object.
![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/Initial_dataframe.png)

As can be seen from the  figure above, the dataset consists of 3966 samples with 10 categories.

The dataset is divided into 2 parts: data_input (all the categories except the medicines) and data_target (medicines column).
![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/input_target.jpg)

## Data Cleaning

There are 2 primary data types in the dataset: numeric and categorical data (data with text). Screenshots of numeric and categorical datasets from the input dataset is given below:
![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_numeric.png)
![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_categorical.png)

The numeric dataset is cleaned by replacing *null* object with 0.

In the categorical dataset including the target dataset (medicines), the missing values seem to be represented by empty arrays, *null* objects and *none* string. In order to clean the dataset, I replaced the empty objects with *null* objects and then replaced the *null* objects by *None* to make the dataset uniform. We haven't used other approaches such as replacing with the mean or dropping rows/columns with empty objects mainly because the dataset is very small. The cases of all the text in all the columns are converted to lowercase as well, for uniformity.

> **Potential Issues**
> It's not a very good idea to replace missing values with 0 or null, but I need to do more analysis on the relationship between each of these different features and their relationships with the medicine and figure out the best approach.

## Data Processing

Majority of machine learning algorithms can only work with numeric data, so I wanted to find the best way to represent the entire dataset numerically.

There are 3 categorical features inside the input dataset that needs to be processed (sex, symptoms and diagnoses). 

The **sex** column only consists of 2 values: *male* and *female*. So I just one-hot encoded the data, which is a common way to represent categorical dataset for machine learning algorithms. It basically converts each of the datasets into a binary representation. So, in this case, for example, we represent *male* as *01* and *female* as *10*. The reason for not using numeric labels such as *1* and *2* directly to represent the dataset is because the magnitude of the numeric value of the features matters when it come to the machine learning algorithms. So, the overall prediction accuracy will be very erroneous if it assumes *2* > *1* just because they were the random labels given to different text values in the data.

The **symptoms** and **diagnoses** are represented slightly differently. As shown in previous figures, both of these features are represented by an array of phrases. For both of them, I created a vocabulary separately using all the unique phrases used to describe **symptoms** and **diagnoses**. Then I just convert each of the text arrays to a sparse vector form, which consists of counts of each of the **symptoms** or **diagnoses** present in the array. 

So for example, this is a screenshot of the initial **symptoms** column:

![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_symptoms.png)

The third row in vector form is as follows, where it can be seen that there is 1 for each of the 4 symptoms present in the array, and the rest of values are all 0 denoting that no other symptom is present:

![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_symptoms_final.png)

Some screenshots for the **diagnoses** column is also given below:

![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_diagnoses.png)
![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_diagnoses_final.png)

> **Potential Issues**
> In this case, we are assuming certain symptoms and diagnoses are always represented using the same phrases, which is not the case in reality. Ideally a word2vec model that can put all the similarly worded symptoms or diagnoses in the same group will work better. However, I will need a bigger dataset and machine with more resources in order to model something like that. There may be other approaches that I have not thought of yet to represent this kind of data, but I need to do more research.

After procesing all the text data, all the arrays obtained are concatenated:
![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_input_processed.png)

The input data is also scaled because, as mentioned earlier the magnitude of feature set impacts a machine learning model. So features with hugely varied numeric values will be of more importance to the model. So we need to scale them before feeding to the classifier.
![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_input_normalized.png)


The target dataset, **medicines** is label encoded, where we first create a numeric label for each unique medicine in the dataset. Then encode the **medicine** arrays using these labels. Since, it is the target dataset, the magnitude of the label does not matter so we don't necessarily need to use binary representation to represent the dataset. Screenshots of the **medicine dataset** is given below: 

![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_target.png)

![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/data_target.png)

## Learning Algorithm

It is often hard to tell right away which algorithm will work best with a given dataset. So generally people often use different algorithms on the subset of the data to see which works best. Due to time and resource constraints, I decided to just try using **Logistic Regression**, which is a classification algorithm using linear discriminant that outputs the probability that a point belongs to a certain class given the features. We mainly decided to use it because:
1) We need a probability of all the labels instead of getting a few labels as an output because certain labels may already be given while testing so we need to output the medicine with the highest probability that is not already provided.
2) It seemed to take less memory and time. A few other classifiers that I tried earlier was taking too long or making my laptop unusable because it was using up too much memory.

Also, logistic regression, like most of the other ML algorithms, only works with single labels. So, in order to use a logistic regression model, we first need to turn it into a single label classification problem. So I tried two different approaches in order to transform the multilabel classification problem. One is the **Binary Relevance** approach, which trains a separate classifier for each label in a sample. This approach assumes each label present in a sample is independent of each other, which may not be ideal for this case, because some medicines are always more likely to be prescribed together than the others. So I tried the **Label Powerset** approach as well, which train a separate classifier for every label combination found in the training set.

 > Limitations and Possible Improvements
 >
 >
 > Firstly, I am training in my laptop so I can't do anything that needs to run for more than a few hours or takes up too much resources. 
 >
 > So I haven't been able to use cross validation, which is a method often used to find the ideal parameters for a model by dividing the training set into training and validation sets and finding the model that gives the best overall accuracy on the different validation sets. This also reduces the risk of overfitting.
 >
 > Also, the features used should have ideally been transformed using something like PCA  (Principal Component Analysis), which basically reduces the dimension of the feature set using the dependencies between the feature, which I think would have been ideal in this case. But I did not get the time to do that.
 
## Prediction Output

The model can predict the most likely set of labels given an unfinished prescription. Since the input prescription may contain a few medicines as well, the final output is one of the predicted medicines that was not provided as an input.

> Small bug that should be fixed when it's trained next time:
>
> Since number of medicines provided is always different in the dataset, I paddeed the additional trailing fields with 0s. I realized while testing that one of the medicine labels is also encoded as 0.
> 
> Temporary Workaround:
>
> I am just not taking into account the trailing 0's during recommendation. This will not be a problem as long as the particular medicine encoded as 0 is not the last recommended medicine by the algorithm. The chances of that are very low with this particular model and dataset because the medicine only appears twice in the training dataset so it is not very likely to be recommended in the first place and it is way more unlikely to be last label in the output.

## Evaluation
I evaluated the model on a small test set of about 300 prescriptions that I separated before starting the training. I tested using precision, recall and F-score, which seem like the most appropriate measures for this case. Precision score measures the percentage of correct prescribed medicines among the medicine labels selected by the algorithm. Recall score measures the percentage of correct medicine labels selected by the algorithm. F-measure is a combined precision and recall measure. The scores are shown below:

Evaluation Scores from using **Binary Relevance** to transform the multilabel classifier:

![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/eval_model1.png)

Evaluation Scores from using **Label Powerset** to transform the multilabel classifier:

![Alt text](https://github.com/jin1004/Recommendation_project/blob/master/extras/images/eval_model2.png)

Obiously the scores overall are still pretty low. Implementing the other approaches discussed earlier probably will increase the accuracy by some margin. However, it's probably not going to improve too much without a much larger dataset.
