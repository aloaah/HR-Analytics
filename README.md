


---
<p><img alt="https://www.ceratizit.com/" height="120px" src="https://consent.trustarc.com/v2/asset/09:13:49.986pc83s3_CERATIZIT_G_WEB_color_transparent.png" align="left" hspace="10px" vspace="0px"></p>  <h1>HR-Analytics</h1>
Data science open position challenge in order to lay the foundation for a technical discussion during the interview with the condidate 



---

## Table of content

* [0. Modules and versions](#modules)
* [1. About the context of this study](#content)
* [2. Getting started: basic steps](#Steps)
  * [a. Preprocessing](#preprocessing)
  * [b. Modeling](#modeling)
  * [c. Scoring](#scoring)
* [3. Models benchmarking](#benchmarking)
  * [a. Imbalanced Data](#imbalanced)
  * [b. SMOTE Data](#smote)
* [4. Best Mode selection](#bestmodel)
  * [a. Fine tuning](#tuning)
  * [b. Scores](#scores)
* [5. Submission](#submission)
	
## 0. Modules and versions <a name="modules">
	
* pandas==1.1.5
* numpy==1.19.5
* matplotlib==3.2.2
* seaborn==0.11.1
* category-encoders==2.2.2
* regex==2019.12.20
* xgboost==1.4.2
* scikit-learn==0.22.2.post1
* imbalanced-learn==0.4.3

## 1. About the context of this study <a name="content">

### a. **Business or activity:**</br>
A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which conduct by the company. Many people signup for their training. Company wants to know which of these candidates are really wants to work for the company after training or looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and `categorization` of candidates. `Information related to demographics, education, experience are in hands from candidates signup and enrollment`.

The given dataset was designed to understand the factors that lead a person to leave their current job for HR research as well. Through a model(s) that uses current data on `qualifications`, `demographics`, and `experience`, we will be able to `predict the likelihood of a candidate seeking a new job or working for the company`, as well as `interpreting the factors that influence the employee's decision.`

The whole data divided to train and test . Target isn't included in test but the test target values data file is in hands for related tasks. A sample submission correspond to enrollee_id of test set provided too with columns : `enrollee _id , target`(submission format).

**Note:**
`The dataset is imbalanced. Most features are categorical (Nominal, Ordinal, Binary), some with high cardinality. Missing imputation can be a part of your pipeline as well.`

---
* Since we have labeled data, we are typically in the context of supervised learning.

* From the brief description of the problem, we can notice that we are facing a classification task and that the data are already collected. Therefore, there is no need to collect data unless we want to enrich the data to get more relevant information. 

* We are therefore looking for a classifier that produces probabilities of belonging to one of two classes that predict the probability that an applicant will seek a new job or work for the company, a classifier that predicts not the classes to which the examples belong, but the probability that the examples fit a particular class as well as the interpretation of the factors influencing the employee's decision.

* We can either use any binary classifier to learn a fixed set of classification probabilities (e.g., p in [0.0, 1.0] ) or conditional probability models which can be generative models such as naive Bayes classifiers or discriminative models such as logistic regression (LR), neural networks, if we use cross-entropy as the cost function with sigmoidal output units. This will provide us with the estimates we are looking for.




## 2. Getting started<a name="Steps">
 The basic steps that I will follow to address this problem and ensure to achieve business value and minimize the risk of error are as follows: 

1.   Understanding the Business
2.   Loading the data
3.   Preprocessing and exploratory analysis
4.   Predictive modeling
5.   Interpretation of results

Ps: we could iterate between the steps depending on our objectives and the result of each step..

### a. Preprocessing  <a name="preprocessing">
  
### b. Modeling  <a name="modeling">

### c. Loss and scoring  <a name="scoring">

## 3. Models benchmarking<a name="benchmarking">
  
### a. Imbalanced Data  <a name="imbalanced">

### b. SMOTE Data  <a name="smote">
  
## 4.  Best Mode selection <a name="bestmodel">

### a. Fine tuning  <a name="tuning">
  
### b. Scores  <a name="scores">
  
## 5. Submission  <a name="submission">

Thanks to our Image Recognition teacher, Pedro Octaviano, who spent inconsiderate amount of his free time to teach and help us produce the best we could.


 * Made by: Wael GRIBAA, Amine OMRI, Sabina REXHA, Abdou Akim GOUMBALA and Meryem GASSAB
 * Date: 25th May 2021
