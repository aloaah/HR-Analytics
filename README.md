


---
<p><img alt="https://www.ceratizit.com/" height="120px" src="https://consent.trustarc.com/v2/asset/09:13:49.986pc83s3_CERATIZIT_G_WEB_color_transparent.png" align="left" hspace="10px" vspace="0px"></p>  <h1>HR-Analytics</h1>
Data science open position challenge in order to lay the foundation for a technical discussion during the interview with the condidate 



---

## Table of content

* [0. Modules and versions](#modules)
* [1. About the context of this study](#content)
* [2. About the HPA-SCC competition](#competition)
  * [a. Context](#context)
  * [b. Links](#links)
  * [c. Scoring](#scoring)
* [3. Models used](#models)
  * [a. HPA Cell Segmentator](#segmentator)
  * [b. Organelle Classifier](#org_class)
* [4. API](#api)
  * [a. Installation](#install)
  * [b. How to use](#howto)
* [5. Thanks](#thanks)
	
## 0. Modules and versions <a name="modules">
	
* click 7.0
* Flask 2.0.1
* imageio 2.9.0
* importlib-metadata 3.10.0
* numpy 1.19.5
* pillow 8.2.0
* pydantic 1.8.2
* opencv-python 4.5.1.48
* scikit-image 0.16.2
* scipy 1.4.1
* tensorflow 2.4.1
* torch 1.4.0
* torchvision 0.5.0

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




## 2. Getting started
 The basic steps that I will follow to address this problem and ensure to achieve business value and minimize the risk of error are as follows: 

1.   Understanding the Business
2.   Loading the data
3.   Preprocessing and exploratory analysis
4.   Predictive modeling
5.   Interpretation of results

Ps: we could iterate between the steps depending on our objectives and the result of each step..

### a. Context  <a name="context">
  
![Kaggle HPA - Single Cell Classification](/img/hpa_kaggle.PNG)

The goal is to design an algorithm able to detect the organelle highlighted by the green channel of an image.
  
Each image is composed of four channels. The naming scheme of those channels is arbitrary as they are not to be associated with true colors : they just represent different microscopical observation of the same view.
Red :  microtubule
Blue : nuclei
Yellow : endoplasmic reticulum (ER) channels
Green for protein of interest of which we try to predict the class.

Those organelle are our outputs and are classified between 18 classes :
- Nucleoplasm
- Nuclear membrane
- Nucleoli
- Nucleoli fibrillar center
- Nuclear speckles
- Nuclear bodies
- Endoplasmic reticulum
- Golgi apparatus
- Intermediate filaments
- Actin filaments
- Microtubules
- Mitotic spindle
- Centrosome
- Plasma membrane
- Mitochondria
- Aggresome
- Vesicles and punctate cytosolic patterns
... and one Negative class.


### b. Links  <a name="links">

Human Protein Atlas - Single Cell Classification competition : https://www.kaggle.com/c/hpa-single-cell-image-classification

### c. Loss and scoring  <a name="scoring">

The score used for this project was the categorical cross-entropy loss. It's a metric that needs to be minimized.

Here is its formula, for C classes, S samples and p meaning "positive" :

![Categorical cross entropy](/img/categorical_cross_entropy.jpg)
	
The evaluation metrics is the classic categorical accuracy.

## 3. Models used <a name="models">
  
### a. HPA Cell Segmentator  <a name="segmentator">

A Deep learning model based on the infamous "U-NET" but greatly improved and enriched by the original team.
  
![Original U-Net](/img/unet.png)
  
Its purpose is to locate and instanciate each individual cell in an image composed of the three aformentioned channels (except the "green").
  
You can find more information about this model here : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
The GitHub of the HPA_Cell_Segmentator is accessible here : https://github.com/CellProfiling/HPA-Cell-Segmentation
  
### b. Organelle Classifier  <a name="org_class">
  
A custom Tensorflow model currently in development.
  
Please read the R&D notebooks "Building and tuning the model" to get some details about its architecture, available in this repertory /https://github.com/WGribaa/organelle_classifier/tree/main/R%26D%20Notebooks%20and%20presentation .
	
Please read the pdf file "Cell_level_segmentation - presentation (FR).pd" (available in the same repertory) to read about our intuitions and what we achieved sor far.
  
## 4. API <a name="api">

### a. Installation  <a name="install">
  
Easy to use, you can simple launch "app.py" and let the script install the cnessary HPA_Cell_Segmentator from GitHub. Just ensure all the modules in requirements.txt are installed in your environment.
Once launched, you should acces the API Flask interface in any Web Browser at the address: "localhost:5000*".

### a. How to use  <a name="howto">
  
![API Home page](/img/api_home.PNG)
  
Click on "Parcourir" for each channel and make sure you send :
  - image according to the correct channels.
  - four images which belong to the same observation/sample.
  
Then please wait. Dependiong on your hardware, the HPA Cell Segmentator and our Tensorflow "Organelle Classifier" could take up to 30 seconds.
  
Once the loading and predicting process are over, the result will appear in another page.
  
![API Result page](/img/api_result.PNG)
 
The recomposed nucleoli, microtubule and endoplasmic reticulum channels will appear, recombined, in the top-left image.
A view of the mask and the "green" channel of interest, as well as the cell-wise bounding boxes, will appear in the top-right image.

Your four sent images will appeared, colored in the middle.
  
The prediction and confidence (namely "Prédiction" and "Confiance") will be explicitely displayed in the bottom.
  
You can then click "Accueil" to make a new prediction.

  
## 5. Thanks  <a name="thanks">

Thanks to our Image Recognition teacher, Pedro Octaviano, who spent inconsiderate amount of his free time to teach and help us produce the best we could.


 * Made by: Wael GRIBAA, Amine OMRI, Sabina REXHA, Abdou Akim GOUMBALA and Meryem GASSAB
 * Date: 25th May 2021
