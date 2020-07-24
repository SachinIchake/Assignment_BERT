## Problem Statement
	In this project, we have used various natural language processing techniques and machine learning algorithms to classify fake news articles using Machine Learning and Deep Learning models.

###### 	Prerequisites
	Python 3.6
	Libraries: 
	    •	Sklearn (scikit-learn)
	    •	numpy
	    •	scipy
	    •	numpy
	    •	torch
	    •	torchvision
	    •	matplotlib
	    •	tensorflow
	    •	tqdm
	    •	pandas
	    •	numpy
	    •	keras
	    •	sklearn
	    •	nltk

	pip install –r requirement.txt

###### Dataset Description
    •	train.csv: A full training dataset with the following attributes:
        o	id: unique id for a news article
        o	title: the title of a news article
        o	author: author of the news article
        o	text: the text of the article; could be incomplete
        o	label: a label that marks the article as potentially unreliable 
            	1: unreliable
            	0: reliable
    •	test.csv: A testing training dataset with all the same attributes at train.csv without the label.



###### File Structure
	.
	|
	+-- input
	++--  	data
	|   	+-- train.csv
	|   	+-- test.csv
	+-- src
	|   +-- config.py
	|   +-- dataset.py
	|   +-- mlalgorithms.py
	|   +-- engine.py
	|   +-- model.py
	|   +-- predictML.py
	|   +-- train.py
	+-- README.txt
	+-- requirements.txt

### Train Model
###### Model Parameters:
	•	MAX_LEN = 512
	•	TRAIN_BATCH_SIZE = 64
	•	TEST_BATCH_SIZE = 32
	•	EPOCHS = 10
	•	MODEL_PATH = "../input/model/model.pt"
	•	TRAINING_FILE = "../input/data/train.csv"
	•	NEW_TRAINING_FILE = "../input/data/train_with_new_features.csv"
	•	TESTING_FILE = "../input/data/test.csv" 
	•	GLOVE="../input/model/glove.6B.300d.txt"
	•	MODEL_PATH_ML = "../input/model"

###### Deep Learning
	Execute below command to train deep learning models [LSTM,BERT] and saves the model to MODEL_PATH directory
	Python train.py
	Note: As per EDA, we can use title and text or erhier one to train the model. For now, I have used text for training and inferencing.
###### Precision: 
###### Recall: 
