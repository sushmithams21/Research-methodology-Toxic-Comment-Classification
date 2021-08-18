
# Toxic Comment Classification with data augmentation

## Table of Contents
-	Dataset Overview
-   Data Preprocessing
-   Methodology
-	Results

## Pre requisites required:

We import all the libaries we need.

import pandas as pd\
import numpy as np\
import nltk\
from tensorflow.python.keras.backend import dropout\
nltk.download('stopwords')\
nltk.download('wordnet')\
from nltk.corpus import stopwords\
stop_words = set(stopwords.words('english'))\
from keras.preprocessing.text import Tokenizer\
from keras.preprocessing.sequence import pad_sequences\
from keras.layers import Dense, Input, Embedding, Dropout\
from keras.layers import GlobalMaxPooling1D\
from keras.models import Sequential\
from keras.layers import Conv1D, MaxPooling1D\
from sklearn.model_selection import train_test_split\
from sklearn.metrics import f1_score\
from tqdm import tqdm\
import nlpaug.augmenter.word as naw\
from sklearn.utils import shuffle\
import collections\


- FastText word embedding of 300d dimension is downloaded from the below link\

  [https://fasttext.cc/docs/en/english-vectors.html](targetURL)

### Dataset Overview

Research is conducted on the dataset provided by Kaggle[9] which is multi-label Wikipedia’s talk page edits. There are 159,571 observations in the training dataset and 153,164 observations in the testing dataset. It has 2 classes namely toxic and non-toxic and 6 labels for toxic class.
The comments have been labelled by human raters for toxic behaviour. The types of toxicity are:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The training dataset is considered for this paper and the 6 labels for toxic class is considered as one class as toxic. So only 2 classes namely toxic and non-toxic are taken as target labels. FastText word embedding of 300d is considered for embedding to make use of transfer learning. The dataset is imbalanced with 15294 for toxic class being minority and 144277 for non-toxic class. The models are evaluated with data augmentation and without data augmentation and compared the results. The data augmentation method used here is synonym replacement.

### Data Preprocessing

Prior to the effective training of our machine learning models, it is necessary to ensure the training set is cleaned and free of error or missing values. So, we checked the dataset for missing values and no missing values found, performed lowercasing, stopwords removal, retained only the alphabets and removed special characters, punctuation, non-word characters ,hyperlinks, html tag and then lemmatized using nltk library.

- Punctuation removal
- Removing Characters in between Text.
- Removing Repeated Characters.
- Converting data to lower-case.
- Removing Punctuation.
- Removing unnecessary white spaces in between words.
- Removing “\n”.\
- Removing Non-English characters.
- Html removal
- Lemmatization
- Stop words removal

### Methodology

![Fig 1: Flow of proposed methodology](https://drive.google.com/uc?export=view&id=1xWk8tCzo2CSCnA9ejXNDaU5GvX5klkLK)

We proposed an ensemble model of CNN and LSTM classifiers with fastText word embeddings. As per the research conducted, many researchers have claimed that CNN and LSTM give the best accuracy compared other machine learning algorithms. LSTM can effectively preserve the characteristics of historical information in long text sequences whereas CNN can extract the local features of the text .So from analysis we proposed to use the ensemble model aiming to achieve comparatively better performance. We compared the ensemble model accuracy with CNN and LSTM models trained separately. We evaluate the model using F1-score since the dataset is imbalanced along with accuracy .The flow of proposed methodology is shown in Fig 1.

### Results

![Table 1](https://drive.google.com/uc?export=view&id=1BmKNxDjFZz9MTrLLAxHNuqOGCmyJm3x5)

![Table 2](https://drive.google.com/uc?export=view&id=1oNM9Uum4Q8tOB1QsVFWAIkywl41vJ-tO)

Though the expectation was to get good results for ensemble model(95.53%), CNN (95.83%) performed well compared to other 2 classifiers and ensemble model gave the least accuracy with respect to accuracy metric. But LSTM resulted in good F1-score(0.76754) compared to CNN and LSTM-CNN without data augmentation as shown in Table 1.

With data augmentation of 10000 samples, LSTM resulted in 95.70% of accuracy and 0.7824 of F1-score whereas CNN gave 95.69% of accuracy and 0.7782 F1-score and LSTM-CNN gave 95.54% of accuracy and 0.7727 of F1-score.The table 2 clearly shows that LSTM results were better compared to other 2 models.
