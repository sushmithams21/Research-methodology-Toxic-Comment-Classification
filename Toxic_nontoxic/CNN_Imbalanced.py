import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input,Embedding, Dropout
from keras.layers import GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# To read the dataset
data = pd.read_csv('Toxic_nontoxic/data.csv') 
data = data.dropna(how='any', axis=0)  # drop na columns is present

print("Dataset shape:",data.shape)   # (159571 , 8)

# To check for missing values in train data
print("---------Check for missing values in train data------------")
print(data.isnull().sum())

# Read the already preprocessed csv file
data_read = pd.read_csv("Toxic_nontoxic/toxic-preprocessed-data.csv")

print("--------------------------------------------------------")
print(data_read.isna().sum())

# Replace na entries with empty string
data_read = data_read.replace(np.nan, '', regex=True)
print("--------------------------------------------------------")
print(data_read.isna().sum())

X_data = data_read["comment_text"].values
y_label = data_read["toxic"].values


# Splitting the dataset into train,validation and test set in the ratio 70:10:20 respectively
X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.30, random_state=42 , shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.20, random_state=42 , shuffle=True)

print('Number of entries in each category:')
print('training: ', X_train.shape)
print('validation: ', X_val.shape)
print('testing: ', X_test.shape)

#------Model training------------------------

max_features=100000      
max_pad_len = 150              
embedding_dim_fasttext = 300

# Tokenization using keras tokenizer
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
list_tokenized_train = tokenizer.texts_to_sequences(X_train)
list_tokenized_test = tokenizer.texts_to_sequences(X_test)
list_tokenized_val = tokenizer.texts_to_sequences(X_val)

word_index=tokenizer.word_index
print("Words in Vocabulary: ",len(word_index))

#-----Padding----------

X_train=pad_sequences(list_tokenized_train, maxlen=max_pad_len, padding = 'post')
X_test=pad_sequences(list_tokenized_test, maxlen=max_pad_len, padding = 'post')
X_val=pad_sequences(list_tokenized_val, maxlen=max_pad_len, padding = 'post')

embedding_dim_fasttext = 300
embeddings_index_fasttext = {}

# fasttext word embedding of 300d 
f = open('wiki-news-300d-1M.vec/wiki-news-300d-1M.vec', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index_fasttext[word] = np.asarray(values[1:], dtype='float32')
f.close()

# Creating embedding matrix
embedding_matrix_fasttext = np.random.random((len(word_index) + 1, embedding_dim_fasttext))
for word, i in word_index.items():
    embedding_vector = embeddings_index_fasttext.get(word)
    if embedding_vector is not None:
        embedding_matrix_fasttext[i] = embedding_vector
print(" Completed!")

# Creating the CNN model

model=Sequential()
model.add(Embedding(max_features,embedding_dim_fasttext,input_length=max_pad_len))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128,kernel_size = 5,padding='valid',activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(filters = 128,kernel_size = 5,padding='valid',activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=24,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X_train, y_train, batch_size = 32, epochs = 1, validation_data = (X_val, y_val))

### Prediction for test data
y_predict = model.predict(X_test)

print("--------------------------------------")
print(" Predicted values are",y_predict)

#  Evaluation of test data to calculate f1-score
accuracy = model.evaluate(X_test , y_test)
f1score = f1_score(y_test, y_predict.round())

print("-----------CNN without data augmentation----------------------------")
print(" Accuracy =",accuracy)
print("F1-Score : {}".format(f1score))

print("Completed")