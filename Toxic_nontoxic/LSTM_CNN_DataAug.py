import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from functools import reduce
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import GlobalMaxPool1D
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import nlpaug.augmenter.word as naw
from sklearn.utils import shuffle
import collections

data = pd.read_csv('Toxic_nontoxic/data.csv')
data = data.dropna(how='any', axis=0)

x=data["toxic"].value_counts()
x_nontoxic = x[0]
x_toxic = x[1]

print("Number of Non Toxic comments = " , x_nontoxic)
print("Number of Toxic comments = " , x_toxic)
print("Dataset shape:",data.shape)   # (159571 , 8)

# To check for missing values in train data
print("---------Check for missing values in train data------------")
print(data.isnull().sum())

data_read = pd.read_csv("Toxic_nontoxic/toxic-preprocessed-data.csv")

print("--------------------------------------------------------")
print(data_read.isna().sum())

data_read = data_read.replace(np.nan, '', regex=True)
print("--------------------------------------------------------")
print(data_read.isna().sum())

X_data = data_read["comment_text"].values
y_label = data_read["toxic"].values


""""Splitting data"""

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.30, random_state=42 , shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.20, random_state=42 , shuffle=True)

print('Number of entries in each category:')
print('training: ', X_train.shape)
print('validation: ', X_val.shape)
print('testing: ', X_test.shape)

count_val = collections.Counter(y_train)
print(count_val)
num_toxic = count_val[1]
num_nontoxic =count_val[0]

print("Number of toxic comments in training data=",num_toxic)
print("Number of non-toxic comments in training data=",num_nontoxic)


#-----------------Data Augmentation-------------------------

aug_w2v = naw.WordEmbsAug(
    model_type='fasttext', model_path='wiki-news-300d-1M.vec/wiki-news-300d-1M.vec',
    action="substitute")


def augment_text(df,samples=300,pr=0.2):
    aug_w2v.aug_p=pr
    new_text=[]
    
    ##dropping samples from validation
    df_n=df[df["toxic"]==1].reset_index(drop=True)

    ## data augmentation loop
    for i in tqdm(np.random.randint(0,len(df_n),samples)):
        
            text = df_n.iloc[i]['comment_text']
            augmented_text = aug_w2v.augment(text)
            new_text.append(augmented_text)
    
    
    ## dataframe
    new=pd.DataFrame({'comment_text':new_text,'toxic':1})
    df=shuffle(df.append(new).reset_index(drop=True))
    return df

df_Xtrain = pd.DataFrame(X_train , columns=["comment_text"])
df_Ytrain = pd.DataFrame(y_train,columns=["toxic"])

df_train = pd.concat([df_Xtrain , df_Ytrain] , axis=1)

num_toxic_aug = 10000
train_new = augment_text(df_train,samples=num_toxic_aug)   ## change samples to 0 for no augmentation
X_train = train_new
print("Shape of augmented train=",train_new.shape)

#------Save the augmented train dataset to csv--------------

train_new.to_csv("toxic_nontoxic/Train_dataAugmented_LSTM_CNN.csv" , index=False)

train_data = pd.read_csv("toxic_nontoxic/Train_dataAugmented_LSTM_CNN.csv")
print("Train shape after aumentation = ",train_data.shape)

print("--------------------------------------------------------")
print(train_data.isna().sum())

train_data = train_data.replace(np.nan, '', regex=True)
print("--------------------------------------------------------")
print(train_data.isna().sum())


X_train = train_data["comment_text"].values
y_train = train_data["toxic"].values

print(X_train.shape)
print(y_train.shape)


#------Model training------------------------

max_features=100000      
max_pad_len = 150              
embedding_dim_fasttext = 300

# Tokenization
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

f = open('wiki-news-300d-1M.vec/wiki-news-300d-1M.vec', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index_fasttext[word] = np.asarray(values[1:], dtype='float32')
f.close()

embedding_matrix_fasttext = np.random.random((len(word_index) + 1, embedding_dim_fasttext))
for word, i in word_index.items():
    embedding_vector = embeddings_index_fasttext.get(word)
    if embedding_vector is not None:
        embedding_matrix_fasttext[i] = embedding_vector
print(" Completed!")

#------------LSTM-CNN---------------------

inp=Input(shape=(max_pad_len, ),dtype='int32')

embedding_layer = Embedding(len(word_index) + 1,
                           embedding_dim_fasttext,
                           weights = [embedding_matrix_fasttext],
                           input_length = max_pad_len,
                           trainable=False,
                           name = 'embeddings')
embedded_sequences = embedding_layer(inp)

x = LSTM(units = 64, return_sequences=True,name='lstm_layer')(embedded_sequences)

x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(x)

x = MaxPooling1D(3)(x)
  
x = GlobalMaxPool1D()(x)
  
x = BatchNormalization()(x)
  
x = Dense(50, activation="relu", kernel_initializer='he_uniform')(x)
  
x = Dropout(0.2)(x)

x = Dense(40, activation="relu", kernel_initializer='he_uniform')(x)
  
x = Dropout(0.2)(x)
  
preds = Dense(1, activation="sigmoid", kernel_initializer='glorot_uniform')(x)

model = Model(inputs=inp, outputs=preds)

model.summary()


model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X_train, y_train, batch_size = 32, epochs = 1, validation_data = (X_val, y_val))

### Prediction for test data
y_predict = model.predict(X_test)

print("--------------------------------------")
print(" Predicted values are",y_predict)


accuracy = model.evaluate(X_test , y_test)
f1score = f1_score(y_test, y_predict.round())

print("--------LSTM-CNN with Data Augmentation------------------------------")
print(" Accuracy =",accuracy)
print("F1-Score : {}".format(f1score))

print("Completed")

