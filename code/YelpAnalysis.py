"""**Yelp Data Analysis**

In this project we perform sentimental analysis on yelp data set (https://www.yelp.com/dataset ).

review.json file is used to perform sentimental analysis on businesses.

Only first 200000 (positive + negative) reviews are extracted from the dataset and analysis is performed.

user ratings > 3 are positive else negative.

Final classification accuracies by different models are reported :


1.   RF - Random forest classifier
2.   DENSE - Dense feed forward network
3.   CONV - Processing with COVNET
4.   RNN - Simple recurrent neural network
5.   LSTM - Long short term memory
6.   BI-LSTM - Bi directional LSTM

Review JSON path
"""

import os,sys
review_path = "review.json";
if os.path.isfile(review_path):
    pass
else:
    print("Invalid file path. Please place the python file in the same folder as review.json and rerun.")
    sys.exit(0)

"""Required libraries"""

from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Bidirectional, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import RMSprop
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline

"""Read json"""

max_records = 100000
x_raw = []
y_raw = []
reviews = pd.read_json(review_path,lines=True,chunksize=max_records)
pos = 0
neg = 0
for chunks in reviews:
  if len(y_raw) == max_records*2:
    break
  for index, rec in chunks.iterrows():
    y = rec["stars"]
    if pos < max_records and y > 3:            
      pos = pos + 1
      x_raw.append(rec["text"])
      y_raw.append(1)
    elif neg < max_records:
      neg = neg + 1
      x_raw.append(rec["text"])
      y_raw.append(0)
      
print("Length of data : ",len(x_raw))

"""Only first 150 words are used for review (**input_shape**).

**vocab_len** is a parameter which is used to train Embedding layer whose size is equal to the number of unique words in sequences.
"""

input_shape = 100
vocab_len = None

"""* Stop words which doesn't add much meaning to the sequences are filtered.
* Tokens are padded to same length and vectorized
"""

x_text = []
for w in x_raw:
    arr = [s for s in text_to_word_sequence(w) if not s in STOPWORDS]  
    x_text.append(arr)
t = Tokenizer()
t.fit_on_texts(x_text)
vocab_len = len(t.word_index) + 1

sequences = t.texts_to_sequences(x_text)
data = pad_sequences(sequences, maxlen=input_shape)
labels = np.asarray(y_raw)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

"""Split data into train, test and validation set."""

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.1,random_state=101)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=102)
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_val:', x_val.shape)
print('y_val:', y_val.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)

"""Function return model of specified type

* name: Name of the model 
* vocab_len : Embedding vocabulary size
* inp_shape : Input shape
"""

def getModel(name,vocab_len,inp_shape):
    if name == "DENSE":
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(inp_shape,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    elif name == "RNN":
        model = Sequential()
        model.add(Embedding(vocab_len, 32))
        model.add(SimpleRNN(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
        model.add(SimpleRNN(32, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    elif name == "LSTM":
        model = Sequential()
        model.add(Embedding(vocab_len, 32))
        model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
        model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    elif name == "BI-LSTM":
        model = Sequential()
        model.add(Embedding(vocab_len, 32))
        model.add(Bidirectional(LSTM(32, return_sequences=True,dropout=0.1, recurrent_dropout=0.1)))
        model.add(Bidirectional(LSTM(32, dropout=0.1, recurrent_dropout=0.1)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    elif name == "CONV":
        model = Sequential()
        model.add(Embedding(vocab_len, 32))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    elif name == "RF":
        model = RandomForestClassifier(verbose=0,n_jobs=-1,random_state=136)
        return model

"""Plot accuracies"""

def plot_accuracy(model_name,history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(model_name+' - Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

"""Train model and predict accuracies"""

model_list = ["RF","DENSE","CONV","RNN","LSTM","BI-LSTM"]
accuracy_report = []
for name in model_list:
    print("Processing model : ",name);
    model = getModel(name,vocab_len,input_shape)
    y_pred = None
    
    if name == "DENSE" or name == "RNN" or name == "LSTM" or name == "BI-LSTM" or name == "CONV":
        history = model.fit(x_train, y_train,epochs=5,batch_size=128,validation_data=(x_val, y_val)) 
        plot_accuracy(name,history)
        y_pred = model.predict_classes(x_test)        
    elif name == "RF":
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
    accuracy_report.append([
        format(accuracy_score(y_test,y_pred)*100,'.2f'),
        format(precision_score(y_test,y_pred)*100,'.2f'),
        format(recall_score(y_test,y_pred)*100,'.2f'),
        format(f1_score(y_test,y_pred)*100,'.2f')])
    print(accuracy_report[-1])

from prettytable import PrettyTable
print("Testing results:")
t = PrettyTable(['Model','Acuracy','Precision','Recall','F1-Score'])
for a,b in zip(model_list,accuracy_report):
    t.add_row([a,b[0],b[1],b[2],b[3]])
print(t)