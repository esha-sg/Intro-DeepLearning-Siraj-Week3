import pandas as pd
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences, VocabularyProcessor

from sklearn.cross_validation import train_test_split
import numpy as np

ign = pd.read_csv("ign.csv")

#Data Preprocessing
#Fill null with empty strings
ign.fillna(value='',inplace=True)
#Adding new column sentiment based on score_phrase
bad = ['Bad', 'Awful', 'Painful', 'Unbearable', 'Disaster']
ign['sentiment'] = ign.score_phrase.isin(bad).map({True: 'Negative', False: 'Positive'})
#Adding column text which contains information about many columns
ign['text'] = ign['title'].str.cat(ign['platform'], sep=' ').str.cat(ign['genre'], sep=' ').\
    str.cat(ign['editors_choice'], sep=' ').str.cat(ign['sentiment'], sep = ' ').str.cat(ign['score'].apply(str), sep = ' ')

#Required columns for input and putput
X = ign.text
Y = ign.score_phrase

#Convert the strings in input to vectors
vocabX = VocabularyProcessor(20)
X = np.array(list(vocabX.fit_transform(X)))
#11 classes to predict from 0-10
vocabY = VocabularyProcessor(1)
Y = np.array(list(vocabY.fit_transform(Y))) - 1
#Convert to 11 dimensional vectors
Y = to_categorical(Y, nb_classes = 11)

#Split into training and testing set
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2)

#Build the network
#Input has length of 20
nn = tflearn.input_data([None, 20])
#Word vectors are then casted into 128 dimensions each creating embedding
nn = tflearn.embedding(nn, input_dim=10000, output_dim=128)
#Each input is then sent to two LSTM layers
nn = tflearn.lstm(nn, 128, dropout=0.7, return_seq=True)
nn = tflearn.lstm(nn, 128, dropout=0.7)
#Output is sent to fully connected neural network
nn = tflearn.fully_connected(nn, 11, activation='softmax')
#Adam optimizer is used for regression
nn = tflearn.regression(nn, optimizer='adam', learning_rate=0.001,
						 loss='categorical_crossentropy')

model = tflearn.DNN(nn, tensorboard_verbose=0)

model.load('./games.tfl')
Model = model

pred = [np.argmax(i) for i in Model.predict(testX)]
true = [np.argmax(i) for i in testY]

print('RNN Classifier\'s Accuracy: %f\n' % metrics.accuracy_score(true, pred))
