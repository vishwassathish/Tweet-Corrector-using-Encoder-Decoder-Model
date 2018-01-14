import keras
import gensim
import json
import numpy as np
import csv
import nltk
print('success. Ignore the warning')


#loading data
f = open("M:/5th Sem/AI/Ass2/consolidated.csv", "r")
data = csv.reader(f, delimiter = ',')
next(data)
corrected = []
original = []

#tokenizing
for row in data :
    if (row[-1] == "1") :
        original.append(nltk.word_tokenize(row[1]))
        corrected.append(nltk.word_tokenize(row[2]))
        
print('success')

#preprocessing original text
d = {}
for i in original :
    for j in i :
        d[j] = d.get(j, 0) + 1
for i in range(len(original)) :
    for j in range(len(original[i])) :
        original[i][j] = original[i][j].lower()
print('success')

#making one hot vectors and index to word mappings
from collections import Counter, defaultdict
flat = [" ".join(x) for x in corrected]
new_list = []
for item in flat :
    new_list += item.split(' ')
for i in range(len(new_list)) :
    new_list[i] = new_list[i].lower()
c = Counter(new_list)
del new_list
new_list = []
for i, j in c.items() :
    if j > 2 :
        new_list.append(i)
flat = list(set(new_list))
print("Length of corrected vocabulary = ", len(flat))

out_feature = len(flat)

one_hot_corrected = {}
index_to_word = {}
k = [0 for x in range(len(flat)+1)]
for i in range(len(flat)) :
    temp = k[:]
    temp[i]= 1
    one_hot_corrected[flat[i]] = temp[:]
    index_to_word[i] = flat[i]
del flat
k[-1] = 1
one_hot_corrected = defaultdict(lambda : k[:], one_hot_corrected) #take __unk__
index_to_word = defaultdict(lambda : "__unk__", index_to_word)

print('success')

#word2vec
IN_FEATURES = 100
word2vec_original = gensim.models.Word2Vec(original + [["__unk__"]], min_count=3, size = IN_FEATURES)
print('success')


#creating x and y data
NB_WORDS = 20
master_x = []
fifty_zeroes = IN_FEATURES * [0]
for i in original :
    tmp = []
    for j in i :
        if (j in word2vec_original and len(tmp) < NB_WORDS) :
            tmp.append(word2vec_original[j])
        elif len(tmp) < NB_WORDS :
            tmp.append(100 * [0])
    while (len(tmp) < NB_WORDS) :
        tmp.append(fifty_zeroes)
    master_x.append(tmp)

master_y = []
for sentence in corrected :
    new = []
    for word in sentence :
        new.append(one_hot_corrected[word])
    new = new[:NB_WORDS]
    while(len(new) < NB_WORDS) :
        new.append(one_hot_corrected["__unk__"])
    master_y.append(new[:])
print('success')


#splitting data into training and validation
INPUT_SAMPLE_SIZE = 4500

VALIDATION_SPLIT = 0.1
master_x = np.array(master_x)
master_y = np.array(master_y)
indices = np.arange(master_x.shape[0])
np.random.shuffle(indices)
master_x = master_x[indices]
master_y = master_y[indices]

x_data = master_x[:INPUT_SAMPLE_SIZE]
y_data = master_y[:INPUT_SAMPLE_SIZE]

nb_validation_samples = int(VALIDATION_SPLIT * x_data.shape[0])

print(x_data.shape, y_data.shape, nb_validation_samples)

x_train = x_data[:-nb_validation_samples]
y_train = y_data[:-nb_validation_samples]
x_val = x_data[-nb_validation_samples:]
y_val = y_data[-nb_validation_samples:]

print('success')


#model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

print(len(x_val), len(x_val[0]), len(x_val[0][0]))

n_features = IN_FEATURES
n_timesteps_in = NB_WORDS
batch_size = 100
model = Sequential()
model.add(LSTM(50, input_shape=(n_timesteps_in, n_features)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(out_feature + 1, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# train LSTM
model.fit(x_train, y_train,
          batch_size=batch_size,
          validation_data=[x_val, y_val],
          epochs=10) 