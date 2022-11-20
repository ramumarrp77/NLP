# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 01:17:36 2019

@author: Ram Kumar R P
"""

import tensorflow_datasets as tdfs
imdb1,info = tdfs.load("imdb_reviews/subwords8k",with_info=True, as_supervised=True)

train_data , test_data = imdb1['train'], imdb1['test']

tokenizer = info.features['text'].encoder

print(tokenizer.subwords)

sample_string = 'Tensorflow, from basics to mastery'

tokenized_string = tokenizer.encode(sample_string)

print('The Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)

print('The original string is {}'.format(original_string))


for ts in tokenized_string:
    print('{} ---->{}'.format(ts,tokenizer.decode([ts])))
import tensorflow as tf

embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.summary()

num_epochs = 10

model.compile(loss ='binary_crossentropy',
              optimizer ='adam',
              metrics = ['accuracy'])

history =  model.fit(train_data, epochs = num_epochs,validation_data = test_data)

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_"+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

