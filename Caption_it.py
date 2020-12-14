
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import numpy as np
from matplotlib import pyplot as plt
import os
from math import log
import pickle


word_to_index = None
with open('./Helper/word_to_index.pkl', 'rb') as f:
  word_to_index = pickle.load(f)



index_to_word = None
with open('./Helper/index_to_word.pkl', 'rb') as f:
  index_to_word = pickle.load(f)


model = load_model('./Model Weights/model_weights_50.h5')
model._make_predict_function()

inception = InceptionV3(weights = 'imagenet', input_shape = (299, 299, 3))


cnn_model = Model(inputs = inception.input, outputs = inception.layers[-2].output)
cnn_model._make_predict_function()

def encode(img):
    
    img = image.load_img(img,target_size=(299,299))
 
    img = image.img_to_array(img)
        
    # Normalising
    img = preprocess_input(img)

    img = np.expand_dims(img,axis = 0) ## to make img look like batch of size 1 = (1,299,299,3)

    feature_vector = cnn_model.predict(img) # shape (1,2048)
        
    return feature_vector




def predict(image, beam_width = 3, alpha = 0.7,max_len = 38):
  l = [('<start>', 1.0)]
  for i in range(max_len):
    temp = []
    for j in range(len(l)):
      sequence = l[j][0]
      prob = l[j][1]
      if sequence.split()[-1] == '<end>':
        t = (sequence, prob)
        temp.append(t)
        continue
      encoding = [word_to_index[word] for word in sequence.split() if word in word_to_index]
      encoding = pad_sequences([encoding], maxlen = max_len, padding = 'post')
      pred = model.predict([image, encoding])[0]
      pred = list(enumerate(pred))
      pred = sorted(pred, key = lambda x: x[1], reverse = True)
      pred = pred[:beam_width]
      for p in pred:
        if p[0] in index_to_word:
            t = (sequence + ' ' + index_to_word[p[0]], (prob + log(p[1])) / ((i + 1)**alpha))
            temp.append(t)
    temp = sorted(temp, key = lambda x: x[1], reverse = True)
    l = temp[:beam_width]
  caption = l[0][0]
  caption = caption.split()[1:-1]
  caption = ' '.join(caption)
  return caption



def caption_this_image(img):
    enc_img = encode(img)
    return predict(enc_img)






