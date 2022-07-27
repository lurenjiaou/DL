import numpy as np
from keras.callbacks import ModelCheckpoint
from scipy.io import loadmat

from model import EEGNet2

data = loadmat("C:/Users/ou/Desktop/proprecess/s1/bottom_data.mat")
data = data['data']
label = loadmat("C:/Users/ou/Desktop/proprecess/s1/bottom_label.mat")
label = label['bottom_label']
shape = list(data.shape)
shape.append(1)
data = data.reshape(shape)

model = EEGNet2()
model.load_weights('checkpoint_3_10.h5')
probs = model.predict(data, batch_size = 64, verbose = 1)
preds = probs.argmax(axis=-1)
count=0
for i in range(len(label)):
    if preds[i] == label[i][0]:
        count=count+1
acc = count/len(label)
print(count)
print(len(label))
print(acc)