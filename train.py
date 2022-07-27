import numpy as np
from keras.callbacks import ModelCheckpoint
from scipy.io import loadmat

from model import EEGNet2, DeepConvNet, ShallowConvNet

data = loadmat("C:/Users/user/Desktop/proprecess12/s2/right_data.mat")
data = data['data']
label = loadmat("C:/Users/user/Desktop/proprecess12/s2/right_label.mat")
label = label['right_label']
print(label.shape)
shape = list(data.shape)
shape.append(1)
data = data.reshape(shape)
np.random.seed(31)
np.random.shuffle(data)
np.random.seed(31)
np.random.shuffle(label)
print(data.shape)
print(label.shape)
model = EEGNet2()

train_data=data[0:(int)(len(data)*0.8)]
train_label=label[0:(int)(len(label)*0.8)]
val_data = data[(int)(len(data)*0.8):len(data)]
val_label = label[(int)(len(label)*0.8):len(label)]

# val_data = loadmat("C:/Users/user/Desktop/proprecess12/s2/bottom_data.mat")
# val_data = val_data['data']
# val_label = loadmat("C:/Users/user/Desktop/proprecess12/s2/bottom_label.mat")
# val_label = val_label['bottom_label']
# shape = list(val_data.shape)
# shape.append(1)
# val_data = val_data.reshape(shape)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics = ['accuracy'])
checkpointer = ModelCheckpoint(filepath='checkpoint_3_10.h5',monitor='val_loss',verbose=1, save_best_only=True)

fittedModel = model.fit(train_data,train_label, batch_size = 32, epochs =50, verbose = 1,validation_split=0.1,
                            callbacks=[checkpointer])

model = EEGNet2()
model.load_weights('checkpoint_3_10.h5')
probs = model.predict(val_data, batch_size = 64, verbose = 1)
preds = probs.argmax(axis=-1)
count=0
for i in range(len(val_label)):
    if preds[i] == val_label[i][0]:
        count=count+1
acc = count/len(val_label)
print(count)
print(len(val_label))
print(acc)
print("success")