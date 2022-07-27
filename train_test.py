# 训练集非叠加平均 测试集叠加平均
import numpy as np
from keras.callbacks import ModelCheckpoint
from scipy.io import loadmat

from model import EEGNet2

data = loadmat("C:/Users/user/Desktop/mean_data15_1/s1/merge_data.mat")
data = data['merge_data']
label = loadmat("C:/Users/user/Desktop/mean_data15_1/s1/merge_label.mat")
label = label['label']
print(label.shape)
shape = list(data.shape)
shape.append(1)
data = data.reshape(shape)
np.random.seed(131)
np.random.shuffle(data)
np.random.seed(131)
np.random.shuffle(label)
print(data.shape)
print(label.shape)
ans10 = []
ans20 = []
ans30 = []
for k in range(10):
    ans10.append(0)
    ans20.append(0)
    ans30.append(0)
for t in range(10):
    model = EEGNet2()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='checkpoint_3_10.h5', monitor='val_accuracy', verbose=1,
                                   save_best_only=True)

    fittedModel = model.fit(data, label, batch_size=64, epochs=10, verbose=1, validation_split=0.2,
                            callbacks=[checkpointer])

    val_data = loadmat("C:/Users/user/Desktop/mean_data15_1/s2/merge_data.mat")
    val_data = val_data['merge_data']
    val_label = loadmat("C:/Users/user/Desktop/mean_data15_1/s2/merge_label.mat")
    val_label = val_label['label']

    shape = list(val_data.shape)
    shape.append(1)
    val_data = val_data.reshape(shape)
    model = EEGNet2()
    model.load_weights('checkpoint_3_10.h5')
    probs = model.predict(val_data, batch_size=64, verbose=1)
    preds = probs.argmax(axis=-1)
    count = 0
    for i in range(len(val_label)):
        if preds[i] == val_label[i][0]:
            count = count + 1
    acc = count / len(val_label)
    print(count)
    print(len(val_label))
    print(acc)
    print(t)
    ans10[t] = acc
    model = EEGNet2()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='checkpoint_3_10.h5', monitor='val_accuracy', verbose=1,
                                   save_best_only=True)

    fittedModel = model.fit(data, label, batch_size=64, epochs=20, verbose=1, validation_split=0.2,
                            callbacks=[checkpointer])

    model = EEGNet2()
    model.load_weights('checkpoint_3_10.h5')
    probs = model.predict(val_data, batch_size=64, verbose=1)
    preds = probs.argmax(axis=-1)
    count = 0
    for i in range(len(val_label)):
        if preds[i] == val_label[i][0]:
            count = count + 1
    acc = count / len(val_label)
    print(count)
    print(len(val_label))
    print(acc)
    print(t)
    ans20[t] = acc
    model = EEGNet2()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='checkpoint_3_10.h5', monitor='val_accuracy', verbose=1,
                                   save_best_only=True)

    fittedModel = model.fit(data, label, batch_size=64, epochs=30, verbose=1, validation_split=0.2,
                            callbacks=[checkpointer])

    model = EEGNet2()
    model.load_weights('checkpoint_3_10.h5')
    probs = model.predict(val_data, batch_size=64, verbose=1)
    preds = probs.argmax(axis=-1)
    count = 0
    for i in range(len(val_label)):
        if preds[i] == val_label[i][0]:
            count = count + 1
    acc = count / len(val_label)
    print(count)
    print(len(val_label))
    print(acc)
    print(t)
    ans30[t] = acc

print(ans10)
print("10轮")
print(sum(ans10) / 10)
print(ans20)
print("20轮")
print(sum(ans20) / 10)
print(ans30)
print("30轮")
print(sum(ans30) / 10)