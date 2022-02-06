import numpy as np
import pandas as pd

# 看每個評論的前50個字，多的去除，不足填充。
MAX_POSITIONS_LEN = 50

# %%

# read data.
train = pd.read_csv('data\\train_label.csv')
test = pd.read_csv('data\\test.csv')
unlabel = pd.read_csv('data\\train_nolabel.csv')

# %%

# spilt data.
x_train = train['text'].values
y_train = train['label'].values

x_test = test['text'].values

x_unlabel = unlabel['text'].values

# %%

# 單字的one hot encoding
token_index = {}

for x in x_train:
    for word in x.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

for x in x_unlabel:
    try:
        for word in x.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1
    except:
        continue


# %%

x_train_list = []
idx = 0
for x in x_train:
    x_train_list.append([])
    for word in x.split():
        x_train_list[idx].append(token_index[word])
    idx += 1

x_test_list = []
idx = 0
for x in x_test:
    x_test_list.append([])
    for word in x.split():
        try:
            x_test_list[idx].append(token_index[word])
        except:
            continue
    idx += 1


# %%
from keras.layers import Embedding
from keras.preprocessing import sequence

# x_train = sequence.pad_sequences(x_train_list, maxlen=MAX_POSITIONS_LEN)
# x_test = sequence.pad_sequences(x_test_list, maxlen=MAX_POSITIONS_LEN)

# %%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, SimpleRNN
from tensorflow.keras.optimizers import RMSprop, Adam

MAX_FEATURES = max(token_index.values()) + 1
# MAX_FEATURES = 100000

model = Sequential()

model.add(Embedding(10000, MAX_POSITIONS_LEN))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(),
              loss='binary_crossentropy',
              metrics=['acc'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 如果val_acc超過3輪訓練後沒有改善，就停止訓練。
# 儲存最佳val_acc的模型。
callback_list = [EarlyStopping(monitor='val_acc', patience=5),
                 ModelCheckpoint(filepath='HW4_model.h5', monitor='val_acc', save_best_only=True)]

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=callback_list)

