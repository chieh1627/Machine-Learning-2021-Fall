import numpy as np
import pandas as pd


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

from tensorflow.keras import models

model = models.load_model('./HW4_model.h5')

y_pred = model.predict(x_test)
y_pred_label = np.round(y_pred[:, -1])

# %%
import csv

with open('predict.csv', 'w', newline='') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(['id', 'label'])
    for i in range(y_pred_label.shape[0]):
        writer.writerow([i, int(y_pred_label[i])])
