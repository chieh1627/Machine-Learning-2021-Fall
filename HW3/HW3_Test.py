import os
import csv
import numpy as np
from tensorflow import keras

from glob import glob
from PIL import Image


# get img file path
def get_img_path(file_Path: str) -> np.ndarray:
    img_Path = np.array(glob(os.path.join(file_Path + '\\*.jpg')))

    file_index = []
    for file in img_Path:
        file_index.append(int(file[len(file_Path) + 1: -4]))

    file_index = np.array(file_index)
    sort_index = np.argsort(file_index)

    return np.array(img_Path[sort_index])


# convert .img to np.ndarray
def read_img(img_Path: str) -> np.ndarray:
    # append all image to list.
    img = []
    for i in img_Path:
        with Image.open(i) as image:
            # convert PIL image to np.ndarray, and append to img(list)
            img.append(np.array(image))

    return np.array(img)


if __name__ == '__main__':
    # load data from x_test.csv(pd.dataframe -> np.ndarray)
    # x_test = pd.read_csv('x_test.csv', header=None).values

    # load data from .img files.
    x_test = read_img(get_img_path('data\\test'))

    # convert shape to (64, 64, 1)
    x_test = x_test.reshape(-1, 64, 64, 1)

    # compress pixes from [0, 255] to [0, 1]
    x_test = x_test / 255

    # load model
    model = keras.models.load_model('HW3_model.h5')

    # predict x_test(return each classes probability)
    y_pred_prob = model.predict(x_test)

    # find the max probability and return its index(label).
    y_pred = np.argmax(y_pred_prob, axis=1).reshape(-1, 1)

    # write in the predict results to 'HW3_v4.csv'
    with open('predict.csv', 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['id', 'label'])
        for i in range(int(y_pred.shape[0])):
            writer.writerow([i, int(y_pred[i])])
