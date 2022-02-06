# import package
import os
import numpy as np
import pandas as pd

from glob import glob
from PIL import Image


# %%

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


# For image gen
class ImageGen:
    def __init__(self, image: np.ndarray):
        self.img = image
        self.rows = image.shape[0]
        self.cols = image.shape[1]
        self.center = [int(self.rows / 2), int(self.cols / 2)]
        self.img_gen = np.zeros(shape=(self.rows, self.cols))
        self.transform = np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]])

    # factor > 1表縮小；factor < 1表放大。
    def Zoom(self, factor: float):
        self.transform = np.array([[factor, 0, 0],
                                   [0, factor, 0],
                                   [0, 0, 1]])
        self._Process()

    # 垂直鏡射
    def Vertically(self):
        self.center = [0, 0]
        self.transform = np.array([[-1, 0, self.rows - 1],
                                   [0, 1, 0],
                                   [0, 0, 1]])
        self._Process()

    # 水平鏡射
    def Horizontal(self):
        self.center = [0, 0]
        self.transform = np.array([[1, 0, 0],
                                   [0, -1, self.cols - 1],
                                   [0, 0, 1]])
        self._Process()

    # degree > 0, 順時針旋轉
    # degree < 0, 逆時針旋轉
    def Rotate(self, degree: int):
        deg = np.deg2rad(degree)
        self.transform = np.array([[np.cos(deg), -np.sin(deg), 0],
                                   [np.sin(deg), np.cos(deg), 0],
                                   [0, 0, 1]])
        self._Process()

    # 處理資料
    def _Process(self):
        for i in range(self.rows):
            for j in range(self.cols):
                point = np.array([i - self.center[0], j - self.center[1], 1])

                [x, y, z] = np.dot(self.transform, point)

                x = int(x) + self.center[0]
                y = int(y) + self.center[1]

                # 超過範圍補 0
                if x >= self.rows or y >= self.cols or x < 0 or y < 0:
                    self.img_gen[i][j] = 0
                else:
                    self.img_gen[i][j] = self.img[x][y]


# %%

train_img_path = get_img_path('data\\train')
test_img_path = get_img_path('data\\test')

# %%

x_train = read_img(train_img_path)
x_test = read_img(test_img_path)

# read y_train from 'train.csv' and convert pd.dataframe to np.ndarray, then extracting label.
y_train = pd.read_csv('data/train.csv').values[:, [1]]

# %%

# free memory
del train_img_path, test_img_path


# %%

def Img_Horizontal(data: np.ndarray) -> np.ndarray:
    img = np.zeros(shape=(len(x_train), 64, 64))
    for i in range(len(data)):
        Gen = ImageGen(data[i])
        Gen.Horizontal()
        img[i] = Gen.img_gen
        del Gen
    return img


def Img_Zoom(data: np.ndarray, factor: int) -> np.ndarray:
    img = np.zeros(shape=(len(x_train), 64, 64))
    for i in range(len(data)):
        Gen = ImageGen(data[i])
        Gen.Zoom(factor)
        img[i] = Gen.img_gen
        del Gen
    return img


def Img_Rotate(data: np.ndarray, deg: int) -> np.ndarray:
    img = np.zeros(shape=(len(x_train), 64, 64))
    for i in range(len(data)):
        Gen = ImageGen(data[i])
        Gen.Rotate(deg)
        img[i] = Gen.img_gen
        del Gen
    return img


# 水平翻轉
x_train_flip = Img_Horizontal(x_train)
# 放大10%
x_train_zoom = Img_Zoom(x_train, 0.9)
x_train_Rotate_1 = Img_Rotate(x_train, 10)
x_train_Rotate_2 = Img_Rotate(x_train, -10)
x_train_Rotate_3 = Img_Rotate(x_train_flip, 10)
x_train_Rotate_4 = Img_Rotate(x_train_flip, -10)

# %%

# merge data.
x_train = np.concatenate([x_train, x_train_Rotate_1, x_train_Rotate_2, x_train_Rotate_3, x_train_Rotate_4, x_train_zoom, x_train_flip])
y_train = np.concatenate([y_train, y_train, y_train, y_train, y_train, y_train, y_train])

# %%

# shuffle merge data.
index = np.arange(len(x_train))
np.random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]

# %%

x_train_pd = pd.DataFrame(x_train.reshape(-1, 64 * 64))
y_train_pd = pd.DataFrame(y_train.reshape(-1, 1))

x_train_pd.to_csv('x_train_gen.csv', index=False, header=False)
y_train_pd.to_csv('y_train_gen.csv', index=False, header=False)

# %%

x_test_pd = pd.DataFrame(x_test.reshape(-1, 64 * 64))
x_test_pd.to_csv('x_test.csv', index=False, header=False)
