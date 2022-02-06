import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras


def create_Model() -> keras.models:
    # create CNN model.
    model = keras.models.Sequential()

    # input shape.
    model.add(keras.layers.Input(shape=(64, 64, 1)))

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(7, activation='softmax'))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['acc'])

    return model


def load_data() -> np.ndarray:
    # load data from x_train_gen.csv and y_train_gen.csv(pd.dataframe -> np.ndarray)
    x_train_gen = pd.read_csv('x_train_gen.csv', header=None).values
    y_train_gen = pd.read_csv('y_train_gen.csv', header=None).values

    # convert shape from (4096, 1) to (64, 64, 1)
    x_train_gen = x_train_gen.reshape(-1, 64, 64, 1)
    y_train_gen = y_train_gen.reshape(-1, 1)

    # compress pixes from [0, 255] to [0, 1]
    x_train_gen = x_train_gen / 255

    return x_train_gen, y_train_gen


def shuffle_data(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # shuffle data
    index = np.arange(len(x))
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    return x, y


# 訓練環境：
# CPU：Intel-10700K
# GPU：GIGABYTE RTX 3080 10GB
# MEM：Kingston 64GB(訓練過程約20GB上下)

if __name__ == '__main__':
    # load data.
    x_train_gen, y_train_gen = load_data()

    # shuffle data.
    x_train_gen, y_train_gen = shuffle_data(x_train_gen, y_train_gen)

    # create CNN model.
    model = create_Model()

    # training model.
    history = model.fit(x_train_gen, y_train_gen,
                        epochs=50,
                        batch_size=64,
                        shuffle=True,
                        validation_split=0.2,
                        verbose=True)

    # save model.
    model.save('HW3_model.h5')
