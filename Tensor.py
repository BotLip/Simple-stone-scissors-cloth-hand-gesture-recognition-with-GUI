import tensorflow as tf
from tensorflow import keras, data
from keras.callbacks import TensorBoard
import os
import numpy as np
from numpy.random import shuffle

file_path = 'Hand'


def random_list(image_list, label_list):
    pairs = np.array([image_list, label_list])
    pairs = pairs.transpose()
    shuffle(pairs)
    pairs = np.array(pairs)
    image_list_randomed = list(pairs[:, 0])
    label_list_randomed = list(pairs[:, 1])
    # image_list = [i for i in image_list]
    # 下面必须把string转int，tf.cast不支持string转int64 报错
    label_list_randomed= [int(float(i)) for i in label_list_randomed]

    return image_list_randomed, label_list_randomed


def get_files(file_dir):

    list_r = []
    label_r = []

    list_p = []
    label_p = []

    list_s = []
    label_s = []

    for image_file in os.listdir(file_dir):
        image_file_path = os.path.join(file_dir, image_file)
        for image_name in os.listdir(image_file_path):
            image_name_path = os.path.join(image_file_path, image_name)

            if image_file_path[-1] == 'R':
                list_r.append(image_name_path)
                label_r.append(0)
            elif image_file_path[-1] == 'P':
                list_p.append(image_name_path)
                label_p.append(1)
            elif image_file_path[-1] == 'S':
                list_s.append(image_name_path)
                label_s.append(2)

    image_list = np.hstack((list_r, list_p, list_s))
    label_list = np.hstack((label_r, label_p, label_s))
    image_list, label_list = random_list(image_list, label_list)

    return image_list, label_list


def get_tensor(image_list, label_list):
    images = []
    for image_name_path in image_list:
        image = tf.io.read_file(image_name_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [64, 64])
        image /= 255.0
        images.append(image)

    image_tensor = tf.convert_to_tensor(images)
    label_tensor = tf.convert_to_tensor(label_list)

    return image_tensor, label_tensor


def get_dataset(image_tensor, label_tensor):
    train_data = data.Dataset.from_tensor_slices((image_tensor, label_tensor))
    return train_data


def train(train_data):

    moudel = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                            input_shape=(64, 64, 3)),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Conv2D(64, kernel_size=3, activation='relu'),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(3, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)
    moudel.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    moudel.fit(train_data, epochs=10,  callbacks=[TensorBoard(log_dir=r'D:\py.train\hand_recognition\mytensorboard')])

    moudel.save('hand_recognition.h5')
    print('模型训练完毕\n')

'''
image_list, label_list = get_files(file_path)
image_tensor, label_tensor = get_tensor(image_list, label_list)
train_data = get_dataset(image_tensor, label_tensor)
train_data = train_data.shuffle(1000).batch(16)     # 16x28x28x3
train(train_data)
'''








