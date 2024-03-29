# -*- coding: utf-8 -*-
"""test-script.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OLTuVviTAwXykR-Ofxg3phW_hnLTq5LL
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def get_dataset(data_path, model_type, image_size):
    train_data_path, test_data_path = get_images_paths(data_path)

    train_data = read_images(train_data_path, model_type, image_size)
    test_data = read_images(test_data_path, model_type, image_size)

    shuffle(train_data)

    return train_data, test_data


def get_images_paths(data_path):
    train_images_path = []
    test_images_path = []

    for class_folder in os.listdir(data_path):
        for sub_folder in os.listdir(data_path + '/' + class_folder):
            for img in os.listdir(data_path + '/' + class_folder + '/' + sub_folder):

                image_path = os.path.join(data_path + '/' + class_folder + '/' + sub_folder, img)
                if image_path.endswith(".csv"):
                    continue

                if sub_folder == 'Train':
                    train_images_path.append(image_path)
                else:
                    test_images_path.append(image_path)

    # print(train_images_path)
    return train_images_path, test_images_path


def read_images(images_paths, model_type, image_size):
    images = []

    for i in images_paths:
        image = cv2.imread(i, 0)
        image = resize_image(image, image_size, model_type)

        image_label = create_label(i)
        images.append([np.array(image), image_label])

    return images


def resize_image(image, image_size, model_type):
    if model_type == 'HOG':
        return cv2.resize(image, (image_size, 2 * image_size))
    else:
        return cv2.resize(image, (image_size, image_size))


def create_label(image_path):
    image_label = image_path.split('/')[5]
    image_Classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    label_encoded = np.zeros((20, 1))

    for i in range(len(image_Classes)):
        if image_label == str(i + 1):
            label_encoded[i] = 1

    return label_encoded

test=r'C:/Users/Update/Documents/Test Samples Classification'
train_data, test_data = get_dataset(test,model_type='CNN', image_size=128)
model = keras.models.load_model(r'C:/Users/Update/Downloads/basic_CNN_Model_ExtraTrain')
CNN.test_model(test_data, model, visualise=False)