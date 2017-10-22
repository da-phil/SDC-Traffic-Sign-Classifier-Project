#!/usr/bin/env python

import os, sys
import pickle
import numpy as np
import csv
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from sklearn import preprocessing
import cv2
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from keras.preprocessing.image import ImageDataGenerator
from tabulate import tabulate

training_file   = "../train.p"
validation_file = "../valid.p"
testing_file    = "../test.p" 
labels_file     = "signnames.csv"

colorspace = "RGB"  # ["RGB", "YUV", "LAB", "BW"]
show_signs = False


def normalize_for_imshow(image):
    # uint8: leave in range [0-255]
    if image.dtype == np.dtype('uint8'):
        return image
    else: # float32: scale value to range [0, 1]
        if np.min(image) < 0:
            return (image + 1.0) / 2.0
        else:
            return (image - image.mean()) / image.std() 
        

def visualize_dataset(data, indexes, imgs_per_class=10, classes=43, shuffle=False):
    #norm = colors.LogNorm(X_mean + 0.5 * X_std, 1.0, clip='True')
    #norm = colors.LogNorm(vmin=-1.0, vmax=1.0)
    #norm = colors.LogNorm(image.mean() + 0.5 * image.std(), image.max(), clip='True')
    print("min val: %f, max val: %f" % (np.min(data), np.max(data)))
    print("mean: %f, std: %f" %(data.mean(), data.std()))
    data_conv = np.copy(data)

    imshow_cmap = None
    if data.shape[-1] == 1:  # or colorspace == "BW"
        imshow_cmap = "gray"

    if colorspace == "YUV":
        for i, img in enumerate(data):
            data_conv[i] = np.reshape(cv2.cvtColor(img, cv2.COLOR_YUV2RGB), data.shape[1:])
    elif colorspace == "LAB":
        for i, img in enumerate(data):
            data_conv[i] = np.reshape(cv2.cvtColor(img, cv2.COLOR_Lab2RGB), data.shape[1:])

    fig, axarray = plt.subplots(classes, imgs_per_class, sharex=True, sharey=True)
    if shuffle:
    	class_list = random.sample(range(n_classes), classes)
    else:
        class_list = range(classes)

    for i, sign_class in enumerate(class_list):
        print("Showing example images from class %d: %s" % (sign_class, labels[sign_class]))
        axarray[i, 0].set_title(labels[sign_class] + " (class %d)" % sign_class, fontdict={'fontsize': 14})
        rand_items = random.sample(list(indexes[sign_class][0]), imgs_per_class)
        for count, item in enumerate(rand_items):
            image = data_conv[item].squeeze()
            axarray[i, count].set_axis_off()
            axarray[i, count].imshow(normalize_for_imshow(image), cmap=imshow_cmap)

    plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.001)
    return fig


def equalize_dataset(data, colorspace_conversion=None):
    clipLimit = 1.0
    grid_size = (2,2)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=grid_size)

    if colorspace == "BW":
        data_norm = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1), dtype=np.uint8)
    else:
        data_norm = np.copy(data)

    if colorspace_conversion == "RGB":
        print("no equalization for RGB images!")
        # no equalization for RGB images, it only makes sense for brightness or luminosity channels!

    elif colorspace_conversion == "YUV":
        for i, img in enumerate(data):
            data_norm[i] = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(data_norm[i])
            y = clahe.apply(y)
            data_norm[i] = np.reshape(cv2.merge((y,u,v)), data_norm.shape[1:])

    elif colorspace_conversion == "LAB":
        for i, img in enumerate(data):
            data_norm[i] = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(data_norm[i])
            l = clahe.apply(l)
            data_norm[i] = np.reshape(cv2.merge((l,a,b)), data_norm.shape[1:])

    elif colorspace_conversion=="BW":
        for i, img in enumerate(data):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data_norm[i] = np.reshape(clahe.apply(img),  data_norm.shape[1:])

        # update image_shape for one layer b/w image
        image_shape = data_norm.shape[1:]

    else:
        print("No colorspace conversion set!")
        data_norm[i] = clahe.apply(data[i])

    return data_norm


def center_normaize(data, mean, std):
    data = data.astype('float32')
    return (data - mean) / std




if __name__ == '__main__':

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test,  y_test  = test['features'],  test['labels']

    # check if we have equal amount of samples in the feature and label spaces
    assert(len(X_train) == len(y_train))
    assert(len(X_valid) == len(y_valid))
    assert(len(X_test) == len(y_test))

    n_train       = X_train.shape[0]
    n_validation  = X_valid.shape[0]
    n_test        = X_test.shape[0]
    image_shape   = X_train.shape[1:]
    n_classes     = np.unique(y_train).size
    n_total       = n_train + n_validation + n_test

    # Read in label descriptions
    labels = []
    with open(labels_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0].isdecimal():
                labels.append(row[1])

    # Check how many traffic sign example we have in each class within the training dataset
    class_indexes_train = []
    class_indexes_valid = []
    class_indexes_test  = []
    for i in range(n_classes):
        class_indexes_train.append(np.where(y_train==i))
        class_indexes_valid.append(np.where(y_valid==i))
        class_indexes_test.append(np.where(y_test==i))

    df = pd.DataFrame({
        'Traffic sign description': labels,
        'Training'   : [len(class_indexes_train[i][0]) for i in range(n_classes)],
        'Validation' : [len(class_indexes_valid[i][0]) for i in range(n_classes)],
        'Test'       : [len(class_indexes_test[i][0])  for i in range(n_classes)]
    })
    df_percent = pd.DataFrame({
        'Traffic sign description': labels,
        'Training'   : [len(class_indexes_train[i][0]) / n_train * 100       for i in range(n_classes)],
        'Validation' : [len(class_indexes_valid[i][0]) / n_validation * 100  for i in range(n_classes)],
        'Test'       : [len(class_indexes_test[i][0])  / n_test * 100        for i in range(n_classes)]
    })


    pd.options.display.width = 100
    pd.options.display.max_colwidth = 90
    df = df.reindex(columns=['Traffic sign description', 'Training', 'Validation', 'Test'])
    df_percent = df_percent.reindex(columns=['Traffic sign description', 'Training', 'Validation', 'Test'])
    print("Statistics Table: ")
    print(tabulate(df, headers="keys", tablefmt='pipe'))

    print("|   Dataset           |   # Samples  |   % of total amount of samples |")
    print("|--------------------:|-------------:|-------------------------------:|")
    print("| %20s |  %d  | %.2f%% |" % ("Training dataset", n_train, 100*n_train/n_total))
    print("| %20s |  %d  | %.2f%% |" % ("Validation dataset", n_validation, 100*n_validation/n_total))
    print("| %20s |  %d  | %.2f%% |" % ("Test dataset", n_test, 100*n_test/n_total))
    print("| %20s |  %d  |        |" % ("Total", n_total))

    fig1 = plt.figure()
    df_percent.plot.bar(width=0.8, alpha=0.4)
    plt.xlabel('Traffic sign classes')
    plt.ylabel('Samples in Dataset [%]')
    fig1.savefig('./examples/dataset_class_distribution_chart.png', bbox_inches='tight')
    plt.show(True)
    plt.close()
    
    plotlabels = 'Training', 'Validation', 'Test'
    sizes = [100*n_train/n_total, 100*n_validation/n_total, 100*n_test/n_total]
    explode = (0.1, 0.1, 0.1)
    fig2, ax1 = plt.subplots(figsize=(3,3))
    ax1.pie(sizes, explode=explode, labels=plotlabels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig2.savefig('./examples/dataset_distribution_chart.png', bbox_inches='tight')
    plt.show()
    plt.close()

    fig3 = visualize_dataset(X_train, class_indexes_train, imgs_per_class=5, classes=5, shuffle=True)
    fig3.savefig('./examples/dataset_example.png', bbox_inches='tight')
    plt.show()  
    plt.close()

    X_train_norm = equalize_dataset(X_train, colorspace)
    print("before normalization: min val: %f, max val: %f" % (np.min(X_train_norm), np.max(X_train_norm)))
    white_pixels_cnt = len(np.where(X_train == 255)[0])
    black_pixels_cnt = len(np.where(X_train == 0)[0])
    print("White & black pixels before equalization: %d, %d" % (white_pixels_cnt, black_pixels_cnt))
    white_pixels_cnt = len(np.where(X_train_norm == 255)[0])
    black_pixels_cnt = len(np.where(X_train_norm == 0)[0])
    print("White & black pixels after equalization: %d, %d" % (white_pixels_cnt, black_pixels_cnt))

    # Value normalization into -1 to +1 range
    X_mean = 128
    X_std = 128
    X_train_norm = center_normaize(X_train_norm, X_mean, X_std)
    print("normalized data with mean=%.3f and scale=%.3f" % (X_mean, X_std))

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
        zca_whitening=False,
        zca_epsilon=1e-7,
        shear_range=0,
        zoom_range=0.1,
        rescale=None
    )

    datagen.fit(X_train_norm, augment=True)

    X_train_batch, y_train_batch = datagen.flow(X_train_norm, y_train, batch_size=2500, shuffle=True).next()
    indexes  = []
    for i in range(n_classes):
        indexes.append(np.where(y_train_batch==i))

    print("X_train_batch.shape: ", X_train_batch.shape)
    visualize_dataset(X_train_batch, indexes, imgs_per_class=5, classes=5, shuffle=True)
