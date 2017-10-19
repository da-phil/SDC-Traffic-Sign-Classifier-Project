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
        

def visualize_dataset(data, indexes, imgs_per_class=10, classes=43):
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

    fig, axarray = plt.subplots(classes, imgs_per_class, sharex=True, sharey=True, figsize=(8,4))
    for sign_class in range(classes):
        print("Showing example images from class %d: %s" % (sign_class, labels[sign_class]))
        axarray[sign_class, 0].set_title(labels[sign_class], fontdict={'fontsize': 14})
        rand_items = random.sample(list(indexes[sign_class][0]), imgs_per_class)
        for count, item in enumerate(rand_items):
            image = data_conv[item].squeeze()
            axarray[sign_class, count].set_axis_off()
            axarray[sign_class, count].imshow(normalize_for_imshow(image), cmap=imshow_cmap)

    return fig


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

    pd.options.display.width = 100
    pd.options.display.max_colwidth = 90
    df = df.reindex(columns=['Traffic sign description', 'Training', 'Validation', 'Test'])
    print("Statistics Table: ")
    print(tabulate(df, headers="keys", tablefmt='pipe'))

    print("|   Dataset           |   # Samples  |   % of total amount of samples |")
    print("|--------------------:|-------------:|-------------------------------:|")
    print("| %20s |  %d  | %.2f%% |" % ("Training dataset", n_train, 100*n_train/n_total))
    print("| %20s |  %d  | %.2f%% |" % ("Validation dataset", n_validation, 100*n_validation/n_total))
    print("| %20s |  %d  | %.2f%% |" % ("Test dataset", n_test, 100*n_test/n_total))
    print("| %20s |  %d  |        |" % ("Total", n_total))


    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    plotlabels = 'Training', 'Validation', 'Test'
    sizes = [100*n_train/n_total, 100*n_validation/n_total, 100*n_test/n_total]
    explode = (0.1, 0.1, 0.1)
    fig1, ax1 = plt.subplots(figsize=(3,3))
    ax1.pie(sizes, explode=explode, labels=plotlabels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig1.savefig('./examples/dataset_distribution_chart.png', bbox_inches='tight')
    plt.show()  
    plt.close()


    fig2 = visualize_dataset(X_train, class_indexes_train, imgs_per_class=10, classes=3)
    fig2.savefig('./examples/dataset_example.png', bbox_inches='tight')
    plt.show()  
    plt.close()
