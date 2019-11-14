import sklearn
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import keras
from keras.datasets import cifar10, mnist
from skimage.io import imread
import cv2
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Model
from keras import optimizers
from keras.applications import inception_v3
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import sklearn.cluster




# set the training and testing datasets from either the cifar10 or mnist datasets from keras
DATASET='cifar10'

if DATASET is 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
else:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()


# the mnist dataset requires additional shaping so that our NN can be used.  This is done below.
if DATASET is 'mnist':
    x_train = np.stack([x_train, x_train, x_train], axis=-1)
    x_test = np.stack([x_test, x_test, x_test], axis=-1)

    
# simple prints to check the counts and dimensions of our images
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# set our image size to 150 x 150; use cv2.resize to resize to this specification
IMAGE_SIZE = 150

x_train_newsize = []
for i, file in enumerate(x_train):
    lowresim = cv2.resize(x_train[i], dsize=(IMAGE_SIZE,IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    x_train_newsize.append(lowresim)
    
x_test_newsize = []
for i, file in enumerate(x_test):
    lowresim = cv2.resize(x_test[i], dsize=(IMAGE_SIZE,IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    x_test_newsize.append(lowresim)


# print out the resized image to verify it's new size
plt.imshow(x_test_newsize[0])


# to reduce processing time, let's use a subset of 1,000 images
N_IMAGES = 1000

x_test_100 = []
for i in range(0, N_IMAGES):
    x_test_100.append(x_test_newsize[i])
    
    
y_test_100 = []
for i in range(0, N_IMAGES):
    y_test_100.append(y_test[i])
    
# test the length to ensure resizing is correct
print(len(x_test_100))
print(len(y_test_100))


# our NN works best if each rgb/pixel is described by numbers between 0-1; below resizes accordingly
x_test_100 = np.divide(x_test_100, 255.0)


# after changing rgb colors to 0-1, the pictures should be nearly black
plt.imshow(x_test_100[0])


# Initialize InceptionV3 image recognition model, and use the weights from training via imagenet dataset
inception = inception_v3.InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# Setup our Model
x = Flatten()(inception.output)
model = Model(input=inception.input, output=x)


# Run the images through the neural network
embeddings = model.predict(x_test_100)

print(embeddings.shape)
print(embeddings[0])


# using principal component analysis we narrow down to the most important components
pca = PCA(n_components=200)
pca.fit(embeddings)
reduced_embeddings = normalize(pca.transform(embeddings))

#from sklearn.decomposition import FastICA
#ica = FastICA(n_components=20)
#reduced_embeddings = normalize(ica.fit_transform(embeddings))

# using our reduced dataset, we can utilize kmeans clustering to cluster our dataset which is based on our images
kmeans = sklearn.cluster.KMeans(n_clusters=10)
kmeans.fit(reduced_embeddings)
cluster_assignments = kmeans.predict(reduced_embeddings)

#X = np.array(reduced_embeddings)
#sklearn.cluster.KMeans().fit(X)

# Looking at ICA
#from sklearn.decomposition import FastICA
#transformed = FastICA(n_components=3, random_state=0)
#transformedimages = transformed.fit_transform(X)
#transformedimages.shape
#print(transformedimages[1])

#print(centers.cluster_centers_)
#assignedpoints = kmeans.predict(norm)

final_image_check = []
for i in range(0, N_IMAGES):
    final_image_check.append(x_test_newsize[i])

final_check_nums = []
for i in range(0, N_IMAGES):
    final_check_nums.append(y_test_100[i])


list_score_key = list(zip(final_check_nums, cluster_assignments, final_image_check))


cluster_to_category = {}

for item in list_score_key:
    if item[1] not in cluster_to_category:
        cluster_to_category[item[1]] = list()
    cluster_to_category[item[1]].append({'class': item[0], 'image': item[2]})

# setup visualization method
def visualize_category(category_details):
    w=150
    h=150
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    
    category_details = np.random.choice(category_details, columns*rows+1, replace=False)
    
    for i in range(1, columns*rows+1):
        img = category_details[i]['image']
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


# call the batch and visualize
visualize_category(cluster_to_category[7])

