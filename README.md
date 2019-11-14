# Neural-Network
Neural Network to Clustering

The following code leverages a Neural Network (Inception V3) to classify images using vectors, then runs these image vectors through a Kmeans clustering.

The goal of this exercise is to see how good the NN + K-Means duo is in batching and identifying pictures.

Image Dataset:  cifar10, minist; we could import any other dataset if needed

Neural Network:  InceptionV3.  This is trained using the imagenet training set

Clustering Algorithm:  K-Means. Using batches of 200

Results:
- Cifar10:  the results for cifar10 was mediocre.  We can see that the algorithm did a good job batching trucks together, specifically if the color red was incorporated.  Overall the success of this batch was ~80%, meaning 80% of the pictures were accurately batched.
- Minist:  despite this dataset subjectively being easier to recognize, the algorithm had trouble batching accordingly.  It was 100% accurate when batching the number 1, but often had difficulties with other batches.

Suggested improvements:
We lost massive picture quality when resizing our image datasets to be compatible with our Neural Network (150x150 requirement).  If we could improve this process I suspect our accuracy would increase.
