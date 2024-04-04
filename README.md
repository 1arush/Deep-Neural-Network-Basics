# deep-neural-cat-classifier #
This repository is an application of deep neural networks, which is based on my learnings from DeepLearning.AI's course assessments. 

We aim to implement a deep neural network of 4 layers which is used for binary classificiation of images. Makes use of the most basic libraries such
as numpy, pandas, matplotlib, and PIL. The motivation for using a deep-net is that a single neuron with a sigmoid activation is unable to capture 
the more complex nature of the data, i.e. an image of a cat. Adding more layers allows the model to figure out more complex functions, which are 
exactly what we need in order to classify an image as cat (positive) or non-cat (negative).

This notebook uses multiple helper functions to implement forward propagation and backward propagation. These include functions for ReLU, sigmoid as 
well as their derivatives. To aid backpropagation, we store the hidden layer activations of each layer in a "cache", which may use up more memory
for larger models.

Also, it is important to note that it trains on 209 examples of labeled images, and performs sufficiently well on the test set.

Some notes:

-- classifies images as cat or non-cat

-- uses an image dataset in an h5 format
