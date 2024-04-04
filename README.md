# deep-neural-cat-classifier #
This repository is an application of deep neural networks, which is based on my learnings from DeepLearning.AI's course assessments. 

We aim to implement a deep neural network of 4 layers which is used for binary classificiation of images. Makes use of the most basic libraries such
as numpy, pandas, matplotlib, and PIL. The motivation for using a deep-net is that a single neuron with a sigmoid activation is unable to capture 
the more complex nature of the data, i.e. an image of a cat. Adding more layers allows the model to figure out more complex functions, which are 
exactly what we need in order to classify an image as cat (positive) or non-cat (negative).

<img width="631" alt="2layerNN_kiank" src="https://github.com/1arush/deep-neural-network-basics/assets/105356056/063eacee-ae4f-477f-859a-a1f84989e4e5">

This notebook uses multiple helper functions to implement forward propagation and backward propagation. These include functions for ReLU, sigmoid as 
well as their derivatives. The derivatives can be thought of by using a computation graph, which helps understand the propagation of the gradients. To aid backpropagation, we store the hidden layer activations of each layer in a "cache", which may use up more memory for larger models.

Also, it is important to note that it trains on 209 examples of labeled images, and performs sufficiently well on the test set with an accuracy of 0.8.

Some notes:

-- classifies images as cat or non-cat

-- uses standard gradient descent without regularization and rmsprop

-- uses an image dataset in an h5 format


