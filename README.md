# Machine Learning Classification Problem
Objective: 
To use machine learning to implement and evaluate classification algorithms. 
Tasks: 
There are four tasks in this project: -
1)	Implement logistic regression, train it on the MNIST digit images and tune hyper parameters (Appendix 1).  
2)	Implement single hidden layer neural network, train it on the MNIST digit images and tune hyper parameters such as the number of units in the hidden layer (Appendix 2).  
3)	Use a publicly available convolutional neural network package, train it on the MNIST digit images and tune hyper parameters (Appendix 3).  
4)	Test your MNIST trained models on USPS test data and compare the performance with that of the MNIST data.  
Datasets:
1)	MNIST Data: The database contains 60,000 training images and 10,000 testing images. For both training and testing of our classifiers, we will use the MNIST dataset. 
2)	USPS Data: Each digit has 2000 samples available for testing. These are segmented images scanned at a resolution of 100ppi and cropped. We use USPS handwritten digit as another testing data for this project to test whether our models could be generalized to a new population of data. 
Theory:

Logistic Regression:
Logistic regression is a machine learning model for binary classification, i.e. learning to classify data points into one of two categories. It's a linear model, in that the decision depends only on the dot product of a weight vector with a feature vector. This means the classification boundary can be represented as a hyperplane. It's a widely used model in its own right, and the general structure of linear-followed-by-sigmoid is a common motif in neural networks. Softmax regression is a method in machine learning which allows for the classification of an input into discrete classes. Unlike the commonly used logistic regression, which can only perform binary classifications, softmax allows for classification into any number of possible classes

Single Layer Neural Network:
The most common structure of connecting neurons into a network is by layers. The simplest form of layered network is shown in the figure. The shaded nodes on the left are in the so-called input layer. The input layer neurons are to only pass and distribute the inputs and perform no computation. Thus, the only true layer of neurons is the one on the right. Each of the inputs  is connected to every artificial neuron in the output layer through the connection weight. Since every value of outputs  is calculated from the same set of input values, each output is varied based on the connection weights.


Convolutional Neural Network:
We use a open source package tensorflow for solving this. CNNs apply a series of filters to the raw pixel data of an image to extract and learn higher-level features, which the model can then use for classification. CNNs contains three components:
•	Convolutional layers, which apply a specified number of convolution filters to the image. For each subregion, the layer performs a set of mathematical operations to produce a single value in the output feature map. Convolutional layers then typically apply a ReLU activation function to the output to introduce nonlinearities into the model.
•	Pooling layers, which downsample the image data extracted by the convolutional layers to reduce the dimensionality of the feature map in order to decrease processing time. A commonly used pooling algorithm is max pooling, which extracts subregions of the feature map (e.g., 2x2-pixel tiles), keeps their maximum value, and discards all other values.
•	Dense (fully connected) layers, which perform classification on the features extracted by the convolutional layers and downsampled by the pooling layers. In a dense layer, every node in the layer is connected to every node in the preceding layer.
Mini-Batch Stochastic Gradient Descent:
Mini-batch gradient descent is a variation of the gradient descent algorithm that splits the training dataset into small batches that are used to calculate model error and update model coefficients.
Mini-batch gradient descent seeks to find a balance between the robustness of stochastic gradient descent and the efficiency of batch gradient descent. It is the most common implementation of gradient descent used in the field of deep learning.
The strength of mini-batch SGD compared to SGD is that the computation of ∑mi=1 ∇wE (zi) can usually be performed using matrix operation and thus largely out-performs the speed of computing ∇wE (zi) individually and updating w sequentially. However, within same computing time, mini-batch SGD updates the weights much more often than batch gradient descent, which gives mini-batch SGD faster converging speed. The choice of mini-batch size m is the tradeoff of the two effects. 
Implementation:
First, we import the MNIST data files we are going to be classifying. This database contains images of thousands of handwritten digits, and their proper labels. 

We initialize our weights, regularization factor, number of iterations, and learning rate. We then loop over a computation of the loss and gradient, and application of gradient. We perform the softmax transformation. Softmax(z) method allows us to get probabilities for each class score that sum to 100%.

We, then, determine the probabilities and predictions for each class when given a set of input data using getAccuracy method. We can see how the loss decreases over time by constructing a graph.
We construct Single layer neural network using tensorflow, an open source package in python. We construst a hidden layer and try calculating the accuracy using getAccuracy method.
We use mini batch size gradient. Instead of randomly sampling z1 , z2 , ..., zm from the training data each time, the normal practice is we randomly shuffle the training set x1,...,xN , partition it into mini-batches of size m and feed the chunks sequentially to the mini-batch SGD. We loop over all training mini-batches until the training converges. 
We construct CNN using following layers:
1.	Convolutional Layer #1
2.	Pooling Layer #1
3.	Convolutional Layer #2
4.	Pooling Layer #2
5.	Dense Layer #1
6.	Dense Layer #2 (Logits Layer) 
Then, using USPS data, we calculate the test accuracy for each model.

Here, we have taken the following hyperparams:
For logistic regression:
K = 10
iterations = 100
learningRate = 0.00001
lam = 0.5852
batch_size = 500
input_size = 784 
number_of_neurons = 200 
output_size = 10 
For SNN:
batch_size_nn=500 
learning_rate_nn=0.01
iterations_nn = 100
For CNN:
batch_size = 500
iterations = 100


Graphs:

Losses over time in logistic regression:

Results:

Accuracy for logistic regression :
MNIST Training Accuracy 0.8945272727272727
MNIST Test Accuracy 0.9009
MNIST Validation Accuracy 0.901
USPS Test Accuracy 0.10035501775088755

 
 Accuracy for SNN :
MNIST test accuracy : 
[0.97420001]
USPS test accuracy : 
[0.46097305]

 
 Accuracy for CNN :
MNIST test accuracy : 0.9168
USPS test accuracy  :  0.414721
Conclusion:

1.	We have successfully implemented logistic regression, trained it on the MNIST digit images and tuned hyperparameters.  
2.	We have successfully implemented single hidden layer neural network, trained it on the MNIST digit images and tuned hyperparameters.
3.	We have used a convolutional neural network package tensorflow, trained it on the MNIST digit images and tuned hyperparameters 
4.	We have tested  our MNIST trained models on USPS test data and compared the performance with that of the MNIST data. 
Yes, our findings support the No Free Lunch theorem. The No Free Lunch theorem states that any two optimization algorithms are equivalent when their performance is averaged across all possible problems. For example, when comparing two algorithms, one may provide a better performance for a specific problem, and the second algorithm may provide better results for another specific problem, but in the overall average, the algorithms will provide a more or less similar optimization result. Our experiment with the USPS and MNIST Datasets, successfully demonstrates the No Free Lunch Theorem which can be seen from a comparison of above results and graphs.
