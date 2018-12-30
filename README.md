#Convolution neural network for classification

Objective: 
To implement a convolution neural network to determine whether the person in a portrait is wearing glasses or not.

Tasks: 
 
1) Import the images from Celeba and corresponding labels.
2) Divide the imported data into training and testing data. 
3) Reduce the size of the images.
4) Use a publicly available convolutional neural network package, train it on the Celeba images and tune hyper parameters (Appendix 3). 
5) Test your Celeba trained model on test data. 

Datasets:
1)	Celeba Data: The database contains more than 200K images. For both training and testing of our classifiers, we will use the Celeba dataset. Take 80 percent of the data as training data and rest 20 percent as testing data. Format of each image is .jpg.
2)	Label Data: This is a text file which contains labels corresponding to each image. The text file has many columns. The only relevant columns are column containing name of the image and Eyeglasses column. Eyeglasses column contains two values : [-1, 1].
Theory:

Convolutional Neural Network:
We use a open source package tensorflow for solving this. CNNs apply a series of filters to the raw pixel data of an image to extract and learn higher-level features, which the model can then use for classification. CNNs contains three components:
•	Convolutional layers, which apply a specified number of convolution filters to the image. For each sub region, the layer performs a set of mathematical operations to produce a single value in the output feature map. Convolutional layers then typically apply a ReLU activation function to the output to introduce nonlinearities into the model.
•	Pooling layers, which downsample the image data extracted by the convolutional layers to reduce the dimensionality of the feature map in order to decrease processing time. A commonly used pooling algorithm is max pooling, which extracts subregions of the feature map (e.g., 2x2-pixel tiles), keeps their maximum value, and discards all other values.
•	Dense (fully connected) layers, which perform classification on the features extracted by the convolutional layers and down sampled by the pooling layers. In a dense layer, every node in the layer is connected to every node in the preceding layer.
Mini-Batch Stochastic Gradient Descent:
•	Mini-batch gradient descent is a variation of the gradient descent algorithm that splits the training dataset into small batches that are used to calculate model error and update model coefficients.
•	Mini-batch gradient descent seeks to find a balance between the robustness of stochastic gradient descent and the efficiency of batch gradient descent. It is the most common implementation of gradient descent used in the field of deep learning.
•	The strength of mini-batch SGD compared to SGD is that the computation of ∑mi=1 ∇wE (zi) can usually be performed using matrix operation and thus largely out-performs the speed of computing ∇wE (zi) individually and updating w sequentially. However, within same computing time, mini-batch SGD updates the weights much more often than batch gradient descent, which gives mini-batch SGD faster converging speed. The choice of mini-batch size m is the tradeoff of the two effects. 
Implementation:
First, we import the Celeba data files we are going to be classifying. This database contains images of thousands of celebrities. We will also import labels corresponding to each image. 

We use mini batch size gradient. Instead of randomly sampling z1 , z2 , ..., zm from the training data each time, the normal practice is we randomly shuffle the training set x1,...,xN , partition it into mini-batches of size m and feed the chunks sequentially to the mini-batch SGD. We loop over all training mini-batches until the training converges. 
We construct CNN using following layers:
1.	Convolutional Layer #1
2.	Pooling Layer #1
3.	Convolutional Layer #2
4.	Pooling Layer #2
5.	Dense Layer #1
6.	Dense Layer #2 (Logits Layer) 
Then, using testing data, which is 20 percent of Celeba data, we calculate the test accuracy for each model.


