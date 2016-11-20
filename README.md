# NeuralNet---Java
Custom Java classes that can be used to easily build a feedforward-style NN of arbitrary size. The NN learns through training examples and uses backpropagation with stochastic gradient descent to tune its interior weights.

The entry class "OCR" is an application I made to perform optical character recognition on the MNIST data set using an instance of my NeuralNetwork class. The specific parameters I used are shown in the first lines of this program: 60,000 members in the training set, 100 epochs, and 40 hidden layers. Using my NeuralNetwork, I achieved a 96.5% accuracy rate among 10,000 test cases. 

In order to make a working neural net for your own application, create an instance of the NeuralNetwork class using its qualified constructor to specify the number of inputs, hidden layer neurons, output layer neurons, and the learning rate. Interface with the NN using arrays of doubles representing the input and output values. Dimensionality of the inputs/outputs must match the inputs/outputs of the net, obviously.

FUTURE IMPROVEMENTS
1) Allow for more than one hidden layer. This shouldn't be difficult since the Layer class encapsulates a layer's inner functionality. It would mostly be a matter of making the forward/backpropagation methods iterative instead of assuming a set size.
2) Ensure that it can handle invalid parameters passed into its interface methods. Right now it assumes the client application will always do this correctly.
3) Add capability to make the NN convolutional so it would be better suited for vision applications. There are a few ways to approach this, but probably the simplest way would be a CNN class that extends NeuralNetwork which contains a kernel or list of kernels, and which alters the input that ultimately goes into the underlying NN. This would be severely limited compared to the more sophisticated CNN's out there, but I think it would still deliver noticeable improvement.
