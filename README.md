# NeuralNetPlayground

[![MATLAB FEX](https://img.shields.io/badge/MATLAB%20FEX-Click%20Here-green.svg)](http://www.mathworks.com/matlabcentral/fileexchange/57610-a-matlab-recreation-of-the-tensorflow-neural-network-playground)
![Minimum Version](https://img.shields.io/badge/Requires-R2009b-blue.svg)

A MATLAB implementation of the TensorFlow Neural Networks Playground seen on 
[http://playground.tensorflow.org/](http://playground.tensorflow.org/)

## Authors

#### Front-End Graphical User Interface

* [Amro](https://github.com/amroamroamro)

#### Training Algorithm and Engine

* [Ray Phan](https://github.com/rayryeng)

#### Debugging and Testing

* [Jonathan Suever](https://github.com/suever)
* [Qi Wang](https://github.com/GameOfThrow)

## Description

Inspired by the TensorFlow Neural Networks Playground interface readily available online at [http://playground.tensorflow.org/](http://playground.tensorflow.org/), this is a MATLAB implementation of the same Neural Network interface for using Artificial Neural Networks for regression and classification of highly non-linear data.  The interface uses the HG1 graphics system in order to be compatible with older versions of MATLAB.  A secondary purpose of this project is to write a vectorized implementation of training Artificial Neural Networks with Stochastic Gradient Descent as a means of education and to demonstrate the power of MATLAB and matrices.

The goal for this framework is given randomly generated training and test data that fall into two classes that conform to certain shapes or specifications, and given the configuration of a neural network, the goal is to perform either regression or binary classification of this data and interactively show the results to the user, specifically a classification or regression map of the data, as well as numerical performance measures such as the training and test loss and their values plotted on a performance curve over each iteration.  The architecture of the neural network is highly configurable so the results for each change in the architecture can be seen immediately.

There are two files that accompany this repo:

1. `NeuralNetApp.m`: The GUI that creates the interface as seen on TensorFlow Neural Networks Playground but is done completely with MATLAB GUI elements and widgets.  
2. `NeuralNet2.m`: The class that performs the Neural Network training via Stochastic Gradient Descent.  This is used in `NeuralNetApp.m`


## Compatible Versions

Debugged and tested for MATLAB R2009b or newer.

This code can only be run on versions from R2009b and onwards due to the syntax for discarding output variables from functions via (`~`).  If you wish to use this code for older versions (without guaranteeing compatibility), you will need to replace all instances of discarding output variables with dummy variables but you'll be subject to a variety of `mlint` errors.  This effort has not been done on our part as there is very little gain to go to even older versions and so if you desire to run this code on older versions, you will have to do so yourself.


## Running the Program

Ensure that both files `NeuralNetApp.m` and `NeuralNet2.m` are in the same directory.  In the MATLAB Command Window, simply run the `NeuralNetApp.m` file within this directory. Assuming you are working in the directory of where you stored, type in the following and press ENTER:

    >> NeuralNetApp
    
If you want to be explicit, you can use `run` and provide the path to where this file is located on your system:

    >> run(fullfile('path', 'to', 'the', 'NeuralNetApp.m'));
    
If all goes well, you should be presented with a GUI.

## Overview of the GUI

As seen on the TensorFlow Neural Network playground, below is a snapshot of what is produced if the program runs successfully on your system:

<img src = "http://i.stack.imgur.com/zgike.png" style width="100%">

There are several widgets and areas that are of primary concern which we will go through now.

---

### Data

The data section is highlighted below:

<img src = "http://i.stack.imgur.com/cTx9U.png" style width="100%">

Each widget will be discussed below, specifically how you would use each of these widgets:

#### Which dataset do you want to use? 

This is a dropdown menu that allows you to choose which dataset you would like the Neural Network to work on, whether you are using regression or binary classification.  The data are two-dimensional and are generated randomly within a dynamic range of `[-6, +6]` for both dimensions.  A 2D point that is considered positive belongs to one label and values that are negative belong to another.  Points that are orange denote negative labels and points that are blue denote positive labels.  

You may choose from one of four possible datasets:

##### Circle

A dataset where one set of points belonging to one label are within some radius of a circle and the other set of points belong to a label within a larger radius but do not intersect with any points in the smaller radius of points.  Below is such an example:
    
<img src = "http://i.stack.imgur.com/y269P.png" style width="50%">

    
##### XOR

A dataset that is set up in a criss-cross fashion where positive labels belong along one diagonal and negative labels belong to another.  Below is such an example:

<img src = "http://i.stack.imgur.com/liNT5.png" style width="50%">

##### Gaussian

A dataset that is a bit easier where each group of labels belongs to a cluster that is Gaussian distributed.  Below is such an example:

<img src = "http://i.stack.imgur.com/9xlBe.png" style width="50%">

##### Spiral

A challenging dataset.  The name speaks for itself.  You can see below on what an example looks like:

<img src = "http://i.stack.imgur.com/vmOZK.png" style width="50%">

#### Ratio of training to test data

This is a slider widget that asks you how much percentage of your data should be training and the rest being test data.  500 data points are always generated and the training and test data is split according to the value in this slider.  Once the data is generated, a random permutation of points are placed in both sets for use in this tool.  The default decomposition is 50%, so half of the data are training and the other are testing.

#### Noise

You can add uniformly distributed random noise to the data to make the neural network training more challenging.  This noise is defined as a percentage, and so a value of 10 means 10% or a value of 0.1 will be appended on top of the original data.  A value of 0 gives you clean data.  The default value is 10% or 0.1 here.

#### Batch Size

The technique for training the neural network in this tool is to perform Stochastic Gradient Descent.  At each iteration or epoch, randomly sampled training samples of a specified batch size are selected and the neural network gets updated so that regression or classification of these points will be more accurate.  This batch size can be specified here and the default is 10.

---

### Inputs

The inputs section is highlighted below:

<img src = "http://i.stack.imgur.com/UAIXa.png" style width="100%">

This section controls what features you want to use to perform regression or classification of the data where each feature is an input neuron in the input layer. Therefore, you are configuring the input layer in this section.  The data consist of two dimensional data, where `X1` is the horizontal axis and `X2` is the vertical.  There are a variety of features to choose from. Simply click and un-click which features you want to use.   Each feature is accompanied by a preview image of how the negative and positive labels would be classified in a two-dimensional grid of coordinates if you were to choose that feature that are right beside the feature name itself.  Values of white in each preview box are the classification boundary, meaning that this is where the feature would get mapped to 0.  Features that are selected will have a gray shadow surrounding that preview box once you bring focus away from that box.

The available features are the following:

- `X1`: The feature `X1` itself which is just using the first dimension of the data
- `X2`: The feature `X2` itself which is just using the second dimension of the data
- `X1^2`: The feature `X1` but squared
- `X2^2`: The feature `X2` but squared
- `sin(X1)`: The feature `X1` with the `sin` operation applied to it
- `sin(X2)`: The feature `X2` with the `sin` operation applied to it

### Parameters

The Parameters section is shown below:

<img src = "http://i.stack.imgur.com/o5MSO.png" style width="100%">

This section allows you to customize how exactly the neural network should be trained.  There are several dropdown menus that allow you to perform this customization, which are the following:

#### Learning Rate

This stems from Stochastic Gradient Descent and so the learning rate is the amount you want to move forward towards the right solution. This dropdown menu allows you to choose between one of several learning rates. Setting a value too large may make the training oscillate or even diverge and making the value too small may take a long time to allow the training to converge.  Experiment with this value for different datasets.  The default is 0.03.

#### Activation Function

The hidden layers in the neural network all have the same activation function, and you can choose between one of the following activation functions:

- `Tanh`: The hyperbolic tangent - the default
- `Sigmoid`: The sigmoidal function
- `ReLU`: The rectified linear unit
- `Linear`: The linear (or identity) function (i.e. output = input). 

The first three are used primarily for classification while the last one (linear) is usually used in regression problems.

#### Regularization

This is a mechanism to prevent overfitting and to allow your neural network to generalize to new inputs.  The options you have available are:

- `None`: No regularization - the default
- `L1`: L1 regularization using the technique outlined in [Tsuruoka *et al.*'s work](http://aclweb.org/anthology/P/P09/P09-1054.pdf).
- `L2`: L2 regularization

#### Regularization Rate

The strength of the regularization to apply.  This dropdown menu allows you to choose between one of several regularization rates.  Setting a value too large will produce a neural network that has high bias and underfits the data. Setting a value to be too small, or even 0 risks the data to be overfit or has high variance.  Start with a small value, then start increasing the value slowly to see what the effects are.

#### Problem Type

You can choose between classification or regression as the problem to solve, which is what the options in this dropdown menu give you.

---


### Hidden Layers

The hidden layers section is highlighted below:

<img src = "http://i.stack.imgur.com/SGIZ3.png" style width="100%">

This section allows you to customize how many hidden layers there are in this neural network and you are allowed up to five.  You can also choose how many neurons go into each layer and you are allowed up to eight.  The activation function for each neuron in the hidden layers are assumed to be all the same and you can customize what this is as noted previously.

The top of this section allows you to adjust how many hidden layers there are. Simply push the `+` or `-` buttons to add or remove the number of hidden layers in the neural network.  Each hidden layer is customizable in the number of neurons in that layer. You will see that in each layer, there is a pair of `+` and `-` buttons that allow you to add or remove neurons in each layer.

For each hidden neuron that is visualized, you can see the intermediate outputs in a preview image which demonstrate how the negative and positive labels would be classified in a two-dimensional grid of coordinates for that particular neuron.

---

### Run

Of course once you set everything up, you'll want to actually run the training.  That's what the Run section is for and it is shown below highlighted:

<img src = "http://i.stack.imgur.com/8MApm.png" style width="100%">

There are three buttons here that you can click on:

- `Run`: This starts the neural network training.  As you run the training stage, the number of iterations that Stochastic Gradient Descent has made will be shown in an Iterations counter to the right of the three buttons.  At any time, you can **pause** the training by clicking on the `Run` button again, where the button's text changes to `Pause`.  You may also use this opportunity to change the configuration of the neural network and run the training using this as a starting point so you can interactively see how changing the neural network structure affects the output.
- `Step`: Choosing `Run` allows the training to run automatically. You can also choose to **step** through each iteration by clicking on this button so that one iteration passes per click.  Clicking on this while the `Run` option is enabled will automatically pause the training and allows you to step through each iteration interactively.
- `Reset`: This resets the neural network so you are starting with a fresh new one that requires to be trained. The number of iterations is reset as well as all of the weights in the neural network itself.

---

### Outputs

Once you `Run` the training, you can interactively see how the neural network gets trained over the iterations.  This is seen in the Outputs section and is shown below:

<img src = "http://i.stack.imgur.com/p0eVt.png" style width="100%">

This pane interactively shows you the positive and negative training and test data in a graph as blue and orange respectively.  There is a checkbox labeled *Show test data* that shows you what the test data was inside this pane as darker and thicker points. Also, the colour scheme of the background of this graph is for determining the outputs for new values presented to the neural network that it has not seen before.  This data is uniformly spaced between `[-6,+6]` for both dimensions and the colour tells you how each point in the grid was classified as.  Orange values denote negative values and blue values denote positive values.  White values denote the decision boundary where the neural network output is 0.  Therefore, the desired colouring scheme should cluster all blue points to be within a region of blue and all orange points to be within a region of orange.

There are also performance curves displayed at the top of this graph that plot the training and test loss at each iteration of training.  The loss is defined as the sum of squared differences between the predicted and actual outputs for both the training and test data at a particular iteration.  The training loss curve is seen in light gray while the test loss curve is seen in dark grey.  There are also numerical measures that are displayed above the performance curves that give you the training and test loss for the current iteration.

You may also *Discretize output* which sets a threshold on the activation function to give you a hard decision on whether the input is positive or negative, rather than having a range of positive and negative values.

---

## Sample Run

Here is a sample run of what happens under the following settings:

#### Data

* Dataset: Circle
* Ratio of training to test data: 50%
* Noise: 0
* Batch Size: 10

#### Inputs

* Features: `X1` and `X2`

#### Hidden Layers

* 2 Hidden Layers
* First Layer: 4 neurons
* Second Layer: 2 neurons

#### Parameters
* Learning Rate: 0.3
* Activation: `Tanh`
* Regularization: None
* Regularization Rate: 0
* Problem Type: Classification

The neural network ran for 215 iterations, and we get the following window:

<img src = "http://i.stack.imgur.com/YQdrm.png" style width="100%">

We see that the positive data is clustered in a region of blue while the negative data is clustered in a region of orange.  This means the neural network successfully classified most (if not all) of the data points.

---

## NeuralNet2.m

The main engine before the training algorithm is seen in the `NeuralNet2.m` file.  This is a custom class that was written and is well documented to allow a MATLAB user to use it for their purposes in future code that they write.  You can type in `help NeuralNet2` in the command window where this file is located on your system for a comprehensive overview on how to use this class.

---

## Training Tips

If you have tried to configure the neural network and can't seem to get the neural network to converge and classify your data with high accuracy, here are some tips you can try. :

1. Introduce regularization: The data may be overfitting and so try to introduce regularization at a small rate, then gradually increase until you're satisfied.
2. Decrease the learning rate: If you see that the training doesn't converge and oscillates or even diverges, try decreasing the learning rate.
3. Increase the learning rate: If you see that training is very slow and the desired output is slowly getting to where you want it to go, try increasing the learning rate.
4. Change the activation function: If you are performing regression, using `Tanh`, `Sigmoid` or `ReLu` for the activation function will not give good results.  Use a linear activation function instead. The same can be said for classification where linear is not good in that area.  Ensure you use `Tanh`, `Sigmoid` or `ReLu` when perform classification.
5. Changing the number of hidden layers: If the dataset is rather difficult to train, try increasing the number of hidden layers.
6. Changing the number of hidden neurons per hidden layer: A general rule is set the same number of neurons for each layer in the hidden layer, but the general rule is to let the first couple of layers have more neurons than the other layers should decide to vary this.