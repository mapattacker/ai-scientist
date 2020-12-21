# Artificial Neural Network

A perceptron is the simplest unit of a neural network. The input neurons first weights its inputs and then sums them up with a bias. An activation function is then applied, which produces the output for the neuron.

![](https://github.com/mapattacker/ai-scientist/blob/master/images/perceptron.png?raw=true)

A deep neural network simply represents a neural network with many hidden layers between the input and output layers. The architecture of the hidden layers can be very complex, like CNN and LSTM. 

![](https://github.com/mapattacker/ai-scientist/blob/master/images/ann.png?raw=true){: style=width:500px"}

## Activation Function

An activation function tells a perception what outcome it should be. The function differs for input/hidden layers with output layers.

For input/hidden layers, __ReLu__ (Rectified Linear units) is very popular compared to the now mostly obsolete sigmoid & tanh functions because it avoids the __vanishing gradient problem__ and has faster convergence. However, it is susceptible to dead neurons. So variants like Leaky ReLu, MaxOut and other functions are created to address this.

For the output layer, it depends on the type of learning we are trying to train.

| Output Type | Function |
|-|-|
| Binary Classification | `Sigmoid` |
| Multi-Class Classification | `Softmax` |
| Regression | `Linear` |



## Backpropagation

Training a model is about minimising the loss, and to do that, we want to move in the negative direction of the derivative. __Back-propagation__ is the process of calculating the derivatives. This is done by working backwards from the last layer to the first input layer.

__Gradient descent__ is the process of descending through the gradient, i.e. adjusting the parameters of the model to go down through the __loss function__.

__Learning Rate__ (lr) is the most important parameter to obtain the minimum loss. Too large a lr can cause the model to converge too quickly to a suboptimal solution, whereas a lr that is too small can cause the training process to take too long.

![](https://github.com/mapattacker/ai-scientist/blob/master/images/learning-rate.png?raw=true)

| Term | Desc |
|-|-|
| Optimizer | a learning algorithm, refers to the calculation of an error gradient or slope of error and “descent” refers to the moving down along that slope towards some minimum level of error. |
| Learning Rate | a hyperparameter from 0-1 to reach a minimum loss  |
| Batch Size | a hyperparameter of gradient descent that controls the number of training samples to work through before the model’s internal parameters are updated. | 
| Epoch | a hyperparameter of gradient descent that controls the number of complete passes through the training dataset. | 
| Loss Function | a function used to calculate the loss | 