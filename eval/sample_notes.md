# Chapter 2: Optimization Basics

## Gradient Descent

Gradient descent is a first-order iterative optimization algorithm for finding a
local minimum of a differentiable function. At each step, you move the parameters
in the direction opposite the gradient of the loss with respect to the parameters.

The update rule is:

    theta_{t+1} = theta_t - alpha * grad_theta L(theta_t)

where alpha is the learning rate. In this course we use a default learning rate
of 0.01 unless otherwise stated.

## The Chain Rule

The chain rule is the mechanism that makes backpropagation work. For a composition
f(g(x)), the derivative is f'(g(x)) * g'(x). In neural networks this lets us
compute gradients of the final loss with respect to any earlier weight by
multiplying local derivatives along the computation graph.

## Backpropagation

Backpropagation is the application of the chain rule to compute gradients
efficiently in a neural network. It has two passes:

1. A forward pass that computes activations at every layer and stores them.
2. A backward pass that walks from the output loss back toward the input,
   multiplying local Jacobians, to produce a gradient for every parameter.

## Loss Functions

The main loss function covered in this course is the cross-entropy loss for
classification and mean squared error for regression. Cross-entropy penalizes
confident wrong predictions heavily and is the standard choice when the output
is a probability distribution over classes.
