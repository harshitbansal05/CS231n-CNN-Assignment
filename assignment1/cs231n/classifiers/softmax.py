import math
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    loss -= math.log(math.exp(correct_class_score) / np.sum(np.exp(scores)))
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:,j] -= X[i,:] - (math.exp(correct_class_score) / np.sum(np.exp(scores))) * X[i,:]
        continue
      dW[:,j] += (math.exp(scores[j]) / np.sum(np.exp(scores))) * X[i,:]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(np.multiply(W, W))
  dW[:,:] += 2 * reg * W[:,:]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.matmul(X, W)
  scores -= np.max(scores)
  correct_class_scores = np.exp(np.choose(y, scores.transpose()))
  exp_scores = np.sum(np.exp(scores), axis = 1) 
  loss = -np.sum(np.log(np.divide(correct_class_scores, exp_scores)))
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(np.multiply(W, W))

  softmax_matrix = np.exp(scores) / exp_scores[:,None]

  row_indices = W.shape[1] * np.arange(num_train)
  add_factor = np.add(row_indices, y)
  replace_factor = np.ones(num_train)
  sub_matrix = np.zeros(softmax_matrix.shape)
  np.put(sub_matrix, add_factor, replace_factor)

  dW = np.matmul(X.transpose(), softmax_matrix - sub_matrix)
  dW /= num_train

  dW[:,:] += 2 * reg * W[:,:]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

