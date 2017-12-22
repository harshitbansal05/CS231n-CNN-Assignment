import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape, dtype = np.float64) # initialize the gradient as zero
  dW_one = np.zeros(W.shape)
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:] / num_train
        dW[:,y[i]] -= X[i,:] /  num_train
        dW_one[:,j] += X[i,:]
        dW_one[:,y[i]] -= X[i,:]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(np.multiply(W, W))
  dW[:,:] += 2 * reg * W[:,:]

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape, dtype = np.float64) # initialize the gradient as zero

  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  all_scores = np.matmul(X, W)
  correct_label_score = np.choose(y, all_scores.transpose())
  scores_one_matrix = all_scores + np.ones(all_scores.shape)
  loss_matrix = np.maximum((scores_one_matrix.transpose() - correct_label_score).transpose(), np.zeros(all_scores.shape))
  loss = (np.sum(loss_matrix) - num_train)/num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  loss_matrix[loss_matrix > 0] = 1
  count = np.count_nonzero(loss_matrix, axis =  1)
  count = count - np.ones(loss_matrix.shape[0])
  row_indices = W.shape[1] * np.arange(num_train)
  add_factor = np.add(row_indices, y)
  replace_factor = np.negative(count)
  np.put(loss_matrix, add_factor, replace_factor)
  dW = np.matmul(X.transpose(), loss_matrix)
  dW /= num_train
  dW[:,:] += 2 * reg * W[:,:]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
