import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  D = W.shape[0]
  for i in range(N):
    scores = X[i].dot(W) # (C, )
    scores -= np.max(scores)  # shift all scores to let them align to zero
    e_scores = np.exp(scores)  # (C, )
    probabilities = e_scores / np.sum(e_scores) # (C,)
    loss += -np.log(probabilities[y[i]])

    dscores = probabilities # (C, )
    dscores[y[i]] -= 1
    dW += X[i].reshape(D, 1).dot(dscores.reshape(1, C)) #(D, C)
  
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)

  dW /= N
  dW += reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # - W: A numpy array of shape (D, C) containing weights.
  # - X: A numpy array of shape (N, D) containing a minibatch of data.
  # - y: A numpy array of shape (N,)
  N = X.shape[0]

  scores = X.dot(W) # (N, C)
  scores -= np.max(scores)
  e_scores = np.exp(scores)
  probabilities = e_scores / np.sum(e_scores, axis = 1, keepdims = True)
  probabilities_on_correct_classes = probabilities[np.arange(N), y]
  loss = np.sum(-np.log(probabilities_on_correct_classes)) / N
  loss += 0.5 * reg * np.sum(W * W)

  dscores = probabilities # (N, C)
  dscores[np.arange(N), y] -= 1
  dW = X.T.dot(dscores) / N
  dW += reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

