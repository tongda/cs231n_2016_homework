import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    number_of_classes_not_match_margin = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        number_of_classes_not_match_margin += 1
      
    dW[:,y[i]] -= number_of_classes_not_match_margin * X[i]
  
  # X: (500, 3073)
  # W, dW: (3073, 10)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W) # Q:why put 0.5 here??
  dW += reg * W # d(0.5 * reg * np.sum(W * W)) = 0.5 * reg * 2 * W
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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # X: (500, 3073)
  # W, dW: (3073, 10)
  scores = X.dot(W) # (500, 10)
  # print('scores[0]', scores[0])
  # print('scores[1]', scores[1])

  # print('y[0]', y[0])
  # print('y[1]', y[1])
  # print('scores[0][5]', scores[0][y[0]])
  # print('scores[y][0]', scores[:np.array([4])])
  N = X.shape[0]
  correct_class_socres = scores[np.arange(N), y] # (500,)
  diffs = (scores.T - correct_class_socres).T + 1
  diffs[np.arange(N), y] = 0  # clear diffs of correct classes
  margins = np.maximum(0, diffs) # (500, 10)
  
  loss = np.sum(margins) / N
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  dW_factors = np.zeros(diffs.shape) # (500, 10)
  dW_factors[diffs > 0] = 1 # for other non-correct classes
  dW_factors[np.arange(N), y] = - np.sum(dW_factors, axis = 1) # for correct classes

  dW = X.T.dot(dW_factors) # (3073, 10)
  dW /= N
  dW += reg * W # d(0.5 * reg * np.sum(W * W)) = 0.5 * reg * 2 * W 
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
