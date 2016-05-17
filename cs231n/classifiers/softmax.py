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

    num_train = X.shape[0]
    num_dim = X.shape[1]
    num_class = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        exps = np.exp(scores)
        exp_sum = np.sum(exps)
        loss += -scores[y[i]] + np.log(exp_sum)
        dW += X[i].reshape((num_dim, 1)).dot(exps.reshape((1, num_class))) / exp_sum
        dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

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
    num_class = W.shape[1]

    scores = X.dot(W)
    exp_scores = np.exp(scores)
    exp_sum = np.sum(exp_scores, axis=1)
    loss = np.mean(-scores[np.arange(num_train), y] + np.log(exp_sum)) + 0.5 * reg * np.sum(W * W)
    dW = X.T.dot(exp_scores / exp_sum.reshape((num_train, 1)))

    for i in range(num_class):
        dW[:, i] -= np.sum(X[y == i], axis=0).T

    dW /= num_train
    dW += reg * W

    return loss, dW
