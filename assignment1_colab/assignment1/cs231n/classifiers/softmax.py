from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    dW = dW.T
    for i in range(num_train):
      scores = np.matmul(X[i], W)
      exp_scores = np.exp(scores)
      exp_sum = np.sum(exp_scores)
      exp_norm = (exp_scores / exp_sum).reshape(-1,1)
      loss += -scores[y[i]] + np.log(exp_sum)

      dW[y[i]] -= X[i]
      dW += np.kron(exp_norm,X[i])

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW = dW.T
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = np.matmul(X,W)
    loss -= np.sum(scores[range(num_train),y])
    exp_scores = np.exp(scores)
    loss += np.sum(np.log(np.sum(exp_scores,axis=1)))

    loss /= num_train
    loss += reg * np.sum(W * W)

    cnt = np.zeros_like(scores)
    cnt[np.arange(len(X)),y] = 1
    exp_scores = exp_scores.T
    exp_scores /= np.sum(exp_scores,axis=0)
    dW = -np.matmul(X.T,cnt) + (np.dot(exp_scores / np.sum(exp_scores,axis=0),X)).T

    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
