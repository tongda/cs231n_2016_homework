import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0] # 500
    num_train = self.X_train.shape[0] # 5000
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        
        # testImg = X[i]
        # trainImg = self.X_train[j]
        
        # print(testImg)
        # print(trainImg)
        # print(testImg - trainImg)
        # print(np.square(testImg - trainImg))
        # print(np.sum(np.square(testImg - trainImg)))
        # print(np.sqrt(np.sum(np.square(testImg - trainImg))))

        dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        # it takes 30'' to run
        
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      
      # my first idea is extend the x[i] from (3072, ) to (3072, 5000) and calculate the distances from all the images in the train set
      # but it's quite slow, even slower than the two loop algorithm.
      # Here is my first code, I think the reason of slow is extending x[i], which leads to more calculation work 
      # dists[i, :] = np.sqrt(np.square(np.reshape(X[i], (X[i].shape[0], 1)) * np.ones((1, num_train)) - self.X_train.T).sum(axis=0))
      
      # then I found that numpy offers an ability of broadcasting which enables me to simplify my extend way. self.X_train - X[i].T is where
      # broadcasting happens.
      # dists[i, :] = np.sqrt(np.square(self.X_train - X[i].T).sum(axis=1))
      
      # X[i] is a vector, I found a vector's .T operation has no effect, which means if `v` is a vector `v == v.T` returns all true
      dists[i, :] = np.sqrt(np.square(self.X_train - X[i].T).sum(axis=1))
      
      if i % 100 == 0:
        print(i)
        
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    X_train_2 = self.X_train*self.X_train # (5000, 3072)
    X_train_2 = np.sum(X_train_2, axis = 1) # (5000, )    
    X_train_2_repeat = np.array([X_train_2]*X.shape[0]) # [X_train_2] => (1, 5000), [X_train_2]*X.shape[0] => (500, 5000)

    X_2 = X*X # (500, 3072)
    X_2 = np.sum(X_2, axis = 1) # (500,)
    X_2_repeat = np.array( [X_2]*self.X_train.shape[0]).transpose() # [X_2] => (1, 500), [X_2]*self.X_train.shape[0] => (5000, 500), .transpose() => (500, 5000)
    
    X_dot_X_train = X.dot(self.X_train.T) # matrix product => (500, 5000)
    
#     print X_train_2_repeat.shape
#     print X_2_repeat.shape
#     print X_dot_X_train.shape
#     
#     
#     print X_train_2_repeat ,"\n"
#     print X_2_repeat ,"\n"
#     print 2*X_dot_X_train
    dists = X_train_2_repeat + X_2_repeat - 2*X_dot_X_train # it follows a simple rule: (a - b)^2 = a*a + b*b - 2*a*b 
    dists = np.sqrt(dists)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    #print(y_pred.shape) => (500,) which is a 500 rows column vector 
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      # print(dists[i])
      # print(np.argsort(dists[i]))
      # print(np.argsort(dists[i])[:k])
      closest_y = np.argsort(dists[i])[:k]
      # print(closest_y)
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      # print(self.y_train[closest_y])
      # print(self.y_train[closest_y].shape)
      # print(np.bincount(self.y_train[closest_y]))
      # print(np.argmax(np.bincount(self.y_train[closest_y])))
      y_pred[i] = np.argmax(np.bincount(self.y_train[closest_y]))
      
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

