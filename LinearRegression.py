import matplotlib.pyplot as plt
import numpy as np

class lr():
  """
  Linear regression using least squares method
  """

  def fit(self, x, y):
    """
    Fit model

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Training data
    y : array_like, shape (n_samples, )
        Target value

    Return
    ----------
    self : returns an instance of self
    """

    x2 = np.power(x, 2)
    xy = x * y.reshape((-1,1))
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x2)
    sum_x_2 = np.power(sum_x, 2)
    sum_xy = np.sum(xy)
    self.n_samples = np.count_nonzero(x)

    self.slope_ = ((self.n_samples * sum_xy) - (sum_x * sum_y)) / ((self.n_samples * sum_x2) - sum_x_2)
    self.intercept_ = (sum_y - (self.slope_ * sum_x)) / self.n_samples

    return self

  def predict(self, x):
    """
    Predict using the linear model  
  
    Parameters  
    ----------  
    X : array_like or spare matrix, shape (n_samples, n_features)  
        Sample  
  
    Returns  
    ----------  
    C : array, shape (n_samples, )
        Returns predicted value
    """

    self.y_pred = (self.slope_ * x) + self.intercept_

    return self.y_pred

  def visual(self, x, y, y_pred):
    plt.plot(x,y,'o',color='blue')
    plt.plot(x, y_pred, color='red', linewidth=2)
    plt.xlabel('X - feature')
    plt.ylabel('Y - label')
    plt.title('Regression Graph')
    plt.show()
