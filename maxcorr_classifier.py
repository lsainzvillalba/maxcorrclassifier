import numpy as np
from numpy import linalg
from sklearn.metrics import accuracy_score

def zero_mean_unit_length(vector):
  """
  transform vector to standardized version
  """
  assert vector.ndim == 1
  norm_vector = vector - vector.mean()
  length = linalg.norm(norm_vector)
  if length > 0:
      norm_vector /= length
  return norm_vector

class MaxCorrClassifier(object):
  def __init__(self, nr_neurons, class_labels, dtype=np.float32):
    self.nr_classes = len(class_labels)
    self.class_labels = class_labels
    self.dtype = dtype
    # for nr of classes, generate weights for nr_neurons
    self._unormalized_weights = np.zeros(
        (self.nr_classes, nr_neurons), dtype=dtype)
    # nr data points per class
    self.nr_points = np.zeros((self.nr_classes), dtype=dtype)
    
  def fit(self, X_train, y_train):
    """Weights for each neuron (feature) will be changed
    incrementally with learning from trials
    X- trials(rows) x neurons(columns)
    y- class labels for each trial in row
    """
    unormalized_weights = self._unormalized_weights
    nr_points = self.nr_points
    for obs, label in zip(X_train, y_train):
        idx_label = self.class_labels.index(label)
        nr_points[idx_label] += 1
        step_size = 1. / nr_points[idx_label]
        unormalized_weights[idx_label] *= (1 - step_size)
        unormalized_weights[idx_label] += step_size * obs
    self._unormalized_weights = unormalized_weights
    self.nr_points = nr_points
  
  def predict(self, X_test):
    """
    # for each datapoint (trial- row) to predict in X with shape (1,nr_neurons)
    # outputs corresponding prediction according to training with fit
    # nr neurons == nr_features
    """
    outputs = []
    for x in X_test:
        # compute correlation pearson coefficient with each of the classifier
        # vectors (rows in  self._unormalized_weights)
        norm_x = zero_mean_unit_length(x)
        all_coefs = []
        for classifier_vector in self._unormalized_weights:
            norm_weights = zero_mean_unit_length(classifier_vector)
            # compute pearson coefficient
            pearson_coef = np.dot(norm_x, norm_weights)
            all_coefs.append(pearson_coef)
        # select the class with highest coefficient as the predicted output
        coefs = np.array(all_coefs)
        label_idx = np.argmax(coefs)
        outputs.append(self.class_labels[label_idx])
    return outputs
    
  def score(self,X_test,y_test):
    """
    computes accuracy score for y prediction outputs computed
    for X_test, ´y_pred´ in comparison to ´y_test´ ground truth
    """
    y_pred = self.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score
