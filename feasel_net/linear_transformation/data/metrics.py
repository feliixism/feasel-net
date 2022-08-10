import numpy as np

# LOSSES:
def cross_entropy(P, Q):
  """
  The cross-entropy calculates the loss for a multi-class prediction Q compared
  to the target P.

  Parameters
  ----------
  P : ndarray
    Target values (one-hot encoded).
  Q : ndarray
    Predicted values.

  Returns
  -------
  H : ndarray
    Cross-entropy values.

  """
  H = -np.sum(P * np.log(Q), axis = -1)
  return H

def entropy(P, Q):
  """
  The entropy gives insights about the information richness of a prediction.

  Parameters
  ----------
  P : ndarray
    Target values (one-hot encoded).
  Q : ndarray
    Predicted values.

  Returns
  -------
  H : ndarray
    Entropy values.

  """
  H = -np.sum(Q * np.log(Q), axis = -1)
  return H

# TEST PERFORMANCE:
def get_information(y_pred, y_true):
  """
  Provides the information of True Positives (TN), True Negatives (TN), False
  Positives (FP) and False Negatives (FN) of a test.

  Parameters
  ----------
  y_pred : ndarray
    The predicted outcome.
  y_true : ndarray
    The target value.

  Returns
  -------
  information : tuple
    Tuple with TP, TN, FP and FN.

  """
  y_pred = np.array(y_pred.max(axis=1, keepdims=1) == y_pred).astype(int)

  classes = np.arange(y_true.shape[1])

  TP, TN, FP, FN = [], [], [], []

  for i in classes:
    pred = y_pred[:, i]
    true = y_true[:, i]

    TP.append(np.sum((pred==true) & (pred==1))) # classified True and True
    TN.append(np.sum((pred==true) & (pred==0))) # classified Wrong and Wrong
    FP.append(np.sum((pred!=true) & (pred==1))) # classified True and Wrong
    FN.append(np.sum((pred!=true) & (pred==0))) # classified Wrong and True

  information = np.array(TP), np.array(TN), np.array(FP), np.array(FN)
  return information

def sensitivity(y_pred, y_true):
  """
  The sensitivity (or recall) is a measure of how well the classifier
  identifies true negatives. A higher sensitivity will be desired, if every
  actually positive case shall be identified. Tests with a high sensitivity are
  used for those diagnosis that, if positive, would cause severe damages.

  TPR = TP / (TP + FN)

  Parameters
  ----------
  y_pred : ndarray
    The predicted outcome.
  y_true : ndarray
    The target value.

  Returns
  -------
  TPR : float
    The sensitivity (or recall or true positive rate TPR) of a classified set.

  """
  TP, TN, FP, FN = get_information(y_pred, y_true)

  TPR = TP / (TP + FN)

  return TPR

def specificity(y_pred, y_true):
  """
  The specificity (or selectivity) is a measure of how well the classifier
  identifies true negatives. It is a counterpart to the sensitivity and usually
  performs less good, if the sensitivity is high. A higher specificity will be
  desired, if every actually negative case shall be identified. Tests with a
  high specificity are applied if a positive diagnosis causes e.g. anxiety or
  stigma rather than doing something good.

  TNR = TN / ( TN + FP)

  Parameters
  ----------
  y_pred : ndarray
    The predicted outcome.
  y_true : ndarray
    The target value.

  Returns
  -------
  TNR : float
    The specificity (or selectivity or true negative rate TNR) of a classified
    set.

  """
  TP, TN, FP, FN = get_information(y_pred, y_true)

  TNR = TN / ( TN + FP)

  return TNR

def precision(y_pred, y_true):
  """
  Unlike its homonym from measurement technology, the precision does not
  estimate how accurate a measurement or production is. It rather estimates
  how good a classification performs within one objective (e.g. dog).

  PPV = TP / (TP + FP)

  Parameters
  ----------
  y_pred : ndarray
    The predicted outcome.
  y_true : ndarray
    The target value.

  Returns
  -------
  PPV : float
    The precision (or positive predictive value PPV) of a classified set.

  """
  TP, TN, FP, FN = get_information(y_pred, y_true)

  PPV = TP / (TP + FP)

  return PPV

def fall_out(y_pred, y_true):
  """
  The fall out is the ratio between the number of negative events wrongly
  categorized as positive and the total number of actual negative events.

  FPR = FP / (TP + TN)

  Parameters
  ----------
  y_pred : ndarray
    The predicted outcome.
  y_true : ndarray
    The target value.

  Returns
  -------
  FPR : float
    The fall-out (or false positive rate) of a classified set.

  """
  TP, TN, FP, FN = get_information(y_pred, y_true)

  FPR = FP / (FP + TN)

  return FPR

def f1_score(y_pred, y_true):
  """
  The F-score or F-measure is a measure of a test's accuracy, whereas the F1-
  score is the harmonic mean of precision and recall.

  F1 = 2TP / (2TP + FP + FN)

  Parameters
  ----------
  y_pred : ndarray
    The predicted outcome.
  y_true : ndarray
    The target value.

  Returns
  -------
  F1 : float
    The F1 score of a classified set.

  """
  TP, TN, FP, FN = get_information(y_pred, y_true)

  F1 = 2*TP / (2*TP + FP + FN)

  return F1

def accuracy(y_pred, y_true):
  """
  The accuracy of a classified set.

  ACC = (TP + TN) / (TP + TN + FP + FN)

  Parameters
  ----------
  y_pred : ndarray
    The predicted outcome.
  y_true : ndarray
    The target value.

  Returns
  -------
  ACC : float
    The accuracy of of a classified set.

  """
  TP, TN, FP, FN = get_information(y_pred, y_true)

  ACC = (TP + TN) / (TP + TN + FP + FN)

  return ACC

# probably does not make sense for multi-class problems:
def ROC(y_pred, y_true):
  """
  The ROC function calculates the true positive rate (TPR) and false positive
  rate (FPR) for multiple randomly weighted outputs. They can be interpolated
  later onto get the receiver operating characteristic curve (ROC).

  Parameters
  ----------
  y_pred : ndarray
    The predicted outcome.
  y_true : ndarray
    The target value.

  Returns
  -------
  ROC : tuple
    A tuple with the array for TPR and FPR averaged over all classes.

  """
  TPR, FPR = [], []
  sampling = 100
  n_classes = y_true.shape[1]
  step_size = (n_classes - 1) / (n_classes * sampling)
  for i in range(n_classes):
    tpr, fpr = [], []
    weights = np.ones(n_classes) * 1 / n_classes
    for j in range(sampling):
      weights = weights - step_size / 2
      weights[i] = weights[i] + 1.5 * step_size
      y = y_pred * weights
      y = np.array(y.max(axis=1, keepdims=1) == y).astype(int)
      tpr.append(sensitivity(y, y_true)[i])
      fpr.append(fall_out(y, y_true)[i])
    tpr, fpr = np.array(tpr), np.array(fpr)
    TPR.append(tpr)
    FPR.append(fpr)

  ROC = np.average(np.array(TPR), axis=0), np.average(np.array(FPR), axis=0)

  return ROC