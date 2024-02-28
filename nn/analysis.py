"""
feasel.nn.analysis
==================

Work to be done...
"""

import numpy as np

class FeaSelAnalysis:
  def __init__(self, feasel_model):
    self.model = feasel_model

  def get_mask(self):
    mask = self.model.callback.log.m[-1]
    return mask

  def empirical_masks(self, n):
    m_l = []

    for i in range(n):
      print(f'{i}. passage for the feature selection.')
      self.model.reset()
      if hasattr(self.model.callback, 'log'):
        self.model.callback.log = None
      self.model.train()
      m_l.append(self.get_mask())

    m_l = np.array(m_l)
    return m_l