import numpy as np
from sklearn.metrics import confusion_matrix

class Metrics:
    def __init__(self, y_true, y_pred):
        self.classes = np.unique([y_true, y_pred])
        self.y_true, self.y_pred = y_true, y_pred
        self.confusion_matrix = self.get_confusion_matrix()
        self.n_samples = self.get_n_samples()
    
    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def get_precision(self):
        self.precision = dict()
        macro = 0
        weighted = 0
        for i, cls in enumerate(self.classes):
            prec = self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i])
            self.precision[f"{cls}"] = prec
            macro += prec
            weighted += self.n_samples[f"{cls}"] * prec
        self.precision["macro"] = macro / (i + 1)
        self.precision["weighted"] = weighted / self.n_samples["total"]
        return self.precision
            
    def get_recall(self):
        self.recall = dict()
        macro = 0
        weighted = 0
        for i, cls in enumerate(self.classes):
            rec = self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[:, i]) 
            self.recall[f"{cls}"] = rec
            macro += rec
            weighted += self.n_samples[f"{cls}"] * rec 
        self.recall["macro"] = macro / (i + 1)
        self.recall["weighted"] = weighted / self.n_samples["total"]
        return self.recall
    
    def get_f1_score(self):
        self.get_precision()
        self.get_recall()
        self.f1 = dict()
        macro = 0
        weighted = 0
        for i, cls in enumerate(self.classes):
            f = 2 * (self.precision[f"{cls}"] * self.recall[f"{cls}"]) / (self.precision[f"{cls}"] + self.recall[f"{cls}"])
            self.f1[f"{cls}"] = f
            macro += f
            weighted += self.n_samples[f"{cls}"] * f
        self.f1["macro"] = macro / (i + 1)
        self.f1["weighted"] = weighted / self.n_samples["total"]
        return self.f1
    
    def get_n_samples(self):
        self.n_samples = dict()
        for cls in self.classes:
            self.n_samples[f"{cls}"] = len(np.argwhere(self.y_true == cls))
        self.n_samples["total"] = len(self.y_true)
        return self.n_samples