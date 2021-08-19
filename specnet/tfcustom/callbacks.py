from keras.callbacks import Callback
from keras.models import Model
import numpy as np
from spec_net.utils.syntax import update_kwargs

class CustomCallback(Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()
    
    def get_layernames(self, layernames):
        if layernames is None:
            self.layernames = []
            for layer in self.model.layers:
                self.layernames.append(layer.name)
        elif isinstance(layernames, list):
            self.layernames = layernames
        else:
            self.layernames = list([layernames])

class CallbackTrigger:
    def __init__(self, threshold = 0.95, pruning_rate = 0.2, n_prune = 1, 
                 d_epochs = 20, d_epochs_max = 250, pruning_type = "exp", 
                 metric = "accuracy", loss = "cross_entropy", 
                 loss_ratio = 0.001, n_val_samples = 5, 
                 inverted_identity = True):
        """
        Parameter class for the trigger control of the feature selection
        callback.

        Parameters
        ----------
        threshold : float, optional
            Threshold value that has to be surpassed in order to trigger the
            update method. The default is 0.95.
        pruning_rate : float, optional
            Percentage of nodes being killed after the update method is 
            triggered. Is only necessary, if the 'reduction_type' is 
            'exponential'. The default is 0.2.
        n_prune : int, optional
            Number of nodes that are removed at each iteration. This option 
            will only be necessary, if 'reduction_type' is 'linear'. 
            The default is 1.
        d_epochs : int, optional
            Number of epochs that the training needs to fulfill all conditions
            in order to trigger the next kill. The default is 20.
        d_epochs_max : int, optional
            Maximum number of epochs that the training has not fulfilled all
            conditions for the next kill. Is also a stopping criterion. 
            The default is 250.
        pruning_type : str, optional
            Defines the type of reduction for the iterative node killing
            process. The two options are 'linear' and 'exp'. 
            The default is 'exp'.
        metric : str, optional
            Defines the metric that is monitored. The default is 'accuracy'.
        loss : str, optional
            Defines the loss function that is executed. 
            The default is 'cross_entropy'.
        n_val_samples : TYPE, optional
            Number of validation samples that are chosen for the evaluation of
            the most important features. The default is 5.

        Returns
        -------
        None.

        """    
    
        self.threshold = threshold
        self.pruning_rate = pruning_rate
        self.n_prune = n_prune
        self.d_epochs = d_epochs
        self.d_epochs_max = d_epochs_max
        self.pruning_type = pruning_type
        self.metric = metric
        self.loss = loss
        self.loss_ratio = loss_ratio
        self.n_val_samples = n_val_samples
        self.inverted_identity = inverted_identity
            
class FeatureSelection(CustomCallback):
    def __init__(self, layer, n_features = None, callback = None):
        """
        This class is a customized callback function to iteratively delete all 
        the unnecassary input nodes (n_in) of a 'LinearPass' layer. The number 
        of nodes is reduced successively with respect to the optimum of the 
        given metric. 
        
        The reduction or killing process is triggered whenever the threshold 
        is surpassed for a given interval (e.g. 90% accuracy for at least 15 
        epochs).
        
        The reduction function itself is an exponentiell approximation towards 
        the desired number of leftover nodes (n_features):
            
        n_features = n_in * (1 - p_kill) ** i
        
        At each step there is a fixed percentage (p_kill) of nodes with 
        irrelevant data that is to be deleted.

        Parameters
        ----------
        layer : str
            Name of layer with class type 'LinearPass'. Specifies the layer 
            where irrelevant nodes are being deleted.
        n_features : float, optional
            Number of leftover optimized relevant nodes. If 'None', the node 
            killing lasts until threshold cannot be surpassed anymore. The 
            default is None.
        

        Returns
        -------
        None.

        """
        super().__init__()
        
        self.layer = layer
        self.n_features = n_features
        
        if callback is None:
            self.trigger_params = CallbackTrigger()
        else:
            self.trigger_params = CallbackTrigger(**callback)
        
        #information variables
        self._log = None
     
    def on_epoch_end(self, epoch, logs = {}):
        #check whether the layer is suited for the callback: must be 'LinearPass' layer
        if self.model.get_layer(self.layer).__class__.__name__ == "LinearPass":
            layer = self.model.get_layer(self.layer)
        else:
            return
        
        if not self._log:
            self._log = {"kill_epochs": [0],
                         "index": [np.argwhere(layer.weights[0].numpy() == 1).squeeze()],
                         "weights": [np.array(layer.weights[0].numpy())],
                         "loss": [np.zeros(np.array(layer.weights[0].numpy()).shape)],
                         "loss_values": [],
                         "n_features": [int(np.sum(np.array(layer.weights[0].numpy())))],
                         "delta": 0,
                         "delta_kill": [0],
                         "criterion_delta_kill": False,
                         "criterion_n_features": False,
                         }
        
        #stopping criterion 1
        if self._log["n_features"][-1] <= self.n_features:
            self._log["criterion_n_features"] = True
        
        #stopping criterion 2
        if self._log["delta_kill"][-1] > self.trigger_params.d_epochs_max - 1:
            self._log["criterion_delta_kill"] = True
       
        if self._log["criterion_delta_kill"]:
            self.model.stop_training = True
            if not self._log["criterion_n_features"]:
                print(f"Epoch {epoch}:\n",
                      "\tThe optimizer did not converge. Adjust",
                      "model parameters and try again.\n",
                      "\tStopped training with", 
                      f"{self.trigger_params.metric} of {logs[self.trigger_params.metric]} using ",
                      f"{self._log['n_features'][-1]} nodes as input.")
                return
            else:
                print(f"Epoch {epoch}:\n",
                      f"\tStopped training with {self.trigger_params.metric}",
                      f"of {logs[self.trigger_params.metric]} using",
                      f"{self._log['n_features'][-1]} nodes as input.")
                return
        
        #executing weight update
        if not self._log["criterion_n_features"]:
            if self._log["delta"] >= self.trigger_params.d_epochs:
                weights = self.calculate_weights(np.array(self._log["weights"][-1]), 
                                                 epoch = epoch,
                                                 n_min = self.n_features)
                layer.set_weights([weights])
        
        if len(self._log["loss_values"]) < 2:
            self._log["loss_values"].append(logs[self.trigger_params.metric])
        
        #counting how often the criteria are met
        if self.trigger_params.metric == ("accuracy" or "val_accuracy"):
            if logs[self.trigger_params.metric] >= self.trigger_params.threshold:
                self._log["delta"] += 1
            else:
                self._log["delta"] = 0 
        elif self.trigger_params.metric == "loss" or "val_loss":
            try:
                ratio = (self._log["loss_values"][1] - logs[self.trigger_params.metric]) / (self._log["loss_values"][0] - logs[self.trigger_params.metric])
                if np.abs(ratio) <= self.trigger_params.loss_ratio:
                    self._log["delta"] += 1
                else:
                    self._log["delta"] = 0 
                    self._log["loss_values"][1] = logs[self.trigger_params.metric]
            except:
                pass
    
        #counting until kill criteria are met
        self._log["delta_kill"][-1] += 1
        
    def save_log(self, epoch, weights, loss):
        self._log["loss"].append(loss)
        self._log["kill_epochs"].append(epoch)
        self._log["index"].append(np.argwhere(weights == 1).squeeze())
        self._log["weights"].append(np.array(weights))
        self._log["n_features"].append(int(np.sum(weights)))
        self._log["delta_kill"].append(self._log["delta_kill"][-1])
    
    def get_validation_subset(self, n_samples = None):
        """
        Extracts an arbitrary subset from the validation data as an input for
        the feature selection algorithm.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples per class that are evaluated via Leave-one-out 
            Cross-validation (LOO) or inverted LOO, respectively. The default 
            is 5.

        Returns
        -------
        None.

        """
        data, labels = self.validation_data[0:2]
        n_classes = len(np.unique(labels))
        indices = []
        for cls in range(n_classes):
            idx = np.argwhere(labels[:,0] == cls)[:, 0]
            if n_samples is None:
                indices.append(idx)
            else:
                indices.append(idx[:n_samples])
        indices = np.concatenate(indices, axis = 0).flatten()
        return data[indices], labels[indices]
    
    def get_prediction_data(self, weights, inverted_identity = True):
        data, labels = self.get_validation_subset(self.trigger_params.n_val_samples)
        n_samples, n_features = data.shape
        n_remaining = int(np.sum(weights))
        n_classes = len(np.unique(labels))
        
        I = np.identity(len(weights))

        if inverted_identity:
            I = np.ones(I.shape) - I
        
        I = I * self._log["weights"][-1]
        
        prediction_data = np.zeros([n_samples * n_remaining, n_features])
        prediction_labels = np.zeros([n_samples * n_remaining, n_classes])
        
        m = 0
        for i in range(len(weights)):
            if weights[i] != 0:
                for n in range(n_samples):
                    prediction_data[m] = data[n] * I[i]
                    j = labels[n]
                    prediction_labels[m, j] = 1
                    m += 1
        return prediction_data, prediction_labels
      
    # CURRENTLY ONLY 1D MATRICES --> EXPAND FOR 2D MATRICES
    def calculate_weights(self, weights, epoch, n_min = None):
        """
        Class method to calculate new weight matrix for the 'LinearPass' layer.

        Parameters
        ----------
        weights : float
            Previously used weight matrix. Initiallized with all weights being 
            1.
        n_min : int
            Minimum number of nodes to be leftover. If 'None', the weights are 
            updated until the threshold condition cannot be met anymore.

        Returns
        -------
        weights : float
            Calculated new sparse weight matrix.

        """
        data, labels = self.get_validation_subset()
        n_classes = len(np.unique(labels))
        
        # getting new number of leftover nodes
        if self.trigger_params.pruning_type == "exp":
            n_features = int(len(weights) * (1 - self.trigger_params.pruning_rate) ** (len(self._log["kill_epochs"])))
        
        elif self.trigger_params.pruning_type == "linear":
            n_features = int(len(weights) - (len(self._log["kill_epochs"]) * self.n_kill))
        
        else:
            raise NameError(f"'{self.trigger_params.pruning_type}' is not valid as 'pruning_type'.")
        
        #securing minimum number of features
        if n_min is not None:
            if n_features <= n_min:
                n_features = n_min
        
        #get prediction data for the measure
        prediction_data, prediction_labels = self.get_prediction_data(weights, 
                                                                      inverted_identity = self.trigger_params.inverted_identity)
        
        P = prediction_labels 
        Q = self.model.predict(prediction_data)
        Q = np.where(Q > 1e-8, Q, 1e-8)
        
        # cross-entropy H measures uncertainty of possible outcomes and is the best loss metric for multiclass classifications
        if self.trigger_params.loss == "cross_entropy":
            H = -np.sum(P * np.log(Q), axis = -1)
            H_max = -np.log(1 / n_classes)
        
        elif self.loss == "entropy":
            H = -np.sum(Q * np.log(Q), axis = -1)
            H_max = -np.sum(1 / n_classes * np.log(1 / n_classes))
        
        #calculates mean of all entropies as loss
        H = np.array(np.array_split(H, np.sum(self._log["weights"][-1])))
        loss = np.mean(H, axis = 1)
        
        #update function: sets all weights given by idx to zero and updates log
        def update(epoch, weights, loss, idx):
            weights[idx] = 0
            # reset and log
            self.save_log(epoch, weights, loss)
            self._log["delta"] = 0
            self._log["delta_kill"][-1] = 0
            print(f"Epoch {epoch}:\n",
                  f"\tDeleted {int(len(weights) - np.sum(weights))} feature(s).\n",
                  f"\tLeft with {self._log['n_features'][-1]} feature(s).")
        
        #getting indices of most entropy
        mask = np.zeros(weights.shape).astype(bool)
        if self.trigger_params.inverted_identity:
            loss = np.amin(loss) * ~weights.astype(bool)[self._log["index"][-1]] + loss * weights[self._log["index"][-1]]
            idx = self._log["index"][-1][(np.argsort(loss)[:-n_features],)]
            mask[idx] = True
            mask = mask[self._log["index"][-1]]
            loss_max = np.amax(loss)
            loss_max_remain = np.amax(loss[~mask])
            loss_max_kill = np.amax(loss[mask])
            if (loss_max_remain - loss_max_kill) >= 0.1 * self.trigger_params.d_epochs / self._log["delta"] * loss_max:
                update(epoch, weights, loss, idx)
        
        else:
            loss = np.amax(loss) * ~weights.astype(bool)[self._log["index"][-1]] + loss * weights[self._log["index"][-1]]
            idx = self._log["index"][-1][np.argsort(loss)[n_features:]]
            mask[idx] = True
            mask = mask[self._log["index"][-1]]
            loss_min_remain = np.amin(loss[~mask])
            loss_min_kill = np.amin(loss[mask])
            if (loss_min_kill - loss_min_remain) >= 0.1 * self.delta / self._log["delta"] * H_max:
                update(epoch, weights, loss, idx)
        
        return weights